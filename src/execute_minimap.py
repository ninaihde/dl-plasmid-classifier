"""
This script executes base-calling with Guppy, followed by the alignment-based minimap2 tool. Thus, "guppy", "minimap2"
and "samtools" have to be installed beforehand.
"""

import click
import glob
import os
import shutil
import subprocess

from Bio import SeqIO


def create_dir_per_class(in_dir, out_dir, class_synonyms):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for file in [f for f in glob.glob(f'{in_dir}/*.fast5') if any(c in f for c in class_synonyms)]:
        shutil.copyfile(file, f'{out_dir}/{os.path.basename(file)}')

    print(f'Finished moving cut files of {class_synonyms[0]} class.')


def basecall_reads(guppy_dir, in_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    guppy_cmd = [f"{guppy_dir}/bin/guppy_basecaller", "--config", f"{guppy_dir}/data/dna_r9.4.1_450bps_fast.cfg",
                 "--num_callers", "2", "--gpu_runners_per_device", "4", "--chunks_per_runner", "256", "--chunk_size",
                 "500", "--device", "cuda:all", "--trim_adapters", "-r", "--trim_primers", "-i", in_dir, "-s", out_dir]
    subprocess.run(guppy_cmd)


def merge_reads(in_dir):
    output_file = f'{in_dir}/{"neg" if "neg" in in_dir else "pos"}.fastq'
    with open(output_file, 'w') as f_out:
        for file in glob.glob(f'{in_dir}/fail/*.fastq') and glob.glob(f'{in_dir}/pass/*.fastq'):
            with open(file, 'r') as f_in:
                for i, record in enumerate(SeqIO.parse(f_in, 'fastq')):
                    r = SeqIO.write(record, f_out, 'fastq')
                    if r != 1:
                        print(f'Error while writing read {record.id} from {file}')

    print(f'Finished merging reads.')


def write_references(in_dir, out_fasta, out_txt):
    for ref in glob.glob(f'{in_dir}/*.fasta'):
        with open(ref, 'r') as f_in:
            for record in SeqIO.parse(f_in, 'fasta'):
                out_txt.write(f'{record.id}\n')
                r = SeqIO.write(record, out_fasta, 'fasta')
                if r != 1:
                    print(f'Error while writing reference {record.id} from {ref}')


def merge_references(ref_pos_dir, ref_neg_dir, out_dir):
    out_filename = f'{out_dir}/all_references.fasta'
    merged_references = open(out_filename, 'w')
    pos_ref_names = open(f'{out_dir}/pos_references.txt', 'w')
    neg_ref_names = open(f'{out_dir}/neg_references.txt', 'w')

    write_references(ref_pos_dir, merged_references, pos_ref_names)
    write_references(ref_neg_dir, merged_references, neg_ref_names)

    return out_filename


def map_reads(reference_file, read_file, bam_file):
    print(f'Mapping {read_file} against reference {reference_file}...')
    minimap_cmd = ['minimap2', '--secondary=no', '-I', '16G', '-ax', 'map-ont', reference_file, read_file]
    minimap_output = subprocess.Popen(minimap_cmd, stdout=subprocess.PIPE)

    samsort_cmd = ['samtools', 'sort', '-O', 'BAM', '-o', bam_file]
    samsort_output = subprocess.Popen(samsort_cmd, stdin=minimap_output.stdout)

    minimap_output.stdout.close()
    samsort_output.communicate()

    samindex_cmd = ['samtools', 'index', bam_file]
    subprocess.run(samindex_cmd)


@click.command()
@click.option('--read_dir', '-i', type=click.Path(exists=True), required=True,
              help='directory containing test reads (.fast5)')
@click.option('--ref_pos_dir', '-rp', type=click.Path(exists=True), required=True,
              help='directory containing positive training references after cleaning (.fasta)')
@click.option('--ref_neg_dir', '-rn', type=click.Path(exists=True), required=True,
              help='directory containing negative references after cleaning (.fasta)')
@click.option('--output_dir', '-o', type=click.Path(), required=True,
              help='output directory for BAM files and merged references')
@click.option('--guppy_dir', '-g', type=click.Path(exists=True), required=True,
              help='directory containing guppy base-caller')
def main(read_dir, ref_pos_dir, ref_neg_dir, output_dir, guppy_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    read_dir_pos = f'{read_dir}_pos'
    create_dir_per_class(read_dir, read_dir_pos, ['pos', 'plasmid'])
    read_dir_neg = f'{read_dir}_neg'
    create_dir_per_class(read_dir, read_dir_neg, ['neg', 'chr'])

    basecall_dir_pos = f'{read_dir_pos}_basecalled'
    basecall_reads(guppy_dir, read_dir_pos, basecall_dir_pos)
    basecall_dir_neg = f'{read_dir_neg}_basecalled'
    basecall_reads(guppy_dir, read_dir_neg, basecall_dir_neg)

    merge_reads(basecall_dir_pos)
    merge_reads(basecall_dir_neg)
    merged_references = merge_references(ref_pos_dir, ref_neg_dir, output_dir)

    pos_bam = f'{output_dir}/pos_read_alignments.bam'
    map_reads(merged_references, f'{basecall_dir_pos}/pos.fastq', pos_bam)
    neg_bam = f'{output_dir}/neg_read_alignments.bam'
    map_reads(merged_references, f'{basecall_dir_neg}/neg.fastq', neg_bam)


if __name__ == '__main__':
    main()
