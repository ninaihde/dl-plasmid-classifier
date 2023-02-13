"""
This script executes the alignment-based minimap2 tool, including random read cutting (to be comparable to our approach)
and base-calling. "guppy", "minimap2" and "samtools" have to be installed beforehand.
"""

import click
import glob
import ont_fast5_api.fast5_read as f5read
import os
import pandas as pd
import shutil
import subprocess

from Bio import SeqIO
from numpy import random
from ont_fast5_api.fast5_interface import get_fast5_file
from ont_fast5_api.multi_fast5 import MultiFast5File
from tqdm import tqdm


def cut_reads(in_dir, out_dir, cutoff, min_seq_len, max_seq_len, random_seed):
    random_gen = random.default_rng(random_seed)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if min_seq_len >= max_seq_len:
        raise ValueError('The minimum sequence length must be smaller than the maximum sequence length!')

    ds_name = os.path.basename(in_dir)
    kept_reads = 0

    # create file for ground truth labels
    label_df = pd.DataFrame(columns=['Read ID', 'GT Label'])

    for input_file in tqdm(glob.glob(f'{in_dir}/*.fast5')):
        output_file = os.path.join(out_dir, os.path.basename(input_file))

        with get_fast5_file(input_file, mode='r') as f5_old, MultiFast5File(output_file, mode='w') as f5_new:
            for i, read in enumerate(f5_old.get_reads()):
                # get random sequence length per read
                seq_len = random_gen.integers(min_seq_len, max_seq_len + 1)

                # only parse reads that are long enough
                if len(read.handle[read.raw_dataset_name]) >= (cutoff + seq_len):
                    kept_reads += 1

                    # store ground truth labels for validation dataset
                    label = 'plasmid' if 'pos' in ds_name or 'plasmid' in ds_name else 'chr'
                    label_df = pd.concat(
                        [label_df, pd.DataFrame([{'Read ID': read.read_id, 'GT Label': label}])],
                        ignore_index=True)

                    # fill new fast5 file
                    read_name = f'read_{read.read_id}'
                    f5_new.handle.create_group(read_name)
                    output_group = f5_new.handle[read_name]
                    f5read.copy_attributes(read.handle.attrs, output_group)
                    for subgroup in read.handle:
                        if subgroup == read.raw_dataset_group_name:
                            raw_attrs = read.handle[read.raw_dataset_group_name].attrs
                            # remove cutoff and apply random sequence length
                            raw_data = read.handle[read.raw_dataset_name][cutoff:(cutoff + seq_len)]
                            output_read = f5_new.get_read(read.read_id)
                            output_read.add_raw_data(raw_data, raw_attrs)
                            new_attr = output_read.handle[read.raw_dataset_group_name].attrs
                            new_attr['duration'] = seq_len
                            continue
                        elif subgroup == 'channel_id':
                            output_group.copy(read.handle[subgroup], subgroup)
                            continue
                        else:
                            if read.run_id in f5_new.run_id_map:
                                # there may be a group to link to, but we must check it actually exists
                                hardlink_source = f'read_{f5_new.run_id_map[read.run_id]}/{subgroup}'
                                if hardlink_source in f5_new.handle:
                                    hardlink_dest = f'read_{read.read_id}/{subgroup}'
                                    f5_new.handle[hardlink_dest] = f5_new.handle[hardlink_source]
                                    continue
                            # if we couldn't hardlink to anything, we need to make the created group available for future reads
                            f5_new.run_id_map[read.run_id] = read.read_id
                        # if we haven't done a special-case copy, we can fall back on the default copy
                        output_group.copy(read.handle[subgroup], subgroup)

    print(f'Number of kept reads: {kept_reads}')

    # store ground truth labels of kept reads
    label_df.to_csv(f'{out_dir}/gt_{ds_name}_labels.csv', index=False)

    print(f'Finished cutting.')


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


def merge_references(in_dir, ds_identifier):
    merged_references = f'{in_dir}/{ds_identifier}_references.fasta'
    with open(merged_references, 'w') as f_out:
        for ref in [f for f in glob.glob(f'{in_dir}/*.fasta') if os.path.basename(f) != f'{ds_identifier}_references.fasta']:
            with open(ref, 'r') as f_in:
                for record in SeqIO.parse(f_in, 'fasta'):
                    r = SeqIO.write(record, f_out, 'fasta')
                    if r != 1:
                        print(f'Error while writing reference {record.id} from {ref}')

    return merged_references


def map_reads(reference_file, read_file, bam_file):
    print(f'Mapping {read_file} against reference {reference_file}...')
    minimap_cmd = ['minimap2', '-ax', 'map-ont', '--secondary=no', reference_file, read_file]
    minimap_output = subprocess.Popen(minimap_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    samsort_cmd = ['samtools', 'sort', '-O', 'BAM', '-o', bam_file]
    samsort_output = subprocess.Popen(samsort_cmd, stdin=minimap_output.stdout, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
    minimap_output.stdout.close()
    samsort_output.communicate()

    samindex_cmd = ['samtools', 'index', bam_file]
    subprocess.run(samindex_cmd)


@click.command()
@click.option('--read_dir', '-r', type=click.Path(exists=True), required=True,
              help='directory containing test reads (.fast5)')
@click.option('--ref_dir', '-f', type=click.Path(exists=True), required=True,
              help='directory containing references (.fasta)')
@click.option('--output_dir', '-o', type=click.Path(), required=True,
              help='directory containing mapping output files (.bam)')
@click.option('--guppy_dir', '-g', type=click.Path(exists=True), required=True,
              help='directory containing guppy base-caller')
@click.option('--cutoff', '-c', default=1000, help='cutoff the first c signals')
@click.option('--min_seq_len', '-min', default=2000, help='minimum number of signals per read')
@click.option('--max_seq_len', '-max', default=8000, help='maximum number of signals per read')
@click.option('--random_seed', '-s', default=42, help='seed for random sequence length generation')
def main(read_dir, ref_dir, output_dir, guppy_dir, cutoff, min_seq_len, max_seq_len, random_seed):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cut_dir = f'{read_dir}_max{int(max_seq_len/1000)}'
    cut_reads(read_dir, cut_dir, cutoff, min_seq_len, max_seq_len, random_seed)

    cut_pos_dir = f'{cut_dir}_pos'
    create_dir_per_class(cut_dir, cut_pos_dir, ['pos', 'plasmid'])
    cut_neg_dir = f'{cut_dir}_neg'
    create_dir_per_class(cut_dir, cut_neg_dir, ['neg', 'chr'])

    basecall_pos_dir = f'{cut_pos_dir}_basecalled'
    basecall_reads(guppy_dir, cut_pos_dir, basecall_pos_dir)
    basecall_neg_dir = f'{cut_neg_dir}_basecalled'
    basecall_reads(guppy_dir, cut_neg_dir, basecall_neg_dir)

    # NOTE: merging of reads and references has to be done only once - i.e., skip when testing different maximum lengths
    merge_reads(basecall_pos_dir)
    merge_reads(basecall_neg_dir)
    merged_references = merge_references(ref_dir, os.path.basename(read_dir))

    pos_bam = f'{output_dir}/pos_read_alignments.bam'
    map_reads(merged_references, f'{basecall_pos_dir}/pos.fastq', pos_bam)
    neg_bam = f'{output_dir}/neg_read_alignments.bam'
    map_reads(merged_references, f'{basecall_neg_dir}/neg.fastq', neg_bam)


if __name__ == '__main__':
    main()
