"""
This script performs all steps required to be able to execute Simulate_{class}.py. I.e., it calculates the number of
reads per dataset of the positive class. Additionally, it checks which chromosome coverage fits best and thus which read
number can be taken for the simulation with negative references. Lastly, the RDS file for plasmids is created and the
one for chromosomes is cleaned.
"""

import click
import glob
import os
import pandas as pd
import pyreadr

from Bio import SeqIO

BASES_PER_SEC = 450  # assuming 4000 signals per second = 450 bases
BASES_PER_1K_SIGNALS = BASES_PER_SEC / 4  # 1000 signals = 112.5 bases


def get_genome_length(fasta_dir):
    genome_len = 0
    for fasta_file in glob.glob(f'{fasta_dir}/*.fasta'):
        with open(fasta_file, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                genome_len += len(record.seq)

    return genome_len


def get_read_length(min_seq_len, max_seq_len):
    # assuming normal distribution
    avg_len = (min_seq_len + max_seq_len) / 2  # e.g. 5k signals

    return int((avg_len / 1000) * BASES_PER_1K_SIGNALS)  # e.g. 5 * 112.5 bases


# Formula from https://en.m.wikipedia.org/wiki/Coverage_(genetics)
def get_n_reads(coverage, genome_length, read_length):
    return int(coverage * genome_length / read_length)


def get_coverage(n_reads, read_length, genome_length):
    return n_reads * read_length / genome_length


def append_refs(rds, ref_dir, ds_type, is_pos_class):
    for fasta_file in glob.glob(f'{ref_dir}/*.fasta'):
        rds = pd.concat([rds, pd.DataFrame([{'assembly_accession': fasta_file.split(os.path.sep)[-1].split('.')[-2],
                                             'fold1': ds_type,
                                             'Pathogenic': is_pos_class}])],
                        ignore_index=True)
    return rds


def create_rds_file(train_ref_pos, val_ref_pos, test_ref_pos):
    rds = pd.DataFrame(columns=['assembly_accession', 'fold1', 'Pathogenic'])
    rds = append_refs(rds, train_ref_pos, 'train', True)
    rds = append_refs(rds, val_ref_pos, 'val', True)
    rds = append_refs(rds, test_ref_pos, 'test', True)

    return rds


def adjust_rds_file(chr_rds_path):
    rds = pyreadr.read_r(chr_rds_path)[None]
    rds['fold1'] = 'train'
    rds['Pathogenic'] = False

    return rds


@click.command()
@click.option('--ref_neg_cleaned', '-neg', help='directory of all files to be simulated for negative class',
              type=click.Path(exists=True), required=True)
@click.option('--train_ref_pos', '-train', help='directory of train files to be simulated for positive class',
              type=click.Path(exists=True), required=True)
@click.option('--val_ref_pos', '-val', help='directory of validation files to be simulated for positive class',
              type=click.Path(exists=True), required=True)
@click.option('--test_ref_pos', '-test', help='directory of test files to be simulated for positive class',
              type=click.Path(exists=True), required=True)
@click.option('--min_seq_len', '-min', help='minimum sequence length (in signals)', default=2000)
@click.option('--max_seq_len', '-max', help='maximum sequence length (in signals)', default=8000)
@click.option('--coverage', '-c', help='average coverage', default=2)
@click.option('--plasmid_rds_path', '-rds_pos', help='filepath to new metadata file (.rds) of positive references',
              type=click.Path(), required=True)
@click.option('--chr_rds_path', '-rds_neg', help='filepath to metadata file (.rds) of negative references',
              type=click.Path(exists=True), required=True)
def main(ref_neg_cleaned, train_ref_pos, val_ref_pos, test_ref_pos, min_seq_len, max_seq_len,
         coverage, plasmid_rds_path, chr_rds_path):
    # 1. calculate number of reads per plasmid dataset
    plasmid_genome_length = get_genome_length(train_ref_pos)
    read_length = get_read_length(min_seq_len, max_seq_len)
    reads_per_p_train = get_n_reads(coverage, plasmid_genome_length, read_length)
    print(f'Number of reads per training plasmid: {reads_per_p_train}\n')

    plasmid_genome_length = get_genome_length(val_ref_pos)
    reads_per_p_val = get_n_reads(coverage, plasmid_genome_length, read_length)
    print(f'Number of reads per validation plasmid: {reads_per_p_val}\n')

    plasmid_genome_length = get_genome_length(test_ref_pos)
    reads_per_p_test = get_n_reads(coverage, plasmid_genome_length, read_length)
    print(f'Number of reads per testing plasmid: {reads_per_p_test}\n')

    # 2. check which read number can be taken for negative class
    chr_genome_length = get_genome_length(ref_neg_cleaned)
    reads_per_c = get_n_reads(coverage, chr_genome_length, read_length)
    print(f'Number of reads per chromosome (Coverage = {coverage}): {reads_per_c}\n')

    summed_read_number_plasmids = reads_per_p_train + reads_per_p_val + reads_per_p_test
    # get coverage of chromosomes with same number of reads as for all plasmids
    chr_coverage = get_coverage(summed_read_number_plasmids, read_length, chr_genome_length)
    print(f'Number of reads per chromosome (Coverage = {round(chr_coverage, 4)}): {summed_read_number_plasmids}\n')

    # 3. create RDS file for positive class
    plasmid_rds_data = create_rds_file(train_ref_pos, val_ref_pos, test_ref_pos)
    pyreadr.write_rds(plasmid_rds_path, plasmid_rds_data)

    # 4. update RDS file of negative class
    chr_rds_data = adjust_rds_file(chr_rds_path)
    pyreadr.write_rds(f'{os.path.dirname(chr_rds_path)}/metadata_neg_ref.rds', chr_rds_data)


if __name__ == '__main__':
    main()
