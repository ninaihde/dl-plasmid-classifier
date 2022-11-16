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


def create_rds_file(fasta_dir, is_pos_class):
    # assembly_accession = ID, fold1 = train/val/test, Pathogenic = whether it is positive class
    rds = pd.DataFrame(columns=['assembly_accession', 'fold1', 'Pathogenic'])
    for fasta_file in glob.glob(f'{fasta_dir}/*.fasta'):
        rds = rds.append({'assembly_accession': fasta_file.split(os.path.sep)[-1].split('.')[-2],
                          'fold1': 'train',
                          'Pathogenic': is_pos_class},
                         ignore_index=True)

    return rds


def clean_rds_file(ref_neg_cleaned, chr_rds_path):
    whole_rds = pyreadr.read_r(chr_rds_path)[None]
    all_ids = whole_rds['assembly_accession'].tolist()
    kept_ids = list()

    for fasta_file in glob.glob(f'{ref_neg_cleaned}/*.fasta'):
        for i in all_ids:
            if i in os.path.basename(fasta_file):
                kept_ids.append(i)
            break

    cleaned_rds = whole_rds[whole_rds['assembly_accession'].isin(kept_ids)]
    pyreadr.write_rds(f'{chr_rds_path.split(".")[-2]}_cleaned.rds', cleaned_rds)


@click.command()
@click.option('--ref_neg_cleaned', '-rn', help='cleaned directory of files to be simulated for negative class',
              type=click.Path(exists=True), required=True)
@click.option('--ref_pos_cleaned', '-rp', help='cleaned directory of files to be simulated for positive class',
              type=click.Path(exists=True), required=True)
@click.option('--min_seq_len', '-min', help='minimum sequence length (in signals)', default=2000)
@click.option('--max_seq_len', '-max', help='maximum sequence length (in signals)', default=8000)
@click.option('--coverage', '-c', help='average coverage', default=2)
@click.option('--plasmid_rds_path', '-pr', help='filepath to new RDS file of positive references', type=click.Path(),
              required=True)
@click.option('--chr_rds_path', '-cr', help='filepath to RDS file of negative references', type=click.Path(exists=True),
              required=True)
def main(ref_neg_cleaned, ref_pos_cleaned, min_seq_len, max_seq_len, coverage, plasmid_rds_path, chr_rds_path):
    # calculate number of reads per class
    plasmid_genome_length = get_genome_length(ref_pos_cleaned)
    read_length = get_read_length(min_seq_len, max_seq_len)
    reads_per_p = get_n_reads(coverage, plasmid_genome_length, read_length)
    print(f'Number of reads per plasmid: {reads_per_p}')

    chr_genome_length = get_genome_length(ref_neg_cleaned)
    reads_per_c = get_n_reads(coverage, chr_genome_length, read_length)
    print(f'Number of reads per chromosome: {reads_per_c}')

    # check whether chromosome coverage isn't too low when taking same number of reads as for plasmids
    chr_coverage = get_coverage(reads_per_p, read_length, chr_genome_length)
    print(f'Coverage of chromosomes with same number of reads as for plasmids: {round(chr_coverage, 4)}')

    # create RDS file for positive class
    # TODO: take care of 623 non-splitted files?
    plasmid_rds_data = create_rds_file(ref_pos_cleaned, True)
    pyreadr.write_rds(plasmid_rds_path, plasmid_rds_data)

    # remove entries from RDS file for negative class that were removed during similarity cleaning
    clean_rds_file(ref_neg_cleaned, chr_rds_path)


if __name__ == '__main__':
    main()
