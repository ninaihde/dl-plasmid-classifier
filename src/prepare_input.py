import bz2
import click
import glob
import gzip
import os
import pandas as pd
import pyreadr
import re
import shutil
import urllib.error
import wget

from Bio import SeqIO


def move_real_test_reads(original_dir, test_dir):
    # TODO: also move NEW real test .fast5 files

    for file_path in glob.glob(f'{original_dir}/*/*/*.fast5'):
        # add suffix to avoid overwriting files and to store labels -> runname__batchname__label
        file_path_splitted = re.split('[/.]+', file_path)
        unique_filename = f'{file_path_splitted[-4]}' \
                          f'__{file_path_splitted[-2]}' \
                          f'__{file_path_splitted[-3].split("_")[0]}.fast5'
        # copy file to new folder
        shutil.copyfile(file_path, f'{test_dir}/{unique_filename}')


def assign_records(file, f_in, helper_csv, f_out_plasmids, f_out_chr):
    for record in SeqIO.parse(f_in, 'fasta'):
        record_ids = [record.id, f'NZ_{record.id}', f'NC_{record.id}']
        if any(i in helper_csv['plasmids'].tolist() for i in record_ids):
            r = SeqIO.write(record, f_out_plasmids, 'fasta')
            if r != 1:
                print(f'Error while writing sequence {record.id} from genome {file}')
        elif any(i in helper_csv['chromosome'].tolist() for i in record_ids):
            r = SeqIO.write(record, f_out_chr, 'fasta')
            if r != 1:
                print(f'Error while writing sequence {record.id} from genome {file}')
        else:
            print(f'Could not assign sequence {record.id} from {os.path.basename(file)} to any ID in CSV file')


def classify_real_test_refs(csv_file, genomes_dir):
    helper_csv = pd.read_csv(csv_file)

    # create own row for each id in 'plasmids' column
    helper_csv = helper_csv.assign(plasmids=helper_csv.plasmids.str.split(';')).explode('plasmids')

    # iterate uncompressed files (exclude already classified files)
    for file in [f for f in glob.glob(f'{genomes_dir}/*.fasta')
                 if not f.endswith('_plasmid.fasta') and not f.endswith('_chromosome.fasta')]:
        with open(file, 'r') as f_in:
            f_out_plasmids = open(f'{file.replace(".fasta", "")}_plasmid.fasta', 'w')
            f_out_chr = open(f'{file.replace(".fasta", "")}_chromosome.fasta', 'w')
            assign_records(file, f_in, helper_csv, f_out_plasmids, f_out_chr)

    # iterate compressed files (exclude already classified files)
    for file in [f for f in glob.glob(f'{genomes_dir}/*.fasta.gz')
                 if not f.endswith('_plasmid.fasta') and not f.endswith('_chromosome.fasta')]:
        with gzip.open(file, 'rt') as f_in:
            f_out_plasmids = open(f'{file.replace(".fasta.gz", "")}_plasmid.fasta', 'w')
            f_out_chr = open(f'{file.replace(".fasta.gz", "")}_chromosome.fasta', 'w')
            assign_records(file, f_in, helper_csv, f_out_plasmids, f_out_chr)


def move_real_test_refs(genomes_dir, test_real_dir):
    for file in [f for f in glob.glob(f'{genomes_dir}/*.fasta')
                 if f.endswith('_plasmid.fasta') or f.endswith('_chromosome.fasta')]:
        if file.endswith('_chromosome.fasta'):
            # take same suffix as for RKI data which is already in test_real_dir
            shutil.copyfile(file, f'{test_real_dir}/{os.path.basename(file).replace("chromosome", "chr")}')
        else:
            shutil.copyfile(file, f'{test_real_dir}/{os.path.basename(file)}')


def download_ref_pos(ref_pos_dir):
    wget.download('https://ccb-microbe.cs.uni-saarland.de/plsdb/plasmids/download/plsdb.fna.bz2',
                  out=f'{ref_pos_dir}/ref_pos.fna.bz2')

    c = 0
    with bz2.open(f'{ref_pos_dir}/ref_pos.fna.bz2', 'rt') as f_in:
        # write each sequence to a separate .fasta file
        for record in SeqIO.parse(f_in, 'fasta'):
            with open(f'{ref_pos_dir}/plsdb_{record.id}.fasta', 'w') as f_out:
                c += 1
                r = SeqIO.write(record, f_out, 'fasta')
                if r != 1:
                    print(f'Error while writing sequence {record.id}')
    print(f'Wrote {c} .fasta files to {ref_pos_dir}')

    # remove one big compressed reference file
    os.remove(f'{ref_pos_dir}/ref_pos.fna.bz2')


def combine_rds_files(rds_file1, rds_file2, new_rds_path):
    rds1 = pyreadr.read_r(rds_file1)[None]
    rds2 = pyreadr.read_r(rds_file2)[None]

    # remove non-shared columns
    rds1 = rds1.drop(columns=['subset'])
    rds2 = rds2.drop(columns=['bioproject.orig', 'assembly_accession.orig'])

    new_rds_data = pd.concat([rds1, rds2], ignore_index=True)
    pyreadr.write_rds(new_rds_path, new_rds_data)


def download_ref_neg(ref_neg_dir, rds_file):
    # TODO: create .fasta per record?
    rds_data = pyreadr.read_r(rds_file)[None]  # get pandas DataFrame with [None]
    ftp_paths = rds_data['ftp_path'].tolist()

    successful_downloads = 0
    failed_downloads = 0
    for folder_path in ftp_paths:
        try:
            # download each compressed reference
            wget.download(f'{folder_path}/{os.path.basename(folder_path)}_genomic.fna.gz',
                          out=f'{ref_neg_dir}/{os.path.basename(folder_path)}.fna.gz')
            successful_downloads += 1
        except urllib.error.URLError as e:
            print(f'{e}: {os.path.basename(folder_path)}_genomic.fna.gz not found')
            failed_downloads += 1
            continue

        # decompress and write all records to new .fasta file
        with gzip.open(f'{ref_neg_dir}/{os.path.basename(folder_path)}.fna.gz', 'rt') as f_in:
            with open(f'{ref_neg_dir}/{os.path.basename(folder_path)}.fasta', 'w') as f_out:
                for record in SeqIO.parse(f_in, 'fasta'):
                    r = SeqIO.write(record, f_out, 'fasta')
                    if r != 1:
                        print(f'Error while writing sequence {record.id} from genome {os.path.basename(folder_path)}')

        # remove compressed reference
        os.remove(f'{ref_neg_dir}/{os.path.basename(folder_path)}.fna.gz')

    print(f'\nSuccessful downloads: {successful_downloads} \nFailed downloads: {failed_downloads}')


@click.command()
@click.option('--original_dir', '-o', type=click.Path(exists=True), required=True,
              help='directory containing real .fast5 data by RKI')
@click.option('--genomes_dir', '-g', type=click.Path(exists=True), required=True,
              help='directory containing real .fasta data (RKI + new)')
@click.option('--test_real_dir', '-t', type=click.Path(), required=True,
              help='directory path to real test data used in testing later on')
@click.option('--ref_pos_dir', '-p', type=click.Path(), required=True,
              help='directory path to positive references used in simulation later on')
@click.option('--ref_neg_dir', '-n', type=click.Path(), required=True,
              help='directory path to negative references used in simulation later on')
@click.option('--csv_file', '-c', type=click.Path(exists=True), required=True,
              help='path to CSV file containing read IDs of the new real .fasta data divided by class')
@click.option('--rds_file1', '-r1', type=click.Path(exists=True), required=True,
              help='path to RDS file of the negative references (from 2018)')
@click.option('--rds_file2', '-r2', type=click.Path(exists=True), required=True,
              help='path to RDS file of the negative references (from 2019)')
@click.option('--rds_file', '-r', type=click.Path(), required=True,
              help='path to final RDS file of the negative references')
def main(original_dir, genomes_dir, test_real_dir, ref_pos_dir, ref_neg_dir, csv_file, rds_file1, rds_file2, rds_file):
    if not os.path.exists(test_real_dir):
        os.makedirs(test_real_dir)
    if not os.path.exists(ref_pos_dir):
        os.makedirs(ref_pos_dir)
    if not os.path.exists(ref_neg_dir):
        os.makedirs(ref_neg_dir)

    move_real_test_reads(original_dir, test_real_dir)  # means .fast5 files
    classify_real_test_refs(csv_file, genomes_dir)
    move_real_test_refs(genomes_dir, test_real_dir)  # means .fasta files

    download_ref_pos(ref_pos_dir)
    combine_rds_files(rds_file1, rds_file2, rds_file)
    download_ref_neg(ref_neg_dir, rds_file)


if __name__ == '__main__':
    main()
