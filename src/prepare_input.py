"""
PREPROCESSING STEP 1/4
This script prepares all needed inputs for prepare_simulation.py. I.e., it moves all real test data to the respective
folders (with the label in all filenames). Additionally, it downloads all references needed for simulation and saves
them in the correct data format (.fasta). The content of this script is separated from prepare_simulation.py because
each researcher will have its own data resources and thus e.g. its own downloading procedure needed to create the input
data for prepare_simulation.py. After running this script, check_megaplasmids.py can be executed to filter out invalid
plasmids.
"""

import bz2
import click
import glob
import gzip
import os
import pandas as pd
import pyreadr
import re
import shutil
import tarfile
import urllib.error
import wget

from Bio import SeqIO
from datetime import datetime
from numpy import random
from zipfile import ZipFile


def move_real_test_reads(real_ncbi_dir, original_dir, test_dir):
    # move real FAST5 files from NCBI
    for zip_folder in glob.glob(f'{real_ncbi_dir}/*.zip'):
        name = os.path.basename(zip_folder).split('.')[0]

        # extract zipped fast5 directory from archive
        with ZipFile(zip_folder, 'r') as archive:
            for folder in archive.namelist():
                if folder.endswith('_fast5.tar.gz'):
                    archive.extract(folder, f'{real_ncbi_dir}')

        # correct inconsistent file naming
        if not os.path.exists(f'{real_ncbi_dir}/{name}/{name}_chr_fast5.tar.gz'):
            file = glob.glob(f'{real_ncbi_dir}/{name}/*chr*_fast5.tar.gz')[0]
            os.rename(file, f'{real_ncbi_dir}/{name}/{name}_chr_fast5.tar.gz')

        if not os.path.exists(f'{real_ncbi_dir}/{name}/{name}_plasmid_fast5.tar.gz'):
            file = glob.glob(f'{real_ncbi_dir}/{name}/*plasmid_fast5.tar.gz')[0]
            os.rename(file, f'{real_ncbi_dir}/{name}/{name}_plasmid_fast5.tar.gz')

        # extract files from zipped fast5 directory
        neg_folder = tarfile.open(f'{real_ncbi_dir}/{name}/{name}_chr_fast5.tar.gz')
        neg_folder.extractall(f'{real_ncbi_dir}')

        pos_folder = tarfile.open(f'{real_ncbi_dir}/{name}/{name}_plasmid_fast5.tar.gz')
        pos_folder.extractall(f'{real_ncbi_dir}')

        # correct inconsistent folder naming
        if not os.path.exists(f'{real_ncbi_dir}/{name}_chr_fast5'):
            file = glob.glob(f'{real_ncbi_dir}/*chr*_fast5')[0]
            os.rename(file, f'{real_ncbi_dir}/{name}_chr_fast5')
        if not os.path.exists(f'{real_ncbi_dir}/{name}_plasmid_fast5'):
            file = glob.glob(f'{real_ncbi_dir}/*plasmid_fast5')[0]
            os.rename(file, f'{real_ncbi_dir}/{name}_plasmid_fast5')

        # move files to real test directory -> new file name: name__batch__label
        for fast5_file in glob.glob(f'{real_ncbi_dir}/{name}_chr_fast5/*.fast5'):
            shutil.copyfile(fast5_file, f'{test_dir}/{name}__{os.path.basename(fast5_file).split(".")[0]}__chr.fast5')

        for fast5_file in glob.glob(f'{real_ncbi_dir}/{name}_plasmid_fast5/*.fast5'):
            shutil.copyfile(fast5_file, f'{test_dir}/{name}__{os.path.basename(fast5_file).split(".")[0]}__plasmid.fast5')

        # remove not needed directories
        shutil.rmtree(f'{real_ncbi_dir}/{name}')
        shutil.rmtree(f'{real_ncbi_dir}/{name}_chr_fast5')
        shutil.rmtree(f'{real_ncbi_dir}/{name}_plasmid_fast5')

    # move real FAST5 files from RKI runs
    for file_path in glob.glob(f'{original_dir}/*/*/*.fast5'):
        # add suffix to avoid overwriting files and to store labels -> run__batch__label
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


def get_completeness_nr(assembly_level):
    if assembly_level == 'Complete Genome':
        return 4
    elif assembly_level == 'Chromosome':
        return 3
    elif assembly_level == 'Scaffold':
        return 2
    elif assembly_level == 'Contig':
        return 1
    else:
        raise ValueError(f'Non-valid assembly level: {assembly_level}!')


def update_rds(genbank, path, rds_data, random_gen):
    species = rds_data[rds_data['ftp_path'] == path].iloc[0]['Species']
    species_genomes = genbank[genbank['organism_name'] == species].reset_index(drop=True)

    # get updated path
    if not species_genomes[species_genomes['refseq_category'] != 'na'].empty:
        print('Representative exists')
        updated_path = species_genomes[species_genomes['refseq_category'] != 'na'].iloc[0]['ftp_path']
    else:
        print('Representative does not exist')
        # add numerical genome completeness
        species_genomes['completeness_nr'] = species_genomes['assembly_level'].apply(lambda l: get_completeness_nr(l))
        # get most complete ones
        most_complete = species_genomes[species_genomes['completeness_nr'] == species_genomes['completeness_nr'].max()]
        # select randomly among most complete ones
        updated_path = most_complete.sample(n=1, random_state=random_gen).iloc[0]['ftp_path']

    # keep ftp-prefix for consistency and get assembly accession
    updated_path = updated_path.replace('https://', 'ftp://')
    updated_assembly_accession = "_".join(os.path.basename(updated_path).split('_')[:2])

    # set updated path and assembly accession
    print(f'Updated path for {species}: {updated_path}')
    rds_data.loc[rds_data['Species'] == species, ['ftp_path']] = updated_path
    rds_data.loc[rds_data['Species'] == species, ['assembly_accession']] = updated_assembly_accession

    return updated_path, rds_data


def download_ref_neg(ref_neg_dir, rds_file, random_gen):
    rds_data = pyreadr.read_r(rds_file)[None]
    rds_data['ftp_path'] = rds_data['ftp_path'].astype(str)
    ftp_paths = rds_data['ftp_path'].tolist()

    # get genbank data for potential failed downloads aka outdated ftp paths
    genbank_path = f'{ref_neg_dir}/genbank_{datetime.today().strftime("%Y_%m_%d")}.txt'
    if not any(fname.startswith('genbank_') for fname in os.listdir(ref_neg_dir)):
        wget.download('https://ftp.ncbi.nlm.nih.gov/genomes/genbank/assembly_summary_genbank.txt', out=genbank_path)
    genbank = pd.read_csv(genbank_path, sep='\t', skiprows=[0])
    genbank.rename(columns={'# assembly_accession': 'assembly_accession'}, inplace=True)

    successful_downloads = 0
    failed_downloads = 0
    for folder_path in ftp_paths:
        try:
            # try to download compressed reference
            wget.download(f'{folder_path}/{os.path.basename(folder_path)}_genomic.fna.gz',
                          out=f'{ref_neg_dir}/{os.path.basename(folder_path)}.fna.gz')
            successful_downloads += 1
        except urllib.error.URLError:
            # if download fails, update path and RDS data
            print(f'{os.path.basename(folder_path)}_genomic.fna.gz not found')
            failed_downloads += 1
            folder_path, rds_data = update_rds(genbank, folder_path, rds_data, random_gen)

            # download updated compressed reference
            wget.download(f'{folder_path}/{os.path.basename(folder_path)}_genomic.fna.gz',
                          out=f'{ref_neg_dir}/{os.path.basename(folder_path)}.fna.gz')

        # decompress and write all records to new .fasta file
        with gzip.open(f'{ref_neg_dir}/{os.path.basename(folder_path)}.fna.gz', 'rt') as f_in:
            with open(f'{ref_neg_dir}/{os.path.basename(folder_path)}.fasta', 'w') as f_out:
                for record in SeqIO.parse(f_in, 'fasta'):
                    r = SeqIO.write(record, f_out, 'fasta')
                    if r != 1:
                        print(f'Error while writing sequence {record.id} from genome {os.path.basename(folder_path)}')

        # remove compressed reference
        os.remove(f'{ref_neg_dir}/{os.path.basename(folder_path)}.fna.gz')

    # save updated RDS if at least one path was updated
    if failed_downloads > 0:
        pyreadr.write_rds(f'{rds_file[:-4]}_updated.rds', rds_data)

    print(f'Replaced {failed_downloads} of {successful_downloads + failed_downloads} URLs in RDS file')


def download_ref_pos(ref_pos_dir):
    wget.download('https://ccb-microbe.cs.uni-saarland.de/plsdb/plasmids/download/plsdb.fna.bz2',
                  out=f'{ref_pos_dir}/ref_pos.fna.bz2')

    c = 0
    with bz2.open(f'{ref_pos_dir}/ref_pos.fna.bz2', 'rt') as f_in:
        # write each sequence to a separate .fasta file
        for record in SeqIO.parse(f_in, 'fasta'):
            with open(f'{ref_pos_dir}/{record.id}.fasta', 'w') as f_out:
                c += 1
                r = SeqIO.write(record, f_out, 'fasta')
                if r != 1:
                    print(f'Error while writing sequence {record.id}')
    print(f'Wrote {c} .fasta files to {ref_pos_dir}')

    # remove one big compressed reference file
    os.remove(f'{ref_pos_dir}/ref_pos.fna.bz2')


@click.command()
@click.option('--real_ncbi_dir', '-b', type=click.Path(exists=True), required=True,
              help='directory containing ZIP directories with new real .fast5 data from NCBI')
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
@click.option('--rds_file', '-r', type=click.Path(exists=True), required=True,
              help='path to RDS file of the negative references (from 2018)')
@click.option('--random_seed', '-s', default=42, help='seed for random operations')
def main(real_ncbi_dir, original_dir, genomes_dir, test_real_dir, ref_pos_dir, ref_neg_dir, csv_file, rds_file,
         random_seed):
    if not os.path.exists(test_real_dir):
        os.makedirs(test_real_dir)
    if not os.path.exists(ref_pos_dir):
        os.makedirs(ref_pos_dir)
    if not os.path.exists(ref_neg_dir):
        os.makedirs(ref_neg_dir)

    print('Moving real test files to simulation directory...')
    move_real_test_reads(real_ncbi_dir, original_dir, test_real_dir)  # means .fast5 files
    classify_real_test_refs(csv_file, genomes_dir)
    move_real_test_refs(genomes_dir, test_real_dir)  # means .fasta files

    print('\nDownloading negative references...')
    random_gen = random.default_rng(random_seed)
    download_ref_neg(ref_neg_dir, rds_file, random_gen)

    print('\nDownloading positive references...')
    download_ref_pos(ref_pos_dir)


if __name__ == '__main__':
    main()
