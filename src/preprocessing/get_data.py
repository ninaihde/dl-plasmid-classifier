"""
This script moves all real data to the target folder (with the label in all filenames). Afterwards, duplicate reads are
removed from the real data which is followed by a splitting into tune and test data. Additionally, this script downloads
all references needed for simulation and saves them in the correct data format (.fasta). The content of this script is
separated from prepare_simulation.py because each user will have its own data resources and thus its own downloading
procedure needed to create the input data for prepare_simulation.py. After running this script, check_megaplasmids.py
can be executed to filter out invalid plasmids.
"""

import bz2
import click
import glob
import gzip
import json
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
from ont_fast5_api.conversion_tools.fast5_subset import Fast5Filter
from ont_fast5_api.fast5_interface import get_fast5_file
from zipfile import ZipFile


# chose 5 plasmids related to Campylobacter coli, Campylobacter jejuni, Leptospira interrogans & Conchiformibius steedae
REAL_TUNE_PLASMIDS = ['20220321_1207_MN24598_FAR91003', '20220419_1137_MN24598_FAR92750', 'l_interrogans', 'c_steedae']


def move_real_reads(real_ncbi_dir, original_dir, real_dir):
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
            shutil.copyfile(fast5_file, f'{real_dir}/{name}__{os.path.basename(fast5_file).split(".")[0]}__chr.fast5')

        for fast5_file in glob.glob(f'{real_ncbi_dir}/{name}_plasmid_fast5/*.fast5'):
            shutil.copyfile(fast5_file, f'{real_dir}/{name}__{os.path.basename(fast5_file).split(".")[0]}__plasmid.fast5')

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
        shutil.copyfile(file_path, f'{real_dir}/{unique_filename}')


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


def classify_real_refs(csv_file, genomes_dir):
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


def move_real_refs(genomes_dir, test_real_dir):
    for file in [f for f in glob.glob(f'{genomes_dir}/*.fasta')
                 if f.endswith('_plasmid.fasta') or f.endswith('_chromosome.fasta')]:
        if file.endswith('_chromosome.fasta'):
            # take same suffix as for RKI data which is already in test_real_dir
            shutil.copyfile(file, f'{test_real_dir}/{os.path.basename(file).replace("chromosome", "chr")}')
        else:
            shutil.copyfile(file, f'{test_real_dir}/{os.path.basename(file)}')


def get_files_per_id(input_dir):
    ids = dict()
    for file in glob.glob(f'{input_dir}/*.fast5'):
        with get_fast5_file(file, mode='r') as fast5_file:
            for read in fast5_file.get_reads():
                if read.read_id in ids:
                    ids[read.read_id].append(os.path.basename(file))
                else:
                    ids[read.read_id] = [os.path.basename(file)]

    with open(f'{input_dir}/real_ids_and_files.json', 'w') as json_file:
        json.dump(ids, json_file)


def get_duplicate_ids(input_dir):
    ids_and_files = json.load(open(f'{input_dir}/real_ids_and_files.json', 'r'))
    remove_per_file = dict()

    for r_id, files in ids_and_files.items():
        # Note: we checked that maximum number of files per ID is always 2 which refers to one plasmid and one chromosome read
        if len(files) > 1:
            if files[0] not in remove_per_file:
                remove_per_file[files[0]] = [r_id]
            else:
                remove_per_file[files[0]].append(r_id)

            if files[1] not in remove_per_file:
                remove_per_file[files[1]] = [r_id]
            else:
                remove_per_file[files[1]].append(r_id)

    with open(f'{input_dir}/remove_per_file.json', 'w') as f:
        json.dump(remove_per_file, f)


def fill_new_folder(input_dir, output_dir, batch_size, threads):
    remove_per_file = json.load(open(f'{input_dir}/remove_per_file.json', 'r'))
    for file in glob.glob(f'{input_dir}/*.fast5'):
        if os.path.basename(file) in remove_per_file:
            with get_fast5_file(file, mode='r') as f5:
                all_ids = f5.get_read_ids()
                remove_ids = remove_per_file[os.path.basename(file)]
                keep_ids = [r_id for r_id in all_ids if r_id not in remove_ids]
                keep_ids_df = pd.DataFrame({'read_id': keep_ids})
                keep_ids_df.to_csv(f'{output_dir}/{os.path.basename(file)[:-6]}_keep_ids.csv', index=False, sep='\t')
                filename_df = pd.DataFrame({'read_id': [file]})
                filename_df.to_csv(f'{output_dir}/{os.path.basename(file)[:-6]}_filename.csv', index=False, sep='\t')

                extractor = Fast5Filter(input_folder=input_dir, output_folder=output_dir,
                                        filename_base=f'{os.path.basename(file)[:-6]}__',
                                        read_list_file=f'{output_dir}/{os.path.basename(file)[:-6]}_keep_ids.csv',
                                        batch_size=batch_size, threads=threads, recursive=False, follow_symlinks=False,
                                        file_list_file=f'{output_dir}/{os.path.basename(file)[:-6]}_filename.csv')
                extractor.run_batch()
                os.remove(f'{output_dir}/{os.path.basename(file)[:-6]}_keep_ids.csv')
                os.remove(f'{output_dir}/{os.path.basename(file)[:-6]}_filename.csv')
        else:
            shutil.copy(file, f'{output_dir}/{os.path.basename(file)}')

    # store CSV with ground truth labels
    gt_labels = pd.DataFrame(columns=['Read ID', 'GT Label'])
    ids_and_files = json.load(open(f'{input_dir}/real_ids_and_files.json', 'r'))
    for r_id, files in ids_and_files.items():
        if len(files) == 1:
            label = 'plasmid' if 'plasmid' in files[0] else 'chr'
            gt_labels = pd.concat([gt_labels, pd.DataFrame([{'Read ID': r_id, 'GT Label': label}])], ignore_index=True)
    gt_labels.to_csv(f'{output_dir}/gt_labels_real.csv', index=False)


def move_chromosomes_by_ids(chr_ids, input_dir, output_dir, batch_size, threads):
    df = pd.DataFrame({'read_id': chr_ids})
    df.to_csv(f'{output_dir}/{os.path.basename(output_dir)}_chromosomes.csv', index=False, sep='\t')

    extractor = Fast5Filter(input_folder=input_dir, output_folder=output_dir, filename_base='chr__',
                            read_list_file=f'{output_dir}/{os.path.basename(output_dir)}_chromosomes.csv',
                            batch_size=batch_size, threads=threads, recursive=False, follow_symlinks=False,
                            file_list_file=None)
    extractor.run_batch()
    os.remove(f'{output_dir}/{os.path.basename(output_dir)}_chromosomes.csv')


def move_chromosomes(random_gen, chr_gt, percentage, input_dir, output_tune, output_test, batch_size, threads):
    # move chromosome reads to new real tune folder
    chr_ids = chr_gt['Read ID'].tolist()
    tune_chr_ids = random_gen.choice(chr_ids, size=int(len(chr_ids) * percentage), replace=False)
    move_chromosomes_by_ids(tune_chr_ids, input_dir, output_tune, batch_size, threads)

    # move chromosome reads to new real test folder
    test_df = chr_gt[~chr_gt['Read ID'].isin(tune_chr_ids)]
    test_chr_ids = test_df['Read ID'].tolist()
    move_chromosomes_by_ids(test_chr_ids, input_dir, output_test, batch_size, threads)


def move_plasmids(input_dir, output_tune, output_test):
    for plasmid_file in glob.glob(f'{input_dir}/*plasmid*.fast5'):
        # move plasmid files to new real tune folder
        if os.path.basename(plasmid_file).startswith(tuple(REAL_TUNE_PLASMIDS)):
            shutil.copyfile(plasmid_file, f'{output_tune}/{os.path.basename(plasmid_file)}')
        # move plasmid files to new real test folder
        else:
            shutil.copyfile(plasmid_file, f'{output_test}/{os.path.basename(plasmid_file)}')


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
@click.option('--real_ncbi_dir', '-ncbi', type=click.Path(exists=True), required=True,
              help='directory containing ZIP directories with new real .fast5 data from NCBI')
@click.option('--original_dir', '-rki', type=click.Path(exists=True), required=True,
              help='directory containing real .fast5 data by RKI')
@click.option('--genomes_dir', '-g', type=click.Path(exists=True), required=True,
              help='directory containing real .fasta data (RKI + NCBI)')
@click.option('--real_dir', '-real', type=click.Path(), required=True,
              help='directory path to real data used in testing and tuning later on')
@click.option('--ref_pos_dir', '-p', type=click.Path(), required=True,
              help='directory path to positive references used in simulation later on')
@click.option('--ref_neg_dir', '-n', type=click.Path(), required=True,
              help='directory path to negative references used in simulation later on')
@click.option('--csv_file', '-c', type=click.Path(exists=True), required=True,
              help='path to CSV file containing read IDs of the real NCBI data divided by class')
@click.option('--rds_file', '-r', type=click.Path(exists=True), required=True,
              help='path to RDS file of the negative references (from 2018)')
@click.option('--random_seed', '-s', default=42, help='seed for random operations')
@click.option('--batch_size', '-b', default=5000, help='number of reads per batch for fast5 creation')
@click.option('--threads', '-t', default=32, help='number of threads to use for fast5 creation')
@click.option('--percentage', '-p', default=0.3, help='percentage of reads to select for real tune data')
def main(real_ncbi_dir, original_dir, genomes_dir, real_dir, ref_pos_dir, ref_neg_dir, csv_file, rds_file,
         random_seed, batch_size, threads, percentage):
    random_gen = random.default_rng(random_seed)

    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
    if not os.path.exists(ref_pos_dir):
        os.makedirs(ref_pos_dir)
    if not os.path.exists(ref_neg_dir):
        os.makedirs(ref_neg_dir)

    # moving real data to scratch directory
    move_real_reads(real_ncbi_dir, original_dir, real_dir)  # means .fast5 files
    classify_real_refs(csv_file, genomes_dir)
    move_real_refs(genomes_dir, real_dir)  # means .fasta files

    # remove duplicate read IDs from real data
    get_files_per_id(real_dir)
    get_duplicate_ids(real_dir)
    real_dir_without_dupl = f'{real_dir}_without_dupl'
    if not os.path.exists(real_dir_without_dupl):
        os.makedirs(real_dir_without_dupl)
    fill_new_folder(real_dir, real_dir_without_dupl, batch_size, threads)

    # split real data into tune and test data
    test_dir = f'{os.path.dirname(real_dir_without_dupl)}/test_real'
    tune_dir = f'{os.path.dirname(real_dir_without_dupl)}/tune_real'
    real_labels = pd.read_csv(f'{real_dir_without_dupl}/gt_labels_real.csv')
    real_chr = real_labels[real_labels['GT Label'] == 'chr']
    move_chromosomes(random_gen, real_chr, percentage, real_dir_without_dupl, tune_dir, test_dir, batch_size, threads)
    move_plasmids(real_dir_without_dupl, tune_dir, test_dir)

    # download negative references
    download_ref_neg(ref_neg_dir, rds_file, random_gen)

    # downloading positive references
    download_ref_pos(ref_pos_dir)


if __name__ == '__main__':
    main()
