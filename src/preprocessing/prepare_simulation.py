"""
This script cleans the reference data of both classes. In addition, it splits the plasmid references it into training,
validation and test data based on the Jaccard similarity score, as we want to generalize our approach for plasmids. We
analyzed the produced removed_contigs.csv with check_contig_cleaning.ipynb but found the same megaplasmids as in
check_megaplasmids.py and no suspicious assemblies. Lastly, this script updates the RDS files needed for the simulation.
"""

import click
import glob
import os
import pandas as pd
import pyreadr
import shutil
import sourmash

from Bio import SeqIO
from numpy import random


def calculate_signatures(input_dir):
    signatures = list()

    for fasta_file in glob.glob(f'{input_dir}/*.fasta'):
        with open(fasta_file, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                mh = sourmash.MinHash(n=0, ksize=31, scaled=100)
                mh.add_sequence(str(record.seq), force=True)
                # add triple (file path, hash, record ID) to list
                signatures.append((fasta_file, mh, record.id))

    return signatures


def clean_neg_references(ref_neg_dir, ref_pos_dir, clean_sim_threshold, ref_neg_dir_cleaned):
    print('Cleaning negative references regarding positive references...')

    ref_neg_sigs = calculate_signatures(ref_neg_dir)
    ref_pos_sigs = calculate_signatures(ref_pos_dir)

    # collect chromosome references too similar to plasmid references
    removed_contigs = dict()
    for n_path, n_hash, n_id in ref_neg_sigs:
        for p_path, p_hash, _ in ref_pos_sigs:
            sim = n_hash.similarity(p_hash, downsample=True)
            if sim >= clean_sim_threshold:
                removed_contigs[(n_path, n_id)] = (sim, os.path.basename(p_path))
                break

    print(f'{len(removed_contigs)} of {len(ref_neg_sigs)} contigs will be removed')

    # move references to new directory by leaving out too similar contigs
    removed_contigs_df = pd.DataFrame(columns=['File', 'Contig', 'Length', 'Similarity', 'Plasmid'])
    for fasta_file in glob.glob(f'{ref_neg_dir}/*.fasta'):
        is_filled = False
        with open(fasta_file, 'r') as f_in, open(f'{ref_neg_dir_cleaned}/{os.path.basename(fasta_file)}', 'w') as f_out:
            for record in SeqIO.parse(f_in, 'fasta'):
                if (fasta_file, record.id) not in removed_contigs:
                    # write kept contigs to new file
                    is_filled = True
                    r = SeqIO.write(record, f_out, 'fasta')
                    if r != 1:
                        print(f'Error while writing contig {record.id} in file {fasta_file}')
                else:
                    # collect properties of removed contigs for later analysis
                    removed_contigs_df = pd.concat(
                        [removed_contigs_df, pd.DataFrame([{'Species': os.path.basename(fasta_file),
                                                            'Contig': record.id,
                                                            'Contig Length': len(record.seq),
                                                            'Similarity': removed_contigs[(fasta_file, record.id)][0],
                                                            'Plasmid': removed_contigs[(fasta_file, record.id)][1]}])],
                        ignore_index=True)
        if not is_filled:
            print(f'Warning: All contigs in {os.path.basename(fasta_file)} were removed.')
            os.remove(f'{ref_neg_dir_cleaned}/{os.path.basename(fasta_file)}')

    removed_contigs_df.to_csv(f'{ref_neg_dir}/removed_contigs.csv', index=False)


def clean_pos_references(test_dir, ref_pos_dir, clean_sim_threshold, ref_pos_dir_cleaned):
    print('Cleaning positive references regarding real test data...')

    test_sigs = calculate_signatures(test_dir)
    ref_sigs = calculate_signatures(ref_pos_dir)

    # collect plasmid references too similar to real test data
    refs_to_remove = list()
    for ref_path, ref_hash, _ in ref_sigs:
        for _, test_hash, _ in test_sigs:
            if ref_hash.similarity(test_hash, downsample=True) >= clean_sim_threshold:
                refs_to_remove.append(ref_path)
                break

    print(f'{len(refs_to_remove)} of {len(ref_sigs)} files will be removed.')

    # move non-similar references to new directory
    # assumes one contig per file, which is why whole file can be deleted
    for fasta_file in glob.glob(f'{ref_pos_dir}/*.fasta'):
        if fasta_file not in refs_to_remove:
            shutil.copyfile(fasta_file, f'{ref_pos_dir_cleaned}/{os.path.basename(fasta_file)}')


def split_by_similarity(signatures, split_sim_threshold, random_gen):
    train = list()
    test_val = list()
    skip_list = list()
    for idx in range(len(signatures)):
        # skip if plasmid is identically with earlier investigated plasmid
        if signatures[idx][0] in skip_list:
            continue

        is_sim = False
        for idx2 in range(idx + 1, len(signatures)):
            sim = signatures[idx][1].similarity(signatures[idx2][1], downsample=True)
            if sim >= split_sim_threshold:
                is_sim = True
                if sim != 1.0:
                    # if plasmids are not identical, add to train set
                    if signatures[idx2][0] not in train:
                        train.append(signatures[idx2][0])
                else:
                    # do not add identically plasmids twice (to avoid simulating too many reads for this plasmid)
                    skip_list.append(signatures[idx2][0])

        # if at least one plasmid (idx2) is similar to current plasmid (idx), add current plasmid to train
        if is_sim and signatures[idx][0] not in train:
            train.append(signatures[idx][0])
        # if no plasmid (idx2) is similar to current plasmid (idx), add current plasmid to test_val
        elif signatures[idx][0] not in train and signatures[idx][0] not in test_val:
            test_val.append(signatures[idx][0])

    # remove all plasmids from train set that should be skipped but were added to train set beforehand
    train = [p for p in train if p not in skip_list]

    # cut test_val in half (to extract validation and test data paths)
    val = random_gen.choice(test_val, size=int(len(test_val) / 2), replace=False)
    test = [path for path in test_val if path not in val]

    print(f'Skipped plasmids:   {len(skip_list)} / {len(signatures)}')
    print(f'Training dataset:   {len(train)}     / {len(signatures)}')
    print(f'Validation dataset: {len(val)}       / {len(signatures)}')
    print(f'Test dataset:       {len(test)}      / {len(signatures)}')
    return train, val, test


def split_pos_references(ref_pos_dir_cleaned, split_sim_threshold, random_gen, train_dir, val_dir, test_dir):
    print('Splitting positive references according to sequence similarity score...')

    signatures = calculate_signatures(ref_pos_dir_cleaned)
    train, val, test = split_by_similarity(signatures, split_sim_threshold, random_gen)

    for (ds_files, ds_dir) in [(train, train_dir), (val, val_dir), (test, test_dir)]:
        if not os.path.exists(ds_dir):
            os.makedirs(ds_dir)

        for file in ds_files:
            shutil.copyfile(file, f'{ds_dir}/{os.path.basename(file)}')

        print(f'References successfully moved to {os.path.basename(ds_dir)}.')


def append_refs(rds, ref_dir, ds_type, is_pos_class):
    for fasta_file in glob.glob(f'{ref_dir}/*.fasta'):
        rds = pd.concat([rds, pd.DataFrame([{'assembly_accession': os.path.basename(fasta_file)[:-6],
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
@click.option('--test_real', '-t', type=click.Path(exists=True), required=True,
              help='directory containing real test data (as .fasta references')
@click.option('--ref_pos', '-p', type=click.Path(exists=True), required=True,
              help='directory containing reference genomes (.fasta) for positive class')
@click.option('--ref_neg', '-n', type=click.Path(exists=True), required=True,
              help='directory containing reference genomes (.fasta) for negative class')
@click.option('--train_ref_pos', '-train', type=click.Path(), required=True,
              help='directory for train references of positive class')
@click.option('--val_ref_pos', '-val', type=click.Path(), required=True,
              help='directory for validation references of positive class')
@click.option('--test_ref_pos', '-test', type=click.Path(), required=True,
              help='directory for test references of positive class')
@click.option('--clean_sim_threshold', '-ct', default=0.9,
              help='threshold for sequence similarity score taken in cleaning procedures of references')
@click.option('--split_sim_threshold', '-st', default=0.4,
              help='threshold for sequence similarity score taken in splitting procedure of positive references')
@click.option('--random_seed', '-s', default=42, help='seed for random operations like splitting datasets')
@click.option('--plasmid_rds_path', '-rds_pos', help='filepath to new metadata file (.rds) of positive references',
              type=click.Path(), required=True)
@click.option('--chr_rds_path', '-rds_neg', help='filepath to metadata file (.rds) of negative references',
              type=click.Path(exists=True), required=True)
def main(test_real, ref_pos, ref_neg, train_ref_pos, val_ref_pos, test_ref_pos, clean_sim_threshold,
         split_sim_threshold, random_seed, plasmid_rds_path, chr_rds_path):
    random_gen = random.default_rng(random_seed)

    print('Preparing chromosome references...\n')
    ref_neg_cleaned = f'{ref_neg}_cleaned'
    if not os.path.exists(ref_neg_cleaned):
        os.makedirs(ref_neg_cleaned)

    # remove plasmids from chromosome references
    clean_neg_references(ref_neg, ref_pos, clean_sim_threshold, ref_neg_cleaned)

    # update RDS file of negative class
    chr_rds_data = adjust_rds_file(chr_rds_path)
    pyreadr.write_rds(f'{os.path.dirname(chr_rds_path)}/metadata_neg_ref.rds', chr_rds_data)

    print('Preparing plasmid references...\n')
    ref_pos_cleaned = f'{ref_pos}_cleaned'
    if not os.path.exists(ref_pos_cleaned):
        os.makedirs(ref_pos_cleaned)

    # clean and split plasmid references
    clean_pos_references(test_real, ref_pos, clean_sim_threshold, ref_pos_cleaned)
    split_pos_references(ref_pos_cleaned, split_sim_threshold, random_gen, train_ref_pos, val_ref_pos, test_ref_pos)

    # create RDS file for positive class
    plasmid_rds_data = create_rds_file(train_ref_pos, val_ref_pos, test_ref_pos)
    pyreadr.write_rds(plasmid_rds_path, plasmid_rds_data)


if __name__ == '__main__':
    main()
