"""
PREPROCESSING STEP 2/3
This script cleans the reference data of both classes. In addition, it splits the plasmid references it into training,
validation and test data based on the Jaccard similarity score, as we want to generalize our approach for plasmids.
Afterwards, prepare_tensors.py can be executed.
"""

import click
import glob
import os
import shutil
import sourmash
import time

from Bio import SeqIO
from collections import Counter
from numpy import random
from varname import nameof


def calculate_signatures(input_dir):
    signatures = list()

    for fasta_file in glob.glob(f'{input_dir}/*.fasta'):
        with open(fasta_file, 'r') as f:
            for idx, record in enumerate(SeqIO.parse(f, 'fasta')):
                mh = sourmash.MinHash(n=0, ksize=31, scaled=100)
                mh.add_sequence(str(record.seq), force=True)
                # add tuple (file_path, hash) to list
                signatures.append((fasta_file, mh))

    return signatures


def clean_neg_references(ref_neg_dir, ref_pos_dir, clean_sim_threshold, ref_neg_dir_cleaned):
    print('Cleaning negative references regarding positive references...')
    # TODO: unify cleaning functions?

    ref_neg_sigs = calculate_signatures(ref_neg_dir)
    ref_pos_sigs = calculate_signatures(ref_pos_dir)

    # collect chromosome references too similar to plasmid references
    refs_to_remove = list()
    for n_path, n_hash in ref_neg_sigs:
        for p_path, p_hash in ref_pos_sigs:
            if n_hash.similarity(p_hash, downsample=True) >= clean_sim_threshold:
                refs_to_remove.append(n_path)
                break

    print(f'{len(refs_to_remove)} of {len(ref_neg_sigs)} reads, i.e.{len(set(refs_to_remove))} of '
          f'{len(set([t[0] for t in ref_neg_sigs]))} files, will be removed.')

    # TODO: remove?
    counts = Counter(refs_to_remove)
    with open(f'{ref_neg_dir}/counts_removed_neg_refs.txt', 'w') as f:
        for path, deleted_reads in counts.most_common():
            overall_reads = [t[0] for t in ref_neg_sigs].count(path)
            f.write(f'{deleted_reads} / {overall_reads}: {path}\n')

    # move non-similar chromosome references to new directory
    ref_names_cleaned = [sig[0] for sig in ref_neg_sigs if sig[0] not in refs_to_remove]
    for filename in ref_names_cleaned:
        shutil.copyfile(filename, f'{ref_neg_dir_cleaned}/{os.path.basename(filename)}')


def split_randomly(paths, train_percentage, val_percentage, random_gen):
    train = random_gen.choice(paths, size=int(len(paths) * train_percentage), replace=False)
    val = random_gen.choice([p for p in paths if p not in train], size=int(len(paths) * val_percentage), replace=False)
    test = [p for p in paths if p not in train and p not in val]

    print(f'Training   dataset: {len(train)} / {len(paths)}')
    print(f'Validation dataset: {len(val)}   / {len(paths)}')
    print(f'Test       dataset: {len(test)}  / {len(paths)}')
    return train, val, test


def move_splitted_refs(ds_paths, sim_dir, ds_name, label):
    folder_path = f'{sim_dir}/{ds_name}_ref_{label}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for file_path in ds_paths:
        shutil.copyfile(file_path, f'{folder_path}/{os.path.basename(file_path)}')

    print(f'References belonging to {ds_name}_ref_{label} were successfully moved.')


def split_neg_references(ref_neg_dir_cleaned, train_percentage, val_percentage, random_gen, sim_dir):
    print('Randomly splitting negative references according to given percentages...')

    paths = glob.glob(f'{ref_neg_dir_cleaned}/*.fasta')
    train, val, test = split_randomly(paths, train_percentage, val_percentage, random_gen)

    for (ds_paths, ds_name) in [(train, nameof(train)), (val, nameof(val)), (test, nameof(test))]:
        move_splitted_refs(ds_paths, sim_dir, ds_name, 'neg')


def clean_pos_references(test_dir, ref_pos_dir, clean_sim_threshold, ref_pos_dir_cleaned):
    print('Cleaning positive references regarding real test data...')

    test_sigs = calculate_signatures(test_dir)
    ref_sigs = calculate_signatures(ref_pos_dir)

    # collect plasmid references too similar to test data
    refs_to_remove = list()
    for ref_path, ref_hash in ref_sigs:
        for test_path, test_hash in test_sigs:
            if ref_hash.similarity(test_hash, downsample=True) >= clean_sim_threshold:
                refs_to_remove.append(ref_path)
                break

    print(f'{len(refs_to_remove)} of {len(ref_sigs)} files will be removed.')

    # move non-similar references to new directory
    ref_names_cleaned = [sig[0] for sig in ref_sigs if sig[0] not in refs_to_remove]
    for filename in ref_names_cleaned:
        shutil.copyfile(filename, f'{ref_pos_dir_cleaned}/{os.path.basename(filename)}')


def split_by_similarity(signatures, split_sim_threshold, random_gen):
    train = list()
    test_val = list()
    skip_list = list()
    for idx in range(len(signatures)):
        # skip if plasmid is identically with earlier investigated plasmid
        if signatures[idx][0] in skip_list:
            continue

        is_sim = False
        for idx2 in range(idx + 1, len(signatures), 1):
            sim = signatures[idx][1].similarity(signatures[idx2][1], downsample=True)
            if sim > split_sim_threshold:
                is_sim = True
                if sim != 1.0 and signatures[idx2][0] not in train:
                    train.append(signatures[idx2][0])
                # if plasmids are identically, add second plasmid to skip list
                skip_list.append(signatures[idx2][0])
                break

        # if at least one plasmid (idx2) is similar to current plasmid (idx), add current plasmid to train
        if is_sim and signatures[idx][0] not in train:
            train.append(signatures[idx][0])
        # if no plasmid is similar to current plasmid, add current plasmid to test_val
        elif signatures[idx][0] not in train and signatures[idx][0] not in test_val:
            test_val.append(signatures[idx][0])

    # Cut test_val in half (to extract validation and test data paths)
    val = random_gen.choice(test_val, size=int(len(test_val) / 2), replace=False)
    test = [path for path in test_val if path not in val]

    # TODO: 623 references missing (34.388 - 21.668 - 6.048 - 6.049)?
    print(f'Training   dataset: {len(train)} / {len(signatures)}')
    print(f'Validation dataset: {len(val)}   / {len(signatures)}')
    print(f'Test       dataset: {len(test)}  / {len(signatures)}')
    return train, val, test


def split_pos_references(ref_pos_dir_cleaned, split_sim_threshold, random_gen, sim_dir):
    print('Splitting positive references according to sequence similarity score...')

    signatures = calculate_signatures(ref_pos_dir_cleaned)
    train, val, test = split_by_similarity(signatures, split_sim_threshold, random_gen)

    for (ds_paths, ds_name) in [(train, nameof(train)), (val, nameof(val)), (test, nameof(test))]:
        move_splitted_refs(ds_paths, sim_dir, ds_name, 'pos')


@click.command()
@click.option('--test_real_dir', '-td', type=click.Path(exists=True), required=True,
              help='directory containing real test data (this script uses .fasta references + classify.py uses .fast5')
@click.option('--ref_pos_dir', '-pd', type=click.Path(exists=True), required=True,
              help='directory containing reference genomes (.fasta) for positive class')
@click.option('--ref_neg_dir', '-nd', type=click.Path(exists=True), required=True,
              help='directory containing reference genomes (.fasta) for negative class')
@click.option('--sim_dir', '-sd', type=click.Path(exists=True), required=True,
              help='directory in which subfolders for splitted references will be created')
@click.option('--clean_sim_threshold', '-ct', default=0.9,
              help='threshold for sequence similarity score taken in cleaning procedure of references')
@click.option('--split_sim_threshold', '-st', default=0.4,
              help='threshold for sequence similarity score taken in splitting procedure of positive references')
@click.option('--random_seed', '-s', default=42, help='seed for random operations')
@click.option('--neg_train_pct', '-tp', default=0.8, help='splitting percentage for negative training references')
@click.option('--neg_val_pct', '-vp', default=0.1, help='splitting percentage for negative validation references')
def main(test_real_dir, ref_pos_dir, ref_neg_dir, sim_dir, clean_sim_threshold, split_sim_threshold, random_seed,
         neg_train_pct, neg_val_pct):
    start_time = time.time()
    random_gen = random.default_rng(random_seed)

    print('Prepare chromosomes...\n')
    ref_neg_dir_cleaned = f'{ref_neg_dir}_cleaned'
    if not os.path.exists(ref_neg_dir_cleaned):
        os.makedirs(ref_neg_dir_cleaned)

    # Note: clean negative before positive references to include potential real plasmids in positive references
    clean_neg_references(ref_neg_dir, ref_pos_dir, clean_sim_threshold, ref_neg_dir_cleaned)
    print(f'Runtime passed: {time.time() - start_time} seconds\n')

    split_neg_references(ref_neg_dir_cleaned, neg_train_pct, neg_val_pct, random_gen, sim_dir)
    print(f'Runtime passed: {time.time() - start_time} seconds\n')

    print('Prepare plasmids...\n')
    ref_pos_dir_cleaned = f'{ref_pos_dir}_cleaned'
    if not os.path.exists(ref_pos_dir_cleaned):
        os.makedirs(ref_pos_dir_cleaned)

    clean_pos_references(test_real_dir, ref_pos_dir, clean_sim_threshold, ref_pos_dir_cleaned)
    print(f'Runtime passed: {time.time() - start_time} seconds\n')

    split_pos_references(ref_pos_dir_cleaned, split_sim_threshold, random_gen, sim_dir)
    print(f'Overall runtime: {time.time() - start_time} seconds')


if __name__ == '__main__':
    main()
