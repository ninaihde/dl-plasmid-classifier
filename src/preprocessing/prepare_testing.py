"""
This script eliminates duplicate read IDs from the real test data by choosing a random class per duplicate read ID. In
addition, it performs cutting of the simulated and real test reads with respect to the maximum sequence length.
"""

import click
import glob
import json
import ont_fast5_api.fast5_read as f5read
import os
import pandas as pd
import shutil

from numpy import random
from ont_fast5_api.conversion_tools.fast5_subset import Fast5Filter
from ont_fast5_api.fast5_interface import get_fast5_file
from ont_fast5_api.multi_fast5 import MultiFast5File
from tqdm import tqdm


PLASMID_LABEL = 0
CHR_LABEL = 1


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


def get_duplicate_ids(input_dir, random_gen):
    ids_and_files = json.load(open(f'{input_dir}/real_ids_and_files.json', 'r'))
    duplicate_ids = pd.DataFrame()

    for r_id, files in ids_and_files.items():
        # Note: we manually checked that maximum number of files per ID is 2 which always refers to one plasmid and one chromosome
        if len(files) > 1:
            if '__chr' in files[0]:
                chr_file = files[0]
                plasmid_file = files[1]
            else:
                chr_file = files[1]
                plasmid_file = files[0]

            duplicate_ids = pd.concat([duplicate_ids,
                                       pd.DataFrame([{'Read ID': r_id,
                                                      'Chromosome File': chr_file,
                                                      'Plasmid File': plasmid_file,
                                                      'Class': random_gen.integers(PLASMID_LABEL, CHR_LABEL + 1)}])],
                                      ignore_index=True)
    duplicate_ids.to_csv(f'{input_dir}/duplicate_ids.csv', index=False)


def get_ids_to_remove(input_dir):
    duplicate_ids = pd.read_csv(f'{input_dir}/duplicate_ids.csv')

    remove_per_chr = duplicate_ids[duplicate_ids['Class'] == PLASMID_LABEL].groupby(
        'Chromosome File')['Read ID'].apply(list).to_dict()
    remove_per_plasmid = duplicate_ids[duplicate_ids['Class'] == CHR_LABEL].groupby(
        'Plasmid File')['Read ID'].apply(list).to_dict()
    remove_per_file = {**remove_per_chr, **remove_per_plasmid}

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


def cut_reads(in_dir, out_dir, cutoff, min_seq_len, max_seq_len, random_gen):
    if min_seq_len >= max_seq_len:
        raise ValueError('The minimum sequence length must be smaller than the maximum sequence length!')

    ds_name = os.path.basename(in_dir)
    kept_reads = 0

    # create file for ground truth labels
    label_df = pd.DataFrame(columns=['Read ID', 'GT Label'])

    for input_file in tqdm(glob.glob(f'{in_dir}/*.fast5')):
        output_file = os.path.join(out_dir, os.path.basename(input_file))
        label = 'plasmid' if any(c in os.path.basename(input_file) for c in ['plasmid', 'pos']) else 'chr'

        with get_fast5_file(input_file, mode='r') as f5_old, MultiFast5File(output_file, mode='w') as f5_new:
            for i, read in enumerate(f5_old.get_reads()):
                # get random sequence length per read
                seq_len = random_gen.integers(min_seq_len, max_seq_len + 1)

                # only parse reads that are long enough
                if len(read.handle[read.raw_dataset_name]) >= (cutoff + seq_len):
                    kept_reads += 1
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

    # store ground truth labels of kept reads
    label_df.to_csv(f'{out_dir}/max{int(max_seq_len/1000)}_gt_{ds_name}_labels.csv', index=False)
    print(f'Number of reads in {out_dir}: {kept_reads}')


@click.command()
@click.option('--real_test_dir', '-real', type=click.Path(exists=True), required=True,
              help='input directory with real test data')
@click.option('--sim_test_dir', '-sim', type=click.Path(exists=True), required=True,
              help='input directory with simulated test data')
@click.option('--random_seed', '-s', default=42, help='seed for random operations')
@click.option('--batch_size', '-b', default=5000, help='number of reads per batch for fast5 creation')
@click.option('--threads', '-t', default=32, help='number of threads to use for fast5 creation')
@click.option('--cutoff', '-c', default=1000, help='cutoff the first c signals')
@click.option('--min_seq_len', '-min', default=2000, help='minimum number of signals per read')
@click.option('--max_seq_lens', '-max', multiple=True, default=[4000, 6000, 8000],
              help='maximum number of signals per read, defining several is possible')
def main(real_test_dir, sim_test_dir, random_seed, batch_size, threads, cutoff, min_seq_len, max_seq_lens):
    random_gen = random.default_rng(random_seed)

    # extract duplicate read IDs in real test data and randomly choose ground truth class per duplicate ID
    get_files_per_id(real_test_dir)
    get_duplicate_ids(real_test_dir, random_gen)
    get_ids_to_remove(real_test_dir)

    # create new folder for FAST5 files without duplicate read IDs
    real_test_dir_dupl = f'{real_test_dir}_dupl'
    if not os.path.exists(real_test_dir_dupl):
        os.makedirs(real_test_dir_dupl)
    fill_new_folder(real_test_dir, real_test_dir_dupl, batch_size, threads)

    # cut real and simulated test reads
    for ds in real_test_dir_dupl, sim_test_dir:
        for mx in max_seq_lens:
            output_dir = f'{ds}_max{int(mx / 1000)}'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            cut_reads(ds, output_dir, cutoff, min_seq_len, mx, random_gen)


if __name__ == '__main__':
    main()
