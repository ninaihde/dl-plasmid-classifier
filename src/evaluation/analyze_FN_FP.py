import click
import glob
import gzip
import os
import pandas as pd
import subprocess

from Bio import SeqIO


BASES_PER_SEC = 450  # assumption: 4000 signals per second = 450 bases
BASES_PER_1K_SIGNALS = BASES_PER_SEC / 4  # 1000 signals = 112.5 bases
SKIPPED_SIGNALS_IN_K = 1  # first 1000 signals skipped


def prepare_fastqs(input_dir, output_dir, prefix, run_id, model_selection_criterion):
    for config_folder in [cf for cf in os.listdir(input_dir) if cf.startswith(prefix)]:
        max_len = int(config_folder.split('_')[1].replace('max', ''))
        cut_method = config_folder.split('_')[2].replace('cut', '')

        gt_labels_chr = pd.read_csv(f'{input_dir}/{config_folder}/gt_val_chr_labels.csv')
        gt_labels_p = pd.read_csv(f'{input_dir}/{config_folder}/gt_val_plasmid_labels.csv')
        gt_labels = pd.concat([gt_labels_chr, gt_labels_p], ignore_index=True)

        for train_folder in glob.glob(f'{input_dir}/{config_folder}/train_*_{run_id}/'):
            n_epochs = int(train_folder.split(os.path.sep)[-2].split('_')[1].replace('epochs', ''))
            print(f'Current folder: Max = {max_len}, Cutting = {cut_method}, Epochs = {n_epochs}')

            val_results = pd.read_csv(f'{train_folder}logs/val_results_epoch{n_epochs - 1}.csv')
            if model_selection_criterion == 'Loss':
                best_model = val_results.iloc[val_results['Validation Loss'].idxmin()]
            else:
                best_model = val_results.iloc[val_results['Validation Accuracy'].idxmax()]
            best_epoch = int(best_model['Epoch'])
            pred_labels = pd.read_csv(f'{train_folder}pred_labels/pred_labels_epoch{best_epoch}.csv')

            merged = pd.merge(gt_labels, pred_labels, left_on='Read ID', right_on='Read ID')

            print('Extracting read IDs of FNs and FPs...')
            FN_ids = get_read_ids(merged, 'chr', 'plasmid', output_dir, max_len, cut_method, n_epochs, run_id, 'FN')
            FP_ids = get_read_ids(merged, 'plasmid', 'chr', output_dir, max_len, cut_method, n_epochs, run_id, 'FP')

            print('Filter and cut sequences...')
            with gzip.open(f'{output_dir}/{run_id}/max{max_len}_cut{cut_method}_{n_epochs}epochs_'
                           f'sequences_FN.fastq.gz', 'wt') as out_file_FN, \
                    gzip.open(f'{output_dir}/{run_id}/max{max_len}_cut{cut_method}_{n_epochs}epochs_'
                              f'sequences_FP.fastq.gz', 'wt') as out_file_FP:
                filter_sequences(input_dir, max_len, out_file_FN, out_file_FP, FN_ids, FP_ids)


def get_read_ids(df, pred_label, gt_label, output_dir, max_len, cut_method, n_epochs, run_id, result_type):
    read_ids = df[(df['Predicted Label'] == pred_label) & (df['GT Label'] == gt_label)]['Read ID'].tolist()

    with open(f'{output_dir}/{run_id}/max{max_len}_cut{cut_method}_{n_epochs}epochs_read_ids_{result_type}.txt', 'w') \
            as id_file:
        for r_id in read_ids:
            id_file.write(f'{r_id}\n')

    return read_ids


def filter_sequences(input_dir, max_len, out_file_FN, out_file_FP, FN_ids, FP_ids):
    for fastq_file in glob.glob(f'{input_dir}/fastq/*.fastq.gz'):
        with gzip.open(fastq_file, 'rt') as in_file:
            for record in SeqIO.parse(in_file, 'fastq'):
                if record.id in FN_ids:
                    record = cut_sequence(max_len, record)
                    r = SeqIO.write(record, out_file_FN, 'fastq')
                    if r != 1:
                        print(f'Error while writing FN sequence {record.id} from {fastq_file}')

                elif record.id in FP_ids:
                    record = cut_sequence(max_len, record)
                    r = SeqIO.write(record, out_file_FP, 'fastq')
                    if r != 1:
                        print(f'Error while writing FP sequence {record.id} from {fastq_file}')


def cut_sequence(max_len, record):
    start_idx = int(BASES_PER_1K_SIGNALS)
    end_idx = int((SKIPPED_SIGNALS_IN_K + max_len) * BASES_PER_1K_SIGNALS)

    all_annotations = record.letter_annotations
    record.letter_annotations = {}
    record.seq = record.seq[start_idx:end_idx]
    record.letter_annotations = {'phred_quality': all_annotations['phred_quality'][start_idx:end_idx]}

    return record


def map_sequences(reference_file, read_file, bam_file):
    print(f'Mapping {read_file} against reference {reference_file}...')
    minimap_cmd = ['minimap2', '-ax', 'map-ont', reference_file, read_file]
    minimap_output = subprocess.Popen(minimap_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    samsort_cmd = ['samtools', 'sort', '-O', 'BAM', '-o', bam_file]
    samsort_output = subprocess.Popen(samsort_cmd, stdin=minimap_output.stdout, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
    minimap_output.stdout.close()
    samsort_output.communicate()

    stat_cmd = ['samtools', 'index', bam_file]
    subprocess.run(stat_cmd)


@click.command()
@click.option('--input_dir', '-i', help='path to input folder', type=click.Path(exists=True))
@click.option('--output_dir', '-o', help='path to output folder', type=click.Path())
@click.option('--prefix', '-p', help='prefix of data folders to evaluate', default='prototypeV1')
@click.option('--run_id', '-r', help='identifier of runs to be evaluated', required=True)  # e.g. 'balancedLoss'
@click.option('--model_selection_criterion', '-s', default='Loss', type=click.Choice(['Loss', 'Accuracy']),
              help='model selection criterion, choose between validation loss and accuracy')
def main(input_dir, output_dir, prefix, run_id, model_selection_criterion):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(f'{output_dir}/{run_id}'):
        os.makedirs(f'{output_dir}/{run_id}')

    prepare_fastqs(input_dir, output_dir, prefix, run_id, model_selection_criterion)

    for read_file in glob.glob(f'{output_dir}/{run_id}/sequences*.fastq.gz'):
        for reference_file in glob.glob(f'{input_dir}/Genomes/*.fasta'):
            bam_file = f'{output_dir}/{run_id}/alignment_{os.path.basename(read_file).replace("sequences_", "")}' \
                       f'_{os.path.basename(reference_file)}.bam'
            map_sequences(reference_file, read_file, bam_file)

    print('Finished.')


if __name__ == '__main__':
    main()
