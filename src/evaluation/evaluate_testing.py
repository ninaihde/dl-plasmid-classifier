"""
This script evaluates the results of the execution of classify.py with test data. Therefore, different evaluation
metrics are calculated and plotted.
"""

import click
import glob
import os
import pandas as pd

from plot_helper import create_barplot_for_several_metrics, create_lineplot_per_max
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score, \
    recall_score


@click.command()
@click.option('--input_data', '-d', help='path to root folder containing data', type=click.Path(exists=True),
              required=True)
@click.option('--input_logs', '-l', help='path to folder with .txt logs/ all prints', type=click.Path(exists=True),
              required=True)
@click.option('--output_plots', '-op', help='path to folder where subfolder for created plots of run ID will be stored',
              type=click.Path(exists=True), required=True)
@click.option('--output_results', '-or', help='path to folder where calculated results will be stored',
              type=click.Path(exists=True), required=True)
@click.option('--prefix', '-p', help='prefix of data folders to evaluate', default='final')
@click.option('--run_id', '-r', help='identifier of runs to be evaluated', required=True)  # e.g. 'balancedLoss'
def main(input_data, input_logs, output_plots, output_results, prefix, run_id):
    # create subdirectory for plots to be generated
    plots_dir = f'{output_plots}/{run_id}'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    metrics = pd.DataFrame(
        columns=['Maximum Sequence Length', 'Cutting Method', 'Number of Epochs', 'TP', 'TN', 'FP', 'FN',
                 'Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1S', 'MCC'])

    for config_folder in [cf for cf in os.listdir(input_data) if cf.startswith(prefix)]:
        gt = pd.read_csv(f'{input_data}/{config_folder}/gt_test_labels.csv')
        max_seq_len = int(config_folder.split('_')[1].replace('max', ''))
        cutting_method = config_folder.split('_')[2].replace('cut', '')

        for test_folder in glob.glob(f'{input_data}/{config_folder}/classify_*_{run_id}/'):
            n_epochs = int(test_folder.split(os.path.sep)[-3].split('_')[1].replace('epochs', ''))
            pred_files = [i for i in glob.glob(f'{test_folder}*.csv')]
            pred = pd.concat([pd.read_csv(f) for f in pred_files])

            # merge ground truth and predicted labels based on read ID
            merged = pd.merge(gt, pred, left_on='Read ID', right_on='Read ID')

            # calculate performance metrics of current testing folder
            metrics = pd.concat([metrics, pd.DataFrame([{
                'Maximum Sequence Length': max_seq_len,
                'Cutting Method': cutting_method,
                'Number of Epochs': n_epochs,
                'TP': len(merged[(merged['Predicted Label'] == 'plasmid') & (merged['GT Label'] == 'plasmid')]),
                'TN': len(merged[(merged['Predicted Label'] == 'chr') & (merged['GT Label'] == 'chr')]),
                'FP': len(merged[(merged['Predicted Label'] == 'plasmid') & (merged['GT Label'] == 'chr')]),
                'FN': len(merged[(merged['Predicted Label'] == 'chr') & (merged['GT Label'] == 'plasmid')]),
                'Accuracy': accuracy_score(merged['GT Label'], merged['Predicted Label']),
                # (TP + TN) / (TP + TN + FP + FN)
                'Balanced Accuracy': balanced_accuracy_score(merged['GT Label'], merged['Predicted Label']),
                # (TPR + TNR) / 2
                'Precision': precision_score(merged['GT Label'], merged['Predicted Label'], pos_label='plasmid'),
                # TP / (TP + FP)
                'Recall': recall_score(merged['GT Label'], merged['Predicted Label'], pos_label='plasmid'),
                # TPR: TP / (TP + FN)
                'F1S': f1_score(merged['GT Label'], merged['Predicted Label'], pos_label='plasmid'),
                # harmonic mean between precision and recall
                'MCC': matthews_corrcoef(merged['GT Label'], merged['Predicted Label']),
            }])], ignore_index=True)

            # store merged dataframe for analysis of FN and FP
            merged.to_csv(f'{test_folder}gt_and_pred_labels.csv', index=False)

    # add missing metrics and store testing results
    metrics['TNR'] = metrics['TN'] / (metrics['TN'] + metrics['FP'])  # specificity
    metrics['FPR'] = metrics['FP'] / (metrics['FP'] + metrics['TN'])  # 1 - specificity
    metrics['FNR'] = metrics['FN'] / (metrics['FN'] + metrics['TP'])
    metrics['ID'] = 'max' + metrics['Maximum Sequence Length'].astype(int).astype(str) \
                    + '_cut' + metrics['Cutting Method'] \
                    + '_ep' + metrics['Number of Epochs'].astype(str)
    metrics.to_csv(f'{output_results}/testing_results_{run_id}.csv', index=False)

    # create plots with 4 subplots each, each subplot showing the testing results with respect to one metric
    metric_groups = [['Balanced Accuracy', 'Precision', 'MCC', 'F1S'], ['Recall', 'TNR', 'FPR', 'FNR']]
    create_barplot_for_several_metrics(metrics, metric_groups, plots_dir, 'test')

    # plot metrics per maximum sequence length
    for metric in ['Balanced Accuracy', 'Accuracy', 'F1S', 'Precision', 'Recall', 'FPR', 'TNR', 'FNR', 'MCC']:
        create_lineplot_per_max(metrics, metric, plots_dir, 'test')

    # extract and plot runtimes
    runtimes = pd.DataFrame(columns=['Maximum Sequence Length', 'Cutting Method', 'Number of Epochs', 'Runtime (min)'])
    for filepath in glob.glob(f'{input_logs}/classify_*_{run_id}.txt'):
        last_line = open(filepath, 'r').readlines()[-1]
        filename_splitted = os.path.basename(filepath).split('_')
        runtimes = pd.concat([runtimes,
                              pd.DataFrame([{'Maximum Sequence Length': int(filename_splitted[1].replace('max', '')),
                                             'Cutting Method': filename_splitted[2].replace('cut', ''),
                                             'Number of Epochs': int(filename_splitted[3].replace('epochs', '')),
                                             'Runtime (min)': float(last_line.split(' ')[3]) / 60}])],
                             ignore_index=True)
    create_lineplot_per_max(runtimes, 'Runtime (min)', plots_dir, 'test')

    print('Finished.')


if __name__ == '__main__':
    main()
