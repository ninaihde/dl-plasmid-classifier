"""
This script evaluates the results of the execution of classify.py with test data. Therefore, different evaluation
metrics are calculated and plotted.
"""

import click
import glob
import os
import pandas as pd

from evaluation_helper import create_barplot_for_several_metrics, create_lineplot_per_max, get_time_and_rss, get_max_gpu_usage
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score, \
    recall_score


@click.command()
@click.option('--input_data', '-d', help='path to root folder containing data', type=click.Path(exists=True),
              required=True)
@click.option('--input_logs', '-l', help='path to folder with .txt logs/ all prints', type=click.Path(exists=True),
              required=True)
@click.option('--output_figures', '-f', help='folder in which subfolder for created figures of run ID will be stored',
              type=click.Path(), required=True)
@click.option('--output_results', '-r', help='path to folder where calculated results will be stored',
              type=click.Path(), required=True)
def main(input_data, input_logs, output_figures, output_results):
    metrics = pd.DataFrame(
        columns=['ID', 'Maximum Sequence Length', 'Dataset', 'Criterion', 'Epochs', 'TP', 'TN', 'FP', 'FN',
                 'Accuracy', 'Balanced Accuracy', 'Precision', 'TPR (Sensitivity)', 'TNR (Specificity)', 'F1S', 'MCC'])

    # create directories for plots and CSVs to be generated
    if not os.path.exists(output_figures):
        os.makedirs(output_figures)
    if not os.path.exists(output_results):
        os.makedirs(output_results)

    for result_folder in glob.glob(f'{input_data}/*/'):
        run_name = os.path.basename(result_folder[:-1])
        max_seq_len = run_name.split('_')[0].replace('max', '')
        n_epochs = run_name.split('_')[1].replace('epochs', '')
        dataset = run_name.split('_')[2]
        criterion = run_name.split('_')[3]

        gt = pd.read_csv(f'{input_data}/max{max_seq_len}_gt_test_{dataset}_labels.csv')
        pred_files = [i for i in glob.glob(f'{result_folder}/*.csv')]
        pred = pd.concat([pd.read_csv(f) for f in pred_files])

        # merge ground truth and predicted labels based on read ID
        merged = pd.merge(gt, pred, left_on='Read ID', right_on='Read ID')

        # calculate performance metrics of current result folder
        metrics = pd.concat([metrics, pd.DataFrame([{
            'ID': run_name,
            'Maximum Sequence Length': int(max_seq_len) * 1000,
            'Dataset': dataset,
            'Criterion': criterion,
            'Epochs': n_epochs,
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
            'TPR (Sensitivity)': recall_score(merged['GT Label'], merged['Predicted Label'], pos_label='plasmid'),
            # Recall/ TPR: TP / (TP + FN)
            'TNR (Specificity)': recall_score(merged['GT Label'], merged['Predicted Label'], pos_label='chr'),
            # TNR: TN / (TN + FP)
            'F1S': f1_score(merged['GT Label'], merged['Predicted Label'], pos_label='plasmid'),
            # harmonic mean between precision and recall
            'MCC': matthews_corrcoef(merged['GT Label'], merged['Predicted Label']),
        }])], ignore_index=True)

        # store merged dataframe
        merged.to_csv(f'{output_results}/OUR_gt_and_pred_labels_{run_name}.csv', index=False)

    # add missing metrics and store testing results
    metrics['TNR'] = metrics['TN'] / (metrics['TN'] + metrics['FP'])  # specificity
    metrics['FPR'] = metrics['FP'] / (metrics['FP'] + metrics['TN'])  # 1 - specificity
    metrics['FNR'] = metrics['FN'] / (metrics['FN'] + metrics['TP'])
    metrics.to_csv(f'{output_results}/OUR_metrics.csv', index=False)

    # create plots with 4 subplots each, each subplot showing the testing results with respect to one metric
    metric_groups = [['Balanced Accuracy', 'Precision', 'MCC', 'F1S'],
                     ['TPR (Sensitivity)', 'TNR (Specificity)', 'FPR', 'FNR']]
    create_barplot_for_several_metrics(metrics, metric_groups, output_figures, 'test', 'Dataset', 'OUR')

    # plot metrics per maximum sequence length
    for metric in ['Balanced Accuracy', 'Accuracy', 'F1S', 'Precision', 'TPR (Sensitivity)', 'FPR', 'TNR (Specificity)',
                   'FNR', 'MCC']:
        create_lineplot_per_max(metrics, metric, output_figures, 'test', 'Dataset', 'Criterion', 'OUR')

    # plot memory consumption (peak RSS, GPU) and different times
    times_and_memory = pd.DataFrame(columns=['ID', 'Maximum Sequence Length', 'Dataset', 'Criterion', 'User Time',
                                             'System Time', 'Elapsed Time', 'Max RSS (GB)', 'Max GPU Usage (GiB)'])
    for log_file in glob.glob(f'{input_logs}/step7_infer_*.txt'):
        if log_file.endswith('_gpu.txt'):
            continue
        else:
            time_file = log_file
            nvidia_file = log_file.replace('.txt', '_gpu.txt')

            mx = log_file.split('_')[2].replace('b', '')
            ds = log_file.split('_')[3]
            ct = log_file.split('_')[4]

            user_time, system_time, elapsed_time, max_rss = get_time_and_rss(time_file)
            max_gpu_usage = get_max_gpu_usage(nvidia_file, 'python')

            times_and_memory = pd.concat([times_and_memory,
                                          pd.DataFrame([{'ID': f'{mx}_{ds}_{ct}',
                                                         'Maximum Sequence Length': int(mx) * 1000,
                                                         'Dataset': ds,
                                                         'Criterion': ct,
                                                         'User Time': user_time,
                                                         'System Time': system_time,
                                                         'Elapsed Time': elapsed_time,
                                                         'Max RSS (GB)': max_rss,
                                                         'Max GPU Usage (GiB)': max_gpu_usage}])],
                                         ignore_index=True)
    times_and_memory.to_csv(f'{output_results}/OUR_times_and_memory.csv', index=False)

    create_lineplot_per_max(times_and_memory, 'User Time', output_figures, 'test', 'Dataset', 'Criterion', 'OUR')
    create_lineplot_per_max(times_and_memory, 'System Time', output_figures, 'test', 'Dataset', 'Criterion', 'OUR')
    create_lineplot_per_max(times_and_memory, 'Elapsed Time', output_figures, 'test', 'Dataset', 'Criterion', 'OUR')
    create_lineplot_per_max(times_and_memory, 'Max RSS (GB)', output_figures, 'test', 'Dataset', 'Criterion', 'OUR')
    create_lineplot_per_max(times_and_memory, 'Max GPU Usage (GiB)', output_figures, 'test', 'Dataset', 'Criterion', 'OUR')

    print('Finished.')


if __name__ == '__main__':
    main()
