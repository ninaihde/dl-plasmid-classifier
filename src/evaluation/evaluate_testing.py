"""
This script evaluates the results produced by executing classify.py with test data. Therefore, different evaluation
metrics, time and resource measurements are calculated and plotted. In addition, if the result CSV files for the execution
of minimap2 (see evaluate_minimap.ipynb) are given to this script, plots combining both approaches are created.

Each configuration (see -d and -l parameter) describes a certain dataset type (real or simulated) and maximum sequence
length applied to the respective test reads. We evaluated six configurations:
  - max4_real
  - max6_real
  - max8_real
  - max4_sim
  - max6_sim
  - max8_sim
"""

import click
import glob
import os
import pandas as pd

from evaluation_helper import create_barplot_for_several_metrics, create_lineplot_per_max, get_time_and_rss, \
    get_max_gpu_usage, create_barplot_per_metric_and_multiple_approaches, datetime_to_seconds
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score


@click.command()
@click.option('--input_data', '-d', type=click.Path(exists=True), required=True,
              help='path to folder containing a subfolder for the classification results of each configuration')
@click.option('--input_logs', '-l', type=click.Path(exists=True), required=True,
              help='path to folder with logs (.txt) for each configuration')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='output folder for result data and figures')
@click.option('--minimap_metrics_csv)', '-m', type=click.Path(exists=True), required=False, default=None,
              help='CSV containing metrics of minimap2 execution')
@click.option('--minimap_times_csv)', '-t', type=click.Path(exists=True), required=False, default=None,
              help='CSV containing time and memory consumption measurements of minimap2 execution')
def main(input_data, input_logs, output, minimap_metrics_csv, minimap_times_csv):
    metrics = pd.DataFrame(
        columns=['ID', 'Maximum Sequence Length', 'Dataset', 'Criterion', 'Epochs', 'TP', 'TN', 'FP', 'FN',
                 'Accuracy', 'Balanced Accuracy', 'Precision', 'TPR (Sensitivity)', 'TNR (Specificity)', 'F1S'])

    # create directories for plots and CSVs to be generated
    if not os.path.exists(output):
        os.makedirs(output)
    if not os.path.exists(f'{output}/figures'):
        os.makedirs(f'{output}/figures')
    if not os.path.exists(f'{output}/results'):
        os.makedirs(f'{output}/results')

    for result_folder in glob.glob(f'{input_data}/*/'):
        run_name = os.path.basename(result_folder[:-1])
        max_seq_len = run_name.split('_')[0].replace('max', '')
        n_epochs = run_name.split('_')[1].replace('epochs', '')
        dataset = run_name.split('_')[2]
        criterion = run_name.split('_')[3]

        # merge ground truth and predicted labels based on read ID
        gt = pd.read_csv(f'{input_data}/max{max_seq_len}_gt_test_{dataset}_labels.csv')
        pred_files = [i for i in glob.glob(f'{result_folder}/batch_*.csv')]
        pred = pd.concat([pd.read_csv(f) for f in pred_files])
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
            'F1S': f1_score(merged['GT Label'], merged['Predicted Label'], pos_label='plasmid')
        }])], ignore_index=True)

        # store merged dataframe
        merged.to_csv(f'{output}/results/OUR_gt_and_pred_labels_{run_name}.csv', index=False)

    # add missing metrics and store testing results
    metrics['FPR'] = metrics['FP'] / (metrics['FP'] + metrics['TN'])  # 1 - specificity
    metrics['FNR'] = metrics['FN'] / (metrics['FN'] + metrics['TP'])
    metrics['ID'] = metrics['ID'].replace('_15epochs', '', regex=True)
    metrics.to_csv(f'{output}/results/OUR_metrics.csv', index=False)

    # create plots with 4 subplots each, each subplot showing the testing results with respect to one metric
    metric_groups = [['Balanced Accuracy', 'Precision', 'Accuracy', 'F1S'],
                     ['TPR (Sensitivity)', 'TNR (Specificity)', 'FPR', 'FNR']]
    create_barplot_for_several_metrics(metrics, metric_groups, f'{output}/figures', 'Dataset', 'OUR')

    if minimap_metrics_csv is not None:
        # embed minimap2's metrics
        metrics['Approach'] = 'Our Approach'
        minimap_metrics = pd.read_csv(minimap_metrics_csv)
        minimap_metrics['Approach'] = 'Guppy + Minimap2'
        shared_metrics = pd.concat([metrics, minimap_metrics], ignore_index=True)
        shared_metrics = shared_metrics[shared_metrics['Criterion'] != 'acc']
        shared_metrics['ID'] = shared_metrics['ID'].replace(['_loss', '_acc'], ['', ''], regex=True)
        shared_metrics.to_csv(f'{output}/results/SHARED_metrics.csv', index=False)

    # plot metrics per maximum sequence length
    for metric in ['Balanced Accuracy', 'Accuracy', 'TPR (Sensitivity)', 'FPR', 'TNR (Specificity)', 'FNR']:
        create_lineplot_per_max(metrics, metric, f'{output}/figures', 'Dataset', 'Criterion', 'OUR')
        if minimap_metrics_csv is not None:
            create_barplot_per_metric_and_multiple_approaches(shared_metrics, metric, f'{output}/figures', 'SHARED')

    # plot memory consumption (peak RSS, GPU) and different times
    times_and_memory = pd.DataFrame(columns=['ID', 'Maximum Sequence Length', 'Dataset', 'Criterion', 'User Time',
                                             'System Time', 'Elapsed Time', 'Max RSS (GB)', 'Max GPU Memory Usage (GiB)'])
    for log_file in glob.glob(f'{input_logs}/step7_infer_*.txt'):
        if '_gpu' in log_file:
            continue
        else:
            mx = log_file.split('_')[2].replace('b', '')
            ds = log_file.split('_')[3]
            ct = log_file.split('_')[4].replace('.txt', '')

            user_time, system_time, elapsed_time, max_rss = get_time_and_rss(log_file)            
            nvidia_file = log_file.replace('.txt', '_gpu.txt')
            max_gpu_usage = get_max_gpu_usage(nvidia_file, 'python')

            times_and_memory = pd.concat([times_and_memory,
                                          pd.DataFrame([{'ID': f'max{mx}_{ds}_{ct}',
                                                         'Maximum Sequence Length': int(mx) * 1000,
                                                         'Dataset': ds,
                                                         'Criterion': ct,
                                                         'User Time': user_time,
                                                         'System Time': system_time,
                                                         'Elapsed Time': elapsed_time,
                                                         'Max RSS (GB)': max_rss,
                                                         'Max GPU Memory Usage (GiB)': max_gpu_usage}])],
                                         ignore_index=True)
    times_and_memory.to_csv(f'{output}/results/OUR_times_and_memory.csv', index=False)

    for measure in ['User Time', 'System Time', 'Elapsed Time', 'Max RSS (GB)', 'Max GPU Memory Usage (GiB)']:
        create_lineplot_per_max(times_and_memory, measure, f'{output}/figures', 'Dataset', 'Criterion', 'OUR')

    if minimap_times_csv is not None:
        # embed minimap2's measures
        times_and_memory['Approach'] = 'Our Approach'
        minimap_times_and_memory = pd.read_csv(minimap_times_csv)
        minimap_times_and_memory['Approach'] = 'Guppy + Minimap2'
        shared_times_and_memory = pd.concat([times_and_memory, minimap_times_and_memory], ignore_index=True)
        shared_times_and_memory['User Time (seconds)'] = shared_times_and_memory['User Time'].apply(datetime_to_seconds)
        shared_times_and_memory['System Time (seconds)'] = shared_times_and_memory['System Time'].apply(datetime_to_seconds)
        shared_times_and_memory['Elapsed Time (seconds)'] = shared_times_and_memory['Elapsed Time'].apply(datetime_to_seconds)

        shared_times_and_memory = shared_times_and_memory[shared_times_and_memory['Criterion'] != 'acc']
        shared_times_and_memory['ID'] = shared_times_and_memory['ID'].replace(['_loss', '_acc'], ['', ''], regex=True)
        shared_times_and_memory['Elapsed Time (minutes)'] = shared_times_and_memory['Elapsed Time (seconds)'] / 60
        shared_times_and_memory.to_csv(f'{output}/results/SHARED_times_and_memory.csv', index=False)

        for measure in ['User Time (seconds)', 'System Time (seconds)', 'Elapsed Time (minutes)', 'Max RSS (GB)',
                        'Max GPU Memory Usage (GiB)']:
            create_barplot_per_metric_and_multiple_approaches(shared_times_and_memory, measure, f'{output}/figures', 'SHARED')


if __name__ == '__main__':
    main()
