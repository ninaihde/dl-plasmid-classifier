import click
import glob
import os
import pandas as pd

from plot_helper import create_barplot_for_several_metrics, create_lineplot_for_single_metric, \
    create_lineplot_for_single_metric_twice, create_lineplot_per_max


@click.command()
@click.option('--input', '-i', help='path to root folder containing data', type=click.Path(exists=True))
@click.option('--output_plots', '-op', help='path to folder where subfolder for created plots of run ID will be stored',
              type=click.Path(exists=True))
@click.option('--output_results', '-or', help='path to folder where calculated results will be stored',
              type=click.Path(exists=True))
@click.option('--prefix', '-p', help='prefix of data folders to evaluate', default='prototypeV1')
@click.option('--run_id', '-r', help='identifier of runs to be evaluated', required=True)  # e.g. 'balancedLoss'
@click.option('--model_selection_criterion', '-s', default='Loss', type=click.Choice(['Loss', 'Accuracy']),
              help='model selection criterion, choose between validation loss and accuracy')
def main(input, output_plots, output_results, prefix, run_id, model_selection_criterion):
    pd.options.mode.chained_assignment = None

    # create subdirectory for plots to be generated
    plots_dir = f'{output_plots}/{run_id}'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    best_models = pd.DataFrame(
        columns=['Maximum Sequence Length', 'Cutting Method', 'Number of Epochs', 'Epoch', 'Validation Accuracy',
                 'Validation Loss', 'TN', 'FP', 'FN', 'TP', 'Balanced Accuracy', 'F1S', 'MCC', 'Precision', 'Recall',
                 'TNR', 'FPR', 'FNR'])

    for config_folder in [cf for cf in os.listdir(input) if cf.startswith(prefix)]:
        for train_folder in glob.glob(f'{input}/{config_folder}/train_*_{run_id}/logs/'):
            # extract logs
            n_epochs = int(train_folder.split(os.path.sep)[-3].split('_')[1].replace('epochs', ''))
            train_results = pd.read_csv(f'{train_folder}train_results_epoch{n_epochs - 1}.csv')
            val_results = pd.read_csv(f'{train_folder}val_results_epoch{n_epochs - 1}.csv')

            # add missing metrics
            val_results['TNR'] = val_results['TN'] / (val_results['TN'] + val_results['FP'])  # specificity
            val_results['FPR'] = val_results['FP'] / (val_results['FP'] + val_results['TN'])  # 1 - specificity
            val_results['FNR'] = val_results['FN'] / (val_results['FN'] + val_results['TP'])

            # extract best model
            if model_selection_criterion == 'Loss':
                val_results_bm = val_results.iloc[val_results['Validation Loss'].idxmin()]
            else:
                val_results_bm = val_results.iloc[val_results['Validation Accuracy'].idxmax()]
            max_seq_len = int(config_folder.split('_')[1].replace('max', ''))
            val_results_bm['Maximum Sequence Length'] = max_seq_len
            cutting_method = config_folder.split('_')[2].replace('cut', '')
            val_results_bm['Cutting Method'] = cutting_method
            val_results_bm['Number of Epochs'] = n_epochs
            best_models = pd.concat([best_models, pd.DataFrame([val_results_bm])], ignore_index=True)

            # plot validation metrics over time
            for metric in ['Balanced Accuracy', 'F1S', 'MCC', 'Precision', 'Recall', 'TNR', 'FPR', 'FNR']:
                create_lineplot_for_single_metric(val_results, metric, max_seq_len, cutting_method, n_epochs, plots_dir)

            # prepare dataframe with training logs for merging
            train_results.columns = train_results.columns.str.lstrip('Training ')
            train_results['TYPE'] = 'Training'

            # prepare dataframe with validation logs for merging
            n_batches = train_results['Batch'].nunique()
            val_results_reduced = val_results[['Epoch', 'Validation Loss', 'Validation Accuracy']]
            val_results_reduced['Batch'] = n_batches - 1
            val_results_reduced.columns = val_results_reduced.columns.str.lstrip('Validation ')
            val_results_reduced['TYPE'] = 'Validation'

            # merge both dataframes for easier plotting
            merged = pd.concat([train_results, val_results_reduced], ignore_index=True)
            merged['Batch_cont'] = merged.apply(lambda row: row['Batch'] * (row['Epoch'] + 1), axis=1)

            # plot loss and accuracy over time
            for measure in ['Loss', 'Accuracy']:
                create_lineplot_for_single_metric_twice(merged, measure, max_seq_len, cutting_method, n_epochs,
                                                        plots_dir)

    # store best models
    best_models['ID'] = 'max' + best_models['Maximum Sequence Length'].astype(int).astype(str) \
                        + '_cut' + best_models['Cutting Method'] \
                        + '_ep' + best_models['Number of Epochs'].astype(str)
    best_models.to_csv(f'{output_results}/validation_results_{run_id}.csv', index=False)

    # create plots with 4 subplots each, each subplot showing the best models with respect to one metric
    metric_groups = [['Balanced Accuracy', 'Precision', 'MCC', 'F1S'], ['Recall', 'TNR', 'FPR', 'FNR']]
    create_barplot_for_several_metrics(best_models, metric_groups, plots_dir, 'val')

    # plot metrics per maximum sequence length
    for m in ['Balanced Accuracy', 'Validation Accuracy', 'F1S', 'MCC', 'Precision', 'Recall', 'FPR', 'TNR', 'FNR']:
        create_lineplot_per_max(best_models, m, plots_dir, 'val')

    print('Finished.')


if __name__ == '__main__':
    main()
