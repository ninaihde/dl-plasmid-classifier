"""
This script offers helper functions used by the evaluation scripts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import time


def get_time_and_rss(logfile):
    with open(logfile) as f:
        for line in f:
            if 'User time' in line:
                user_time = line.split(': ')[1].strip()  # seconds
                user_time = time.strftime('%H:%M:%S', time.gmtime(float(user_time)))  # h:mm:ss
            elif 'System time' in line:
                system_time = line.split(': ')[1].strip()  # seconds
                system_time = time.strftime('%H:%M:%S', time.gmtime(float(system_time)))  # h:mm:ss
            elif 'Elapsed (wall clock) time' in line:
                elapsed_time = line.split(': ')[1].strip()  # h:mm:ss or m:ss
            elif 'Maximum resident set size' in line:
                max_rss = line.split(': ')[1].strip()  # KB
                max_rss = int(max_rss) / 1e+6  # GB

    return user_time, system_time, elapsed_time, max_rss


def get_max_gpu_usage(logfile, process_name):
    gpu_usages = list()
    with open(logfile) as f:
        for line in f:
            if process_name in line:
                gpu_usages.append(line.split(process_name)[1].split('|')[0].strip())

    for i, usage in enumerate(gpu_usages):
        if 'MiB' not in usage:
            raise ValueError('Not all GPU measurements are MiB values!')
        else:
            gpu_usages[i] = int(usage.split('MiB')[0]) / 1024  # GiB

    return max(gpu_usages)


def create_lineplot_for_single_metric_twice(data, measure, mx, cut, epochs, plots_dir):
    plotdata = data.copy()
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(data=plotdata,
                 x='Batch_cont',
                 y=measure,
                 hue='TYPE')

    # adjust text
    plt.title(f'Training and Validation {measure} over Time\n(max sequence length: {mx}k, '
              f'cutted: {cut.lower()} normalization, epochs: {epochs})', fontsize=22)
    ax.set(xlim=(0, (epochs - 1)))
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel(measure, fontsize=18)
    ax.set_xticks(ticks=plotdata[plotdata['TYPE'] == 'Validation']['Batch_cont'].tolist(),
                  labels=plotdata['Epoch'].astype(int).unique(), fontsize=14)
    plt.yticks(fontsize=14)
    plt.rcParams['legend.title_fontsize'] = 14
    ax.legend(title=f'{measure} of', fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(f'{plots_dir}/max{mx}_cut{cut}_epochs{epochs}_{measure.lower()}_over_time.png', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()


def create_lineplot_for_single_metric(data, measure, mx, cut, epochs, plots_dir):
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(data=data,
                 x='Epoch',
                 y=measure)

    # adjust text
    plt.title(f'{measure} over Time\n(max sequence length: {mx}k, cutted: {cut.lower()} normalization, '
              f'epochs: {epochs})', fontsize=22)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel(measure, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.set(xlim=(0, (epochs - 1)))

    plt.tight_layout()
    plt.savefig(f'{plots_dir}/max{mx}_cut{cut}_epochs{epochs}_{measure.lower()}_over_time.png', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()


def create_barplot_for_several_metrics(data, metric_collections, plots_dir, type_id, hue, prefix):
    plotdata = data.copy()
    for h in data[hue].unique():
        plotdata['ID'] = plotdata['ID'].replace(f'_{h}', '', regex=True)
    plotdata = plotdata.sort_values(by='ID')

    for metric_collection in metric_collections:
        # plot 4 metrics (2 x 2) at once
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(22, 16), sharey=True)
        plt.rcParams['legend.title_fontsize'] = 14
        ax = axes.flatten()

        for i, metric in enumerate(metric_collection):
            sns.barplot(data=plotdata,
                        x='ID',
                        y=metric,
                        hue=hue,
                        palette=sns.color_palette('colorblind'),
                        ax=ax[i])

            # place y-values above bars
            for val in ax[i].containers:
                ax[i].bar_label(val, fmt='%.4f')

            # hide legend of subplot
            ax[i].get_legend().remove()

            # adjust text of subplot
            ax[i].set_title(metric, fontsize=22)
            ax[i].set_xlabel('', fontsize=18)
            ax[i].tick_params('x', labelrotation=90)
            ax[i].set_ylabel(metric, fontsize=18)
            ax[i].tick_params(axis='x', labelsize=14)
            ax[i].tick_params(axis='y', labelsize=14)

        plt.suptitle(f'Evaluation of {"Validation" if type_id == "val" else "Test"} Dataset', fontsize=24)
        handles, labels = ax[0].get_legend_handles_labels()
        plt.figlegend(handles, labels, title=hue, ncol=2, fontsize=14, loc='center',
                      bbox_to_anchor=(0.5, 0.9))

        plt.tight_layout(pad=6.0)
        plt.savefig(f'{plots_dir}/{prefix}_{"_".join(metric_collection)}_{hue}_{type_id}.png', dpi=300, facecolor='white')
        plt.close()


def create_lineplot_per_max(data, metric, plots_dir, type_id, hue, style, prefix):
    plotdata = data.copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(data=plotdata,
                 x='Maximum Sequence Length',
                 y=metric,
                 hue=hue,
                 style=style)

    # adjust text
    plt.title(f'{metric.replace(" (min)", "")} per Maximum Sequence Length', fontsize=22)
    plt.xlabel('Maximum Sequence Length', fontsize=18)
    plt.ylabel(metric, fontsize=18)
    plt.xticks(fontsize=14)
    plt.locator_params(axis='x', nbins=plotdata['Maximum Sequence Length'].nunique())
    plt.yticks(fontsize=14)
    plt.rcParams['legend.title_fontsize'] = 14
    ax.legend(fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(f'{plots_dir}/{prefix}_{metric.replace(" ", "_")}_per_max_{hue}_{style}_{type_id}.png', dpi=300, facecolor='white')
    plt.close()
