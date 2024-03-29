"""
This script offers helper functions for the visualization and logfile extraction used by the evaluation scripts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import time


def seconds_to_datetime(time_in_seconds):
    return time.strftime('%H:%M:%S', time.gmtime(float(time_in_seconds)))


def get_time_and_rss(logfile):
    user_time, system_time, elapsed_time, max_rss = None, None, None, None
    with open(logfile) as f:
        for line in f:
            if 'User time' in line:
                user_time = line.split(': ')[1].strip()  # seconds
                user_time = seconds_to_datetime(user_time)  # h:mm:ss
            elif 'System time' in line:
                system_time = line.split(': ')[1].strip()  # seconds
                system_time = seconds_to_datetime(system_time)  # h:mm:ss
            elif 'Elapsed (wall clock) time' in line:
                elapsed_time = line.split(': ')[1].strip()  # h:mm:ss or m:ss
            elif 'Maximum resident set size' in line:
                max_rss = line.split(': ')[1].strip()  # KB
                max_rss = int(max_rss) / 1e+6  # GB

    return user_time, system_time, elapsed_time, max_rss


def datetime_to_seconds(datetime_column):
    h, m, s = map(int, datetime_column.split(':'))
    return int(h) * 3600 + int(m) * 60 + int(s)


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

    return round(max(gpu_usages), 2)


def create_barplot_per_metric_and_multiple_approaches(data, metric, plots_dir, prefix):
    plotdata = data.copy()
    plotdata = plotdata.sort_values(by=['ID'])
    plt.rcParams.update({'font.size': 22})

    # create plot
    fig, ax = plt.subplots(figsize=(18, 11))
    sns.barplot(data=plotdata,
                x='ID',
                y=metric,
                hue='Approach',
                palette=sns.color_palette('colorblind'),
                ci=None)

    # adjust text of subplot
    plt.xlabel('')
    ax.tick_params('x', labelsize=22, labelrotation=90)
    plt.ylabel(metric, fontsize=26)
    ax.tick_params(axis='y', labelsize=22)
    ax.legend(fontsize=26, loc='upper right', bbox_to_anchor=(1, 1.2))

    plt.tight_layout()
    plt.savefig(f'{plots_dir}/{prefix}_{metric.replace(" ", "_")}.png', dpi=300, facecolor='white')
    plt.close()


def create_barplot_for_several_metrics(data, metric_collections, plots_dir, hue, prefix):
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

        handles, labels = ax[0].get_legend_handles_labels()
        plt.figlegend(handles, labels, title=hue, ncol=2, fontsize=14, loc='center', bbox_to_anchor=(0.5, 0.9))

        plt.tight_layout(pad=6.0)
        plt.savefig(f'{plots_dir}/{prefix}_{"_".join(metric_collection)}_{hue}.png', dpi=300, facecolor='white')
        plt.close()


def create_lineplot_per_max(data, metric, plots_dir, hue, style, prefix):
    plotdata = data.copy()
    if 'Time' in metric:
        plotdata = plotdata.sort_values(by=metric, ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(data=plotdata,
                 x='Maximum Sequence Length',
                 y=metric,
                 hue=hue,
                 style=style,
                 ci=None)

    # adjust text
    plt.xlabel('Maximum Sequence Length', fontsize=18)
    plt.ylabel(metric, fontsize=18)
    plt.xticks(fontsize=14)
    plt.locator_params(axis='x', nbins=plotdata['Maximum Sequence Length'].nunique())
    plt.yticks(fontsize=14)
    plt.rcParams['legend.title_fontsize'] = 14
    ax.legend(fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(f'{plots_dir}/{prefix}_{metric.replace(" ", "_")}_per_max_{hue}_{style}.png', dpi=300, facecolor='white')
    plt.close()
