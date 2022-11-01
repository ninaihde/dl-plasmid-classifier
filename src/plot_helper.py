import matplotlib.pyplot as plt
import seaborn as sns


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


def create_barplot_for_several_metrics(data, metric_collections, plots_dir, type_id):
    plotdata = data.copy()
    plotdata = plotdata.replace(['cutAfter_', 'cutBefore_'], ['', ''], regex=True)

    for metric_collection in metric_collections:
        # plot 4 metrics (2 x 2) at once
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(22, 16), sharey=True)
        plt.rcParams['legend.title_fontsize'] = 14
        ax = axes.flatten()

        for i, metric in enumerate(metric_collection):
            sns.barplot(data=plotdata,
                        x='ID',
                        y=metric,
                        hue='Cutting Method',
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
            ax[i].set_ylabel(metric, fontsize=18)
            ax[i].tick_params(axis='x', labelsize=14)
            ax[i].tick_params(axis='y', labelsize=14)

        plt.suptitle(f'Evaluation of {"Validation" if type_id == "val" else "Test"} Dataset', fontsize=24)
        handles, labels = ax[0].get_legend_handles_labels()
        plt.figlegend(handles, labels, title='Cutting Method', ncol=2, fontsize=14, loc='center',
                      bbox_to_anchor=(0.5, 0.9))

        plt.tight_layout(pad=6.0)
        plt.savefig(f'{plots_dir}/{"_".join(metric_collection)}_{type_id}.png', dpi=300, facecolor='white')
        plt.close()


def create_lineplot_per_max(data, metric, plots_dir, type_id):
    plotdata = data.copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(data=plotdata,
                 x='Maximum Sequence Length',
                 y=metric,
                 hue='Cutting Method',
                 style='Number of Epochs')

    # adjust text
    plt.title(f'{metric.replace(" (min)", "")} per Maximum Sequence Length', fontsize=22)
    plt.xlabel('Maximum Sequence Length', fontsize=18)
    plt.ylabel(metric, fontsize=18)
    plt.xticks(fontsize=14, ticks=plotdata['Maximum Sequence Length'],
               labels=plotdata['Maximum Sequence Length'].astype(int).astype(str) + '000')  # spell out max_seq_length
    plt.locator_params(axis='x', nbins=plotdata['Maximum Sequence Length'].nunique())
    plt.yticks(fontsize=14)
    plt.rcParams['legend.title_fontsize'] = 14
    ax.legend(fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(f'{plots_dir}/{metric.replace(" (min)", "")}_per_max_{type_id}.png', dpi=300, facecolor='white')
    plt.close()
