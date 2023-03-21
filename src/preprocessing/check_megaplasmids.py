"""
This script checks the plasmid references for megaplasmids, i.e. plasmids that have more than 450kbp. Both,
non-megaplasmids and megaplasmids, are plotted with regard to their length. Finally, we extract all megaplasmids that
have a certain similarity to our chromosome contigs and check them manually afterwards. Actually, we only found two such
megaplasmids.

Note: Due to our manual checks, the megaplasmid with the ID "LR214949.1" and a length of 843,827 was removed from the
positive reference data as it is a falsely classified. It is part of the anomalous and contaminated assembly
"GCF_900660475.1" (see https://www.ncbi.nlm.nih.gov/assembly/GCF_900660475.1) but nevertheless included in the PLSDB.
"""

import click
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sourmash

from Bio import SeqIO


def get_seq_len(dir):
    lengths = dict()

    for file in glob.glob(f'{dir}/*.fasta'):
        with open(file, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                if file not in lengths:
                    lengths[file] = len(record.seq)
                else:
                    lengths[file] += len(record.seq)

    return lengths


def plot_distribution(plotdata, title, figure_dir, bins):
    _, ax = plt.subplots(figsize=(16, 8))
    sns.histplot(plotdata, x='Length', ax=ax, bins=bins)
    plt.title(f'Distribution of {title} Lengths', fontsize=22)
    plt.ticklabel_format(style='plain', axis='x')
    plt.xlim(0, plotdata['Length'].max())
    plt.xlabel('Length', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{figure_dir}/{title.lower()}_lengths.png', dpi=300, facecolor='white', edgecolor='none')


def calculate_signatures(input_dir, megaplasmids=None):
    signatures = list()

    if megaplasmids:
        files = [f'{input_dir}/{mp}' for mp in megaplasmids]
    else:
        files = glob.glob(f'{input_dir}/*.fasta')

    for fasta_file in files:
        with open(fasta_file, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                mh = sourmash.MinHash(n=0, ksize=31, scaled=100)
                mh.add_sequence(str(record.seq), force=True)
                signatures.append((fasta_file, mh, record.id, len(record.seq)))

    return signatures


@click.command()
@click.option('--ref_neg_dir', '-n', type=click.Path(exists=True), required=True)
@click.option('--ref_pos_dir', '-p', type=click.Path(exists=True), required=True)
@click.option('--figure_dir', '-f', type=click.Path(exists=True), required=True)
@click.option('--sim_threshold', '-t', default=0.9)
def main(ref_neg_dir, ref_pos_dir, sim_threshold, figure_dir):
    species_lengths = get_seq_len(ref_neg_dir)

    plasmid_lengths = pd.DataFrame.from_dict(get_seq_len(ref_pos_dir))
    plasmid_lengths.to_csv(f'{ref_pos_dir}/plasmid_lengths.csv', index=False)
    megaplasmids = plasmid_lengths[plasmid_lengths['Length'] >= 450000]
    other_plasmids = plasmid_lengths[plasmid_lengths['Length'] < 450000]
    megaplasmids.to_csv(f'{ref_pos_dir}/megaplasmid_lengths.csv', index=False)

    plot_distribution(megaplasmids, 'Megaplasmid', figure_dir, 50)
    plot_distribution(other_plasmids, 'Non-Megaplasmid', figure_dir, 'auto')

    ref_neg_sigs = calculate_signatures(ref_neg_dir)
    megaplasmid_sigs = calculate_signatures(ref_pos_dir, megaplasmids)

    df = pd.DataFrame(columns=['Plasmid', 'Plasmid Length', 'Similarity', 'Species', 'Species Length', 'Contig',
                               'Contig Length'])
    for p_path, p_hash, _, p_len in megaplasmid_sigs:
        for n_path, n_hash, n_id, n_len in ref_neg_sigs:
            sim = n_hash.similarity(p_hash, downsample=True)
            if sim >= sim_threshold:
                df = pd.concat([df, pd.DataFrame([{'Plasmid': os.path.basename(p_path),
                                                   'Plasmid Length': p_len,
                                                   'Similarity': sim,
                                                   'Species': os.path.basename(n_path),
                                                   'Species Length': species_lengths[n_path],
                                                   'Contig': n_id,
                                                   'Contig Length': n_len}])])

    df.to_csv(f'{ref_pos_dir}/similar_megaplasmids.csv', index=False)


if __name__ == '__main__':
    main()
