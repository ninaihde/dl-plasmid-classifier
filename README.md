# Deep Learning-Based Plasmid Classifier

This project was developed within the scope of a Master Thesis at the Data Analytics and Computational Statistics chair 
of the Hasso Plattner Institute, University of Potsdam, Germany. It is based on raw electrical signals from Nanopore 
sequencers and classifies whether a bacterial sample, i.e. a DNA molecule, originates from a bacterial chromosome or a 
plasmid. This way, adaptive sampling of Nanopore sequencing data can be supported which substitutes expensive plasmid 
enrichment in the laboratory and thus facilitates plasmid sequencing.

## Installation

To be able to execute the Python scripts of this repository, please create a virtual environment and install all 
requirements with the following commands:

    pip install -r requirements.txt

## Preprocessing

The preprocessing is divided into 6 steps which should be executed in the order described as follows. It starts with the 
collection and cleaning of the data (step 1 and 2), continues with our simulation procedure (step 3 and 4) and ends with 
the normalization (step 5) and optional base-calling (step 6) step.

### 1. Get Data

The [`get_data.py`](src/preprocessing/get_data.py) script takes care of the real data (only used for testing) 
and the reference data (needed for the simulation of training, validation and testing data). First, it moves already 
downloaded real data to the respective folders. Here, we ensure that the class label ("plasmid"/"pos" or "chr"/"neg") is 
stored in the filenames. Second, the script downloads all references and saves them in the correct data format (.fasta). 

**Note:** Make sure to adjust the [`get_data.py`](src/preprocessing/get_data.py) script for your own data and 
downloading procedure!

After running this script, [`check_megaplasmids.py`](src/preprocessing/check_megaplasmids.py) can be executed to filter 
out invalid plasmids, i.e. megaplasmids that have more than 450kbp. For our data, we only found two such megaplasmids, 
one of which was falsely classified and thus manually removed from the dataset.

### 2. Prepare Simulation

The [`prepare_simulation.py`](src/preprocessing/prepare_simulation.py) script cleans the reference data of both classes. 
In addition, it splits the plasmid references it into training, validation and test data based on the Jaccard similarity 
score, as we want to generalize our approach for plasmids. We analyzed the produced ``removed_contigs.csv`` with 
[`check_contig_cleaning.ipynb`](src/preprocessing/check_contig_cleaning.ipynb) but found the same megaplasmids as in 
[`check_megaplasmids.py`](src/preprocessing/check_megaplasmids.py) and no suspicious assemblies. 

Lastly, the simulation scripts [`Simulate_pos.R`](src/preprocessing/simulation/Simulate_pos.R) and 
[`Simulate_neg.R`](src/preprocessing/simulation/Simulate_neg.R) each need an RDS file containing at least 3 columns:
  - ``assembly_accession``: ID of each reference
  - ``fold1``: which dataset the reference belongs to (``"train"``, ``"val"`` or ``"test"``)
  - ``Pathogenic``: whether the reference represents the positive class or not (``True`` for plasmids, ``False``for chromosomes)

With [`prepare_simulation.py`](src/preprocessing/prepare_simulation.py), you can create these RDS files per class.

**Note:** We assume an already existing RDS file for the negative class which is only adjusted in the 
[`prepare_simulation.py`](src/preprocessing/prepare_simulation.py), script! 

### 3. Simulation

The simulation of the reference data is done with the help of [DeepSimulator](https://github.com/liyu95/DeepSimulator) 
and a [wrapping workflow](https://gitlab.com/dacs-hpi/deepac/-/tree/master/supplement_paper/Rscripts/read_simulation) 
invented by Jakub M. Bartoszewicz. 

First, clone the [DeepSimulator](https://github.com/liyu95/DeepSimulator) project and exchange the ``deep_simulator.sh`` 
script in ``DeepSimulator/`` with [our adapted version](src/preprocessing/simulation/deep_simulator.sh) to avoid 
executing base-calling. Next, install DeepSimulator without installing the included base-callers:

    conda remove --name tensorflow_cdpm --all -y
    conda create --name tensorflow_cdpm python=2.7 -y
    conda activate tensorflow_cdpm
    pip install tensorflow==1.2.1 tflearn==0.3.2 tqdm==4.19.4 scipy==0.18.1 h5py==2.7.1 numpy==1.13.1 scikit-learn==0.20.3 biopython==1.74
    conda deactivate

For the execution of the simulation scripts written in R, we recommend setting up a conda environment like this:

    conda create -n r_venv
    conda activate r_venv
    conda install -c r r-essentials r-foreach r-doparallel
    conda install -c bioconda bioawk

Afterwards, you can adjust the parameters of the simulation scripts. In order to set ``FastaFileLocation`` correctly, 
you have to store the positive references for training and validation in a common parent folder. The negative reads will 
be split after the simulation which is why we have to define fewer paths here. The simulation scripts can be executed 
with the following commands:

    Rscript src/preprocessing/simulation/Simulate_pos.R
    Rscript src/preprocessing/simulation/Simulate_neg.R

### 4. Prepare Normalization

Before you are able to normalize the data, with e.g. different maximum sequence lengths, there are 3 steps that have to 
be done once on the simulated data and are handled by the [`prepare_normalization.py`](src/preprocessing/prepare_normalization.py) 
script. First, all single-fast5 files that were created by the simulation need to be merged together into compressed 
multi-fast5 files to avoid exceeding storage limits on your machine. Optionally, you can decide to remove the original 
directories with the huge amount of simulated data. Second, the simulated reads for the negative class have to be split 
into train, validation and test data. Finally, the script moves the merged and simulated test data for the positive 
class into the same directory as the simulated test data of the negative class. We recommend to use as many threads as 
possible for the merging and splitting procedure to speed up computation time.

### 5. Prepare Training

The [`prepare_training.py`](src/preprocessing/prepare_training.py) script normalizes all train and validation data using 
the z-score with the median absolute deviation. In addition, it performs cutting of the reads to a randomly chosen 
sequence length and padding of the reads to a fixed length called max_seq_len. Finally, it saves the train and validation 
data as torch tensors. For the testing datasets, only storing of the ground truth labels is performed. If several batches
are used, the training and validation tensor files can be merged with [`merge_tensors.py`](src/preprocessing/merge_tensors.py). 

### 6. Base-Calling

As the last step of the preprocessing, we base-called all our test data (real and simulated) with guppy. This enables 
the comparison of our approach against typical alignment-based methods like Minimap2. 

## Training

Todo.

## Testing

Todo.

## Evaluation

Todo.
