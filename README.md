# Deep Learning-Based Plasmid Classifier

This project was developed within the scope of a Master Thesis at the Data Analytics and Computational Statistics chair 
of the Hasso Plattner Institute, University of Potsdam, Germany. It is based on raw electrical signals from Nanopore 
sequencers and classifies whether a bacterial sample, i.e. a DNA molecule, originates from a bacterial chromosome or a 
plasmid. This way, adaptive sampling of Nanopore sequencing data can be supported which can substitute expensive plasmid 
enrichment in the laboratory and thus facilitates plasmid sequencing. The architecture of our approach is based on the
[SquiggleNet project](https://github.com/welch-lab/SquiggleNet).

## Installation

To be able to execute the Python scripts of this repository, please create a virtual environment and install all 
requirements with the following command:

    pip install -r requirements.txt

To compare our approach against the alignment-based minimap2 tool, we installed the following tools:
  - guppy_basecaller (version 6.4.2)
  - minimap2 (version 2.24)
  - samtools (version 1.6)

## Preprocessing

The preprocessing is divided into 6 steps which should be executed in the order described as follows. It starts with the 
collection and cleaning of the data (step 1 and 2), continues with our simulation procedure (step 3 and 4) and ends with 
the preparation of train (step 5) and test data (step 6).

### 1. Get Data

The [`get_data.py`](src/preprocessing/get_data.py) script takes care of the real data (only used for testing and tuning 
purposes) and the reference data (needed for the simulation of training, validation and testing data). First, it moves 
already downloaded real data to the target folder. Here, we ensure that the class label ("plasmid"/"pos" or "chr"/"neg") 
is stored in the filenames. The ground truth labels of our real data were assigned with the help of the minimap2 tool. 
Unfortunately, in case of ambiguous mappings, two exactly equal reads with the same read ID were created - one for each 
class. This is why we delete duplicate reads. Third, the real data is split into tune and test data. Lastly, the script 
downloads all references and saves them in the correct data format (.fasta). 

**Note:** Make sure to adjust the [`get_data.py`](src/preprocessing/get_data.py) script for your own data and 
downloading procedure!

After running this script, [`check_megaplasmids.py`](src/preprocessing/check_megaplasmids.py) can be executed to filter 
out invalid plasmids, i.e. megaplasmids that have more than 450kbp. For our data, we only found two such megaplasmids, 
one of which was falsely classified and thus manually removed from the dataset.

### 2. Prepare Simulation

The [`prepare_simulation.py`](src/preprocessing/prepare_simulation.py) script cleans the reference data of both classes. 
In addition, it splits the plasmid references it into training, validation and test data based on the Jaccard similarity 
score, as we want to generalize our approach for novel plasmids. We analyzed the produced ``removed_contigs.csv`` with 
[`check_contig_cleaning.ipynb`](src/preprocessing/check_contig_cleaning.ipynb) but found the same megaplasmids as in 
[`check_megaplasmids.py`](src/preprocessing/check_megaplasmids.py) and no suspicious assemblies. 

Lastly, the simulation scripts [`Simulate_pos.R`](src/preprocessing/simulation/Simulate_pos.R) and 
[`Simulate_neg.R`](src/preprocessing/simulation/Simulate_neg.R) each need an RDS file containing at least 3 columns:
  - ``assembly_accession``: ID of each reference
  - ``fold1``: which dataset the reference belongs to (``"train"``, ``"val"`` or ``"test"``)
  - ``Pathogenic``: whether the reference represents the positive class or not (``True`` for plasmids, ``False``for chromosomes)

With [`prepare_simulation.py`](src/preprocessing/prepare_simulation.py), you can create these RDS files per class.

**Note:** We assume an already existing RDS file for the negative class which is only adjusted in the 
[`prepare_simulation.py`](src/preprocessing/prepare_simulation.py) script! 

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

### 4. Prepare Normalization (Post-Process Simulation)

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
data as torch tensors. 

Afterwards, the [`align_normalized_files.py`](src/preprocessing/align_normalized_files.py) script reduces the number of 
normalized .pt files for the class with a larger number of files to the amount of files that the other class has. The 
reduction is done by distributing all reads in the files to be removed evenly among the files to be kept, i.e. the reads 
are appended at the end of the files to be kept. This reduction must be executed for both the training and validation 
data to be used for training with the CustomDataLoader (which assumes the same amount of files per dataset).

### 6. Prepare Testing

In the last step of our preprocessing, we prepare both the real and simulated test data in 
[`prepare_testing.py`](src/preprocessing/prepare_testing.py). Therefore, the FAST5 files with the real and simulated 
test data are cut to a random sequence length, depending on the given minimum and maximum sequence length.

Since we compare our approach against the minimap2 tool, we need to base-call our test data to execute it with this 
tool. The base-calling, i.e. the conversion to FASTQ files, is performed with Guppy 
(see [`execute_minimap.py`](src/execute_minimap.py)).

## Training

The training can be performed with [`train.py`](src/train.py). The training and validation data of both classes is 
required as an input to this script. In contrast to SquiggleNet, we decided to accept directories for ``-pt``, ``-pv``, 
``-ct`` and ``-cv`` as the data in our use case is too large to only process one big .pt at once like SquiggleNet does. 
This is why we added the CustomDataLoader to the project, loading each file only once per epoch. An example call can 
look like this:

    python train.py -pt path/to/train/plasmids -pv path/to/validation/plasmids -ct path/to/train/chromosomes -cv path/to/validation/chromosomes

The training calculates different performance metrics after each epoch based on the validation data. If you want to see 
how your validation reads were classified during training (read ID to predicted label mapping), you can pass a .txt file 
with one read ID per row to the optional parameters ``-pid`` and ``-cid``. In addition, you can pass an already trained 
model to the parameter ``-i``. 

**Note:** Plasmid reads are labeled with 0 and chromosome reads with 1!

Based on the validation metric calculations, we observed that our model reached a lower balanced accuracy than expected. 
This is why we decided to calculate the optimal decision threshold with respect to the balanced accuracy for inference 
per maximum sequence length (see [`optimize.py`](src/optimize.py)). The optimization is based on a subset of the real 
data we created in [`get_data.py`](src/preprocessing/get_data.py).

## Testing

The inference is based on the real and simulated test data created in step 1, 3 and 4 of the preprocessing. 
[`classify.py`](src/classify.py) normalizes and pads the data with zeros like done for the train and validation data. 
In contrast to SquiggleNet, we create batches for a certain number of reads instead of loading at least one complete 
file per batch which can significantly degrade performance. Moreover, we make use of a custom decision threshold with a 
default of 0.5. An example call can look like this:

    python classify.py -m path/to/trained/model -i path/to/input -o path/to/output

To situate our approach in current research, [`execute_minimap.py`](src/execute_minimap.py) enables the execution of
minimap2. Like already mentioned, base-calling needs to be performed beforehand which is also done by this script.

## Evaluation

The inference results of our approach can be evaluated with [`evaluate_testing.py`](src/evaluation/evaluate_testing.py).
This script calculates different performance metrics, extracts logged measurements and creates corresponding figures. If 
minimap2's results are evaluated with [`evaluate_minimap.ipynb`](src/evaluation/evaluate_minimap.ipynb), 
[`evaluate_testing.py`](src/evaluation/evaluate_testing.py) can create figures based on both approaches. An 
example call can look like this:

    python evaluate_testing.py -d path/to/results -l path/to/logs -o path/to/evaluation/outputs
