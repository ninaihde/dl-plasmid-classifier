#!/bin/sh
#SBATCH --chdir=/hpi/fs00/home/nina.ihde/ma
#SBATCH --job-name="install"
#SBATCH --constraint="ARCH:X86"
#SBATCH --mem=64G
#SBATCH --mail-user=Nina.Ihde@student.hpi.uni-potsdam.de
#SBATCH --mail-type=ALL
#SBATCH --output="/hpi/fs00/home/nina.ihde/ma/logs/install_DeepSimulator.txt"
#SBATCH --error="/hpi/fs00/home/nina.ihde/ma/logs/install_DeepSimulator.txt"

#-> 1. install tensorflow_cdpm
conda remove --name tensorflow_cdpm --all -y
conda create --name tensorflow_cdpm python=2.7 -y
conda activate tensorflow_cdpm
pip install tensorflow==1.2.1 tflearn==0.3.2 tqdm==4.19.4 scipy==0.18.1 h5py==2.7.1 numpy==1.13.1 scikit-learn==0.20.3 biopython==1.74
conda deactivate

#-> 2. install basecaller
#--| 2.1 install albacore_2.3.1
#DeepSimulator/base_caller/albacore_2.3.1/download_and_install.sh

#--| 2.2 install guppy_3.1.5
#DeepSimulator/base_caller/guppy_3.1.5/download_and_install.sh
