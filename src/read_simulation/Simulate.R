# Script based on read simulation by Jakub M. Bartoszewicz
# (see https://gitlab.com/dacs-hpi/deepac/-/tree/master/supplement_paper/Rscripts/read_simulation)

source("src/read_simulation/SimulationWrapper.R")

Workers <- 2

Do.TrainingData <- T
Do.ValidationData <- T
Do.TestData <- T
Do.Balance <- T
Do.Balance.test <- T  # TODO: ?
Do.GetSizes <- T
IMG.Sizes <- F
Do.Clean <- F
#Simulator <- "Mason"
# for nanopore
Simulator <- "DeepSimulator"
# Only affects Mason v0.x
AllowNsFromGenome <- F

#ReadLength <- 250
#MeanFragmentSize <- 600
#FragmentStdDev <- 60
#ReadMargin <- 10
# for nanopore
ReadLength <- 8000
MeanFragmentSize <- 8000
FragmentStdDev <- 0
ReadMargin <- 0

TotalTrainingReadNumber <- 2e07  # TODO
TotalValidationReadNumber <- 25e05  # TODO
TotalTestReadNumber <- 25e05  # TODO
Proportional2GenomeSize <- T
LogTransform <- F
testval.LogTransform <- F

do.Positive <- T
do.Negative <- F  # TODO: how to add paths for both classes? execute per class? how to deal with line 209 + 212?

# only needed for Illumina
pairedEnd <- F
test.pairedEnd <- F

FastaFileLocation <- ""  # TODO: ?
train.FastaFileLocation <- "/hpi/fs00/share/fg/renard/nina.ihde/data/simulation/train_ref_pos"
val.FastaFileLocation <- "/hpi/fs00/share/fg/renard/nina.ihde/data/simulation/val_ref_pos"
test.FastaFileLocation <- "/hpi/fs00/share/fg/renard/nina.ihde/data/simulation/test_ref_pos"
TrainingTargetDirectory <- "/hpi/fs00/share/fg/renard/nina.ihde/data/simulation/train_sim_pos"
ValidationTargetDirectory <- "/hpi/fs00/share/fg/renard/nina.ihde/data/simulation/val_sim_pos"
TestTargetDirectory <- "/hpi/fs00/share/fg/renard/nina.ihde/data/simulation/test_sim_pos"
FastaExtension <- "fasta"
#FilenamePostfixPattern <- "_"
# for the viral dataset
FilenamePostfixPattern <- "\\."  # TODO: ?

HomeFolder <- "/hpi/fs00/share/fg/renard/nina.ihde/"
ProjectFolder <- "data/simulation/"
IMGFile <- "metadata_pos_ref_cleaned.rds"
IMGFile.new <- "metadata_pos_ref_cleaned_sizes.rds"

# install packages not yet installed
packages <- c("foreach", "doParallel")
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == F)) {
  install.packages(packages[!installed_packages])
}

# hide package loading output
invisible(lapply(packages, library, character.only=T))

if (Do.Clean) {

    FastaFiles <- system(paste0("find ", file.path(FastaFileLocation), " -type f -name '*", FastaExtension, "'"), intern=T)
    # ignore old temp files
    FastaFiles <- FastaFiles[!grepl("\\.temp\\.", FastaFiles)]

    library(foreach)
    cat(paste("###Cleaning###\n"))

    Check <- foreach(f=FastaFiles) %do% {
        cat(paste(f, "\n"))
        tempFasta <- sub(paste0("[.]", FastaExtension), paste0(".temp.", FastaExtension), f)
        # 6 std devs in NEAT
        if (pairedEnd) {
            min.contig <- MeanFragmentSize + 6 * FragmentStdDev + ReadMargin
        } else {
            min.contig <- ReadLength + ReadMargin
        }
        status <- system(paste("bioawk -cfastx '{if(length($seq) > ", min.contig, " ) {print \">\"$name \" \" $comment;print $seq}}'", f, ">", tempFasta))

        if (status != 0) {
            cat(paste("ERROR\n"))
        }
        if (file.info(tempFasta)$size > 0) {
            system(paste("cat", tempFasta, ">", f))
        } else {
            cat(paste0("WARNING: all contigs of ", basename(f), " are shorter than ", min.contig, ". Using the longest contig.\n"))
            status <- system(paste("bioawk -cfastx 'length($seq) > max_length {max_length = length($seq); max_name=$name; max_comment=$comment; max_seq = $seq} END{print \">\"max_name \" \" max_comment;print max_seq}'", f, ">", tempFasta))
            if (file.info(tempFasta)$size > 0) {
                system(paste("cat", tempFasta, ">", f))
            } else {
                cat(paste("ERROR\n"))
            }
        }
        file.remove(tempFasta)
    }

    test.FastaFiles <- system(paste0("find ", file.path(test.FastaFileLocation), " -type f -name '*", FastaExtension, "'"), intern=T)
    # ignore old temp files
    test.FastaFiles <- test.FastaFiles[!grepl("\\.temp\\.", test.FastaFiles)]

    Check <- foreach(f=test.FastaFiles) %do% {
        cat(paste(f, "\n"))
        tempFasta <- sub(paste0("[.]", FastaExtension), paste0(".temp.", FastaExtension), f)
        # 6 std devs in NEAT
        if (test.pairedEnd) {
            min.contig <- MeanFragmentSize + 6 * FragmentStdDev + ReadMargin
        } else {
            min.contig <- ReadLength + ReadMargin
        }
        status <- system(paste("bioawk -cfastx '{if(length($seq) > ", min.contig, " ) {print \">\"$name \" \" $comment;print $seq}}'", f, ">", tempFasta))

        if (status != 0) {
            cat(paste("ERROR\n"))
        }
        if (file.info(tempFasta)$size > 0) {
            system(paste("cat", tempFasta, ">", f))
        } else {
             cat(paste0("WARNING: all contigs of ", basename(f), " are shorter than ", min.contig, ". Using the longest contig.\n"))
             status <- system(paste("bioawk -cfastx 'length($seq) > max_length {max_length = length($seq); max_name=$name; max_comment=$comment; max_seq =     $seq} END{print \">\"max_name \" \" max_comment;print max_seq}'", f, ">", tempFasta))
             if (file.info(tempFasta)$size > 0) {
                 system(paste("cat", tempFasta, ">", f))
             } else {
                 cat(paste("ERROR\n"))
             }
        }
        file.remove(tempFasta)
    }
    cat(paste("###Cleaning done###\n"))
}

if (Do.GetSizes) {
    IMGdata <- readRDS(file.path(HomeFolder, ProjectFolder, IMGFile))
    calcSize <- function(x) {
        if (x$fold1 == "test") {
            file.loc <- test.FastaFileLocation
        } else {
            file.loc <- FastaFileLocation
        }

        if (x$fold1 == "train") {
           do.LogTransform <- LogTransform
        } else {
           do.LogTransform <- testval.LogTransform
        }

        if (do.LogTransform) {
            # a bug that was here before 09/05/22 would lead to miscalculating the genome lenght in rare cases when an accession number was contained in another one
            return (log10(as.numeric(system(paste0("find ", file.loc, " -type f -name '", x$assembly_accession, FilenamePostfixPattern, "*' | xargs grep -v \">\" | wc | awk '{print $3-$1}'"), intern=T))))
        } else {
            # a bug that was here before 09/05/22 would lead to miscalculating the genome lenght in rare cases when an accession number was contained in another one
            return (as.numeric(system(paste0("find ", file.loc, " -type f -name '", x$assembly_accession, FilenamePostfixPattern, "*' | xargs grep -v \">\" | wc | awk '{print $3-$1}'"), intern=T)))
        }
    }

    IMGdata$Genome.Size <- sapply(1:nrow(IMGdata), function(i){calcSize(IMGdata[i, c("assembly_accession", "fold1")])})
    saveRDS(IMGdata, file.path(HomeFolder, ProjectFolder, IMGFile.new))
} else {
    IMGdata <- readRDS(file.path(HomeFolder, ProjectFolder, IMGFile))
}

if (IMG.Sizes) {
    IMGdata$Genome.Size <- as.numeric(IMGdata$Genome.Size.....assembled)
}

if (Do.Balance) {
    TrainingReadNumber <- TotalTrainingReadNumber / 2 # per class across all genomes
    ValidationReadNumber <- TotalValidationReadNumber / 2 # per class across all genomes
    training.Fix.Coverage <- F
    validation.Fix.Coverage <- F
} else {
    TrainingReadNumber <- TotalTrainingReadNumber * ReadLength / sum(IMGdata$Genome.Size[IMGdata$fold1 == "train"]) # coverage
    ValidationReadNumber <- TotalValidationReadNumber * ReadLength / sum(IMGdata$Genome.Size[IMGdata$fold1 == "val"]) # coverage
    training.Fix.Coverage <- T
    validation.Fix.Coverage <- T
}
if (Do.Balance.test) {
    TestReadNumber <- TotalTestReadNumber / 2 # per class across all genomes
    test.Fix.Coverage <- F
} else {
    TestReadNumber <- TotalTestReadNumber * ReadLength / sum(IMGdata$Genome.Size[IMGdata$fold1 == "test"]) # coverage
    test.Fix.Coverage <- T
}

Simulate.Dataset <- function(SetName, ReadNumber, Proportional2GenomeSize, Fix.Coverage, ReadLength, pairedEnd,
                             FastaFileLocation, IMGdata, TargetDirectory, MeanFragmentSize=600, FragmentStdDev=60,
                             Workers=1, Simulator=c("Neat", "Mason", "Mason2"), FastaExtension=".fna",
                             FilenamePostfixPattern="_", ReadMargin=10, AllowNsFromGenome=F, RelativeGenomeSizes=F) {

    dir.create(file.path(TargetDirectory), showWarnings=F)

    GroupMembers <- IMGdata[IMGdata$fold1 == SetName,]

    GroupMembers_HP <- which(GroupMembers$Pathogenic)
    GroupMembers_NP <- which(!GroupMembers$Pathogenic)
    if (length(GroupMembers_HP) > 0 & do.Positive) {
        Check.train_HP <- Simulate.Reads.fromMultipleGenomes (Members=GroupMembers_HP, TotalReadNumber=ReadNumber, Proportional2GenomeSize=Proportional2GenomeSize, Fix.Coverage=Fix.Coverage, ReadLength=ReadLength, pairedEnd=pairedEnd, FastaFileLocation=FastaFileLocation, IMGdata=GroupMembers, TargetDirectory=file.path(TargetDirectory, "pathogenic"), FastaExtension=FastaExtension, MeanFragmentSize=MeanFragmentSize, FragmentStdDev=FragmentStdDev, Workers=Workers, Simulator=Simulator, FilenamePostfixPattern=FilenamePostfixPattern, ReadMargin=ReadMargin, AllowNsFromGenome=AllowNsFromGenome, RelativeGenomeSizes=RelativeGenomeSizes)
    }
    if (length(GroupMembers_NP) > 0 & do.Negative) {
        Check.train_NP <- Simulate.Reads.fromMultipleGenomes (Members=GroupMembers_NP, TotalReadNumber=ReadNumber, Proportional2GenomeSize=Proportional2GenomeSize, Fix.Coverage=Fix.Coverage, ReadLength=ReadLength, pairedEnd=pairedEnd, FastaFileLocation=FastaFileLocation, IMGdata=GroupMembers, TargetDirectory=file.path(TargetDirectory, "nonpathogenic"), FastaExtension=FastaExtension, MeanFragmentSize=MeanFragmentSize, FragmentStdDev=FragmentStdDev, Workers=Workers, Simulator=Simulator, FilenamePostfixPattern=FilenamePostfixPattern, ReadMargin=ReadMargin, AllowNsFromGenome=AllowNsFromGenome, RelativeGenomeSizes=RelativeGenomeSizes)
    }
}


# Simulate test reads
if (Do.TestData == T) {
    cat("###Simulating test data...###")
    Simulate.Dataset("test", TestReadNumber, Proportional2GenomeSize, test.Fix.Coverage, ReadLength, test.pairedEnd, test.FastaFileLocation, IMGdata, TestTargetDirectory, Workers=Workers, Simulator=Simulator, FastaExtension=FastaExtension, FilenamePostfixPattern=FilenamePostfixPattern, ReadMargin=ReadMargin, AllowNsFromGenome=AllowNsFromGenome, MeanFragmentSize=MeanFragmentSize, FragmentStdDev=FragmentStdDev, RelativeGenomeSizes=testval.LogTransform)
    cat("###Done!###")
}

# Simulate validation reads
# simulate for each class
if (Do.ValidationData == T) {
    cat("###Simulating validation data...###")
    Simulate.Dataset("val", ValidationReadNumber, Proportional2GenomeSize, validation.Fix.Coverage, ReadLength, pairedEnd, val.FastaFileLocation, IMGdata, ValidationTargetDirectory, Workers=Workers, Simulator=Simulator, FastaExtension=FastaExtension, FilenamePostfixPattern=FilenamePostfixPattern, ReadMargin=ReadMargin, AllowNsFromGenome=AllowNsFromGenome, MeanFragmentSize=MeanFragmentSize, FragmentStdDev=FragmentStdDev, RelativeGenomeSizes=testval.LogTransform)
    cat("###Done!###")
}

# Simulate training reads
# simulate for each class
if (Do.TrainingData == T) {
    cat("###Simulating training data...###")
    Simulate.Dataset("train", TrainingReadNumber, Proportional2GenomeSize, training.Fix.Coverage, ReadLength, pairedEnd, train.FastaFileLocation, IMGdata, TrainingTargetDirectory, Workers=Workers, Simulator=Simulator, FastaExtension=FastaExtension, FilenamePostfixPattern=FilenamePostfixPattern, ReadMargin=ReadMargin, AllowNsFromGenome=AllowNsFromGenome, MeanFragmentSize=MeanFragmentSize, FragmentStdDev=FragmentStdDev, RelativeGenomeSizes=LogTransform)
    cat("###Done!###")
}
