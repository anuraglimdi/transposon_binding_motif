# Transposon binding motif search using deep learning

Code for training and visualizing outputs for models predicting insertion bias of transposons from DNA sequence

## Usage
1. Clone this repository
2. Set up a conda environment using the following command

    `conda env create -f envs/environment.yaml`

3. Activate environment, navigate to repository directory and run

    `pip install -e .`

    This will install the package in editable mode.

4. Code can be accessed using package name `tn_motif`. Use autoreload to use package in editable mode if you're in Jupyter.

5. See `notebooks/run_models.ipynb` for details on how to use the package. Briefly, define model classes in `tn_motif/models/model_classes.py`. The `ModelTraining` object in `tn_motif/utils.models.py` can be used for k-fold cross validation and prediction on a holdout test set.

## Background

Transposon insertion sequencing (TnSeq) is widely used as a genetic screening method for microbial genomics research. Transposon, which are mobile genetic elements, can jump out of a cloning plasmid and insert into the genome, disrupting the expression of the gene. The mariner transposon is one such mobile element, with a site specificity for TA dinucleotides.

TnSeq data is often quite noisy with uneven coverage even within the same gene. Previously, this was attributed to PCR amplification biases. However, in my PhD research, I developed a method (UMI-TnSeq, [Code](https://github.com/anuraglimdi/umi_tnseq), [Paper](https://www.science.org/doi/abs/10.1126/science.add1417)), where I showed that unevenness in coverage does not stem from PCR bias, suggesting that the mariner transposon itself has binding preferences beyond the canonical TA motif.

### Why this matters

Genes are classified as essential if there are no transposon mutation counts mapping to it. If there are no reads within a gene of interest purely due to the nucleotide sequence, it would lead to incorrect classification.

More generally, different bacterial species have different genomic GC content. If the motif is disfavored in high GC genomes, using the mariner transposon for genetic screens may not be appropriate.

## Data

To remove the biases in counts to do mutant fitness, I restricted the analysis to sites that lie within non-essential genes (fitness when disrupted > -2.5%). Processed data from the [original publication](https://www.science.org/doi/abs/10.1126/science.add1417) is stored in `data`.

For model training, I define a neighborhood around the transposon site and one-hot encode this sequence (see `tn_motif/utils/dataset.py` and `tn_motif/utils/encode.py`) for details. 