"""
Module for handling dataset objects
"""

from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from Bio import SeqIO
import torch
from torch.utils.data import Dataset, DataLoader
import tn_motif.utils.encode as enc


# Dataset class for neighborhood sequence and counts
class DNADataset(Dataset):
    def __init__(self, sequences: List[str], counts: List[float]):
        """
        Args:
            sequences (list of str): List of DNA sequences.
            labels (list of float): List of continuous binding scores.
            sequence_length (int): The length to which all sequences will be padded or truncated.
        """
        self.sequences = sequences
        self.labels = counts
        self.sequence_length = len(sequences[0])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        encoded_sequence = enc.one_hot_encode_sequence(sequence)
        return torch.tensor(encoded_sequence, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        )


def get_reference_genome_sequence(path_ref_genome: str) -> str:
    """
    Function for loading reference genome sequence, given path of reference genome
    """

    gb_record = SeqIO.read(open(path_ref_genome, "r"), "genbank")

    return str(gb_record.seq)


def neighborhood_site(
    genome_seq: str, position: int, flank_length: int = 25, constant_seq: bool = False
) -> str:
    """
    Given the sequence for the genome and the position for a TA site,
    this function returns the neighborhood of the sequence, of
    length `flank_length` on each side (5` and 3`).

    - Does not includs the focal TA sequence by default. Set as true to get
      the entire sequence including the TA site
    """
    L = len(genome_seq)
    position = int(position)
    if position - flank_length < 0:
        left_neighborhood = (
            genome_seq[(position - flank_length) % L : -1] + genome_seq[:position]
        )
        right_neighborhood = genome_seq[position + 2 : position + 2 + flank_length]
    elif position + flank_length > L:
        left_neighborhood = genome_seq[position - flank_length : position]
        right_neighborhood = (
            genome_seq[position + 2 : L]
            + genome_seq[0 : position + flank_length + 2 - L]
        )
    else:
        left_neighborhood = genome_seq[position - flank_length : position]
        right_neighborhood = genome_seq[position + 2 : position + 2 + flank_length]

    if constant_seq is True:
        return left_neighborhood + "TA" + right_neighborhood
    else:
        return left_neighborhood + right_neighborhood


# Function for loading the mutant counts and genbank file
def process_inputs(
    path_counts_table: str,
    path_ref_genome: str,
    flank_length: int = 25,
    focal_TA_site: bool = False,
    count_col: str = "UMI_count_t0",
    position_col: str = "position",
    filter_sites_by: str = "neutral_gene",
    downsample_fraction: Optional[float] = None,
) -> Tuple[List[str], List[float]]:
    """
    Load the files containing the counts by position, and the reference genome.

    Return the following:
    - neighboring_sequence List[str]: sequences to be used as input for prediction
    - UMI corrected counts at site List[int]: target column to be predicted, log transformed
    """
    # load the reference sequence
    genome_seq = get_reference_genome_sequence(path_ref_genome=path_ref_genome)
    # load the counts
    counts_table = pd.read_csv(path_counts_table, compression="gzip")
    # downsample counts if specified:
    if downsample_fraction is not None:
        counts_table = counts_table.sample(frac=downsample_fraction)
    # only keep sites defined as True in filter_sites_by
    counts_table_filt = counts_table[counts_table[filter_sites_by]].copy()
    # get neighboring sequence for every site
    neighoring_sequence = [
        neighborhood_site(
            genome_seq=genome_seq,
            position=pos,
            flank_length=flank_length,
            constant_seq=focal_TA_site,
        )
        # only picking sites that are true in the filter_sites_by column
        for pos in counts_table_filt[position_col]
    ]
    # counts are log-transformed
    log_counts = np.log10(counts_table_filt[count_col] + 1)

    return neighoring_sequence, log_counts
