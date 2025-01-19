"""
Module for encoding DNA sequence going into models
"""

import numpy as np
from Bio import SeqIO

ALPHABET = "ATGC"
BASE_IND = {base: ind for ind, base in enumerate(ALPHABET)}
IND_BASE = {ind: base for ind, base in enumerate(ALPHABET)}


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


def one_hot_encode_sequence(sequence, mapping_dict=BASE_IND, alphabet=ALPHABET):
    """
    One hot encode the DNA sequence using the provided alphabet mapping
    """
    encoded = np.zeros([len(sequence), len(ALPHABET)])
    for ix, base in enumerate(sequence):
        encoded[ix, BASE_IND[base]] = 1

    return encoded
