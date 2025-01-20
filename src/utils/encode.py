"""
Module for encoding DNA sequence going into models
"""

import numpy as np

ALPHABET = "ATGC"
BASE_IND = {base: ind for ind, base in enumerate(ALPHABET)}
IND_BASE = {ind: base for ind, base in enumerate(ALPHABET)}


def one_hot_encode_sequence(sequence, mapping_dict=BASE_IND, alphabet=ALPHABET):
    """
    One hot encode the DNA sequence using the provided alphabet mapping
    """
    encoded = np.zeros([len(sequence), len(ALPHABET)])
    for ix, base in enumerate(sequence):
        encoded[ix, BASE_IND[base]] = 1

    return encoded
