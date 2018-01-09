# http://hplgit.github.io/bioinf-py/doc/pub/html/main_bioinf.html#basic-bioinformatics-examples-in-python

# Computing Frequencies
#
# Your genetic code is essentially the same from you are born until you die, and the same in your blood and your brain.
# Which genes that are turned on and off make the difference between the cells. This regulation of genes is orchestrated
# by an immensely complex mechanism, which we have only started to understand. A central part of this mechanism consists
#  of molecules called transcription factors that float around in the cell and attach to DNA, and in doing so turn
# nearby genes on or off. These molecules bind preferentially to specific DNA sequences, and this binding preference
# pattern can be represented by a table of frequencies of given symbols at each position of the pattern. More precisely,
# each row in the table corresponds to the bases A, C, G, and T, while column j reflects how many times the base appears
# in position j in the DNA sequence.

import random

import numpy as np


def generate_string(N, alphabet='ACGT'):
    return ''.join([random.choice(alphabet) for i in xrange(N)])


def freq_numpy(dna_list):
    frequency_matrix = np.zeros((4, len(dna_list[0])), dtype=np.int)
    base2index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for dna in dna_list:
        for index, base in enumerate(dna):
            frequency_matrix[base2index[base]][index] += 1

    return frequency_matrix


dna = generate_string(600000)
r = freq_numpy(dna)
