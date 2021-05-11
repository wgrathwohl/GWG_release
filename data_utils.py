import numpy as np
from evcouplings.align import Alignment, map_matrix
from evcouplings.couplings import model, protocol, tools, read_raw_ec_file
from evcouplings.compare import (
    PDB, DistanceMap, SIFTS, intra_dists,
    multimer_dists, coupling_scores_compared
)
import matplotlib.pyplot as plt
from evcouplings.couplings import MeanFieldCouplingsModel, MeanFieldDCA,\
    parse_header, regularize_frequencies, regularize_pair_frequencies, compute_covariance_matrix,\
    reshape_invC_to_4d, fields
from evcouplings.visualize import plot_contact_map, plot_context
import torch
import torch.nn as nn


class MyDCA:
    """
    Class that provides the functionality
    to infer evolutionary couplings from a given
    sequence alignment using mean field
    approximation.

    Important:
    The input alignment should be an a2m
    alignment with lower / upper columns
    and the target sequence as the first
    record.

    Attributes
    ----------
    _raw_alignment : Alignment
        The input alignment. This should be an
        a2m alignment with lower / upper columns
        and the target sequence as first record.
    index_list : np.array
        List of UniProt numbers of the target
        sequence (only upper case characters).
    alignment : Alignment
        A processed version of the given alignment
        (_raw_alignment) that is then used to
        infer evolutionary couplings using DCA.
    N : int
        The number of sequences (of the processed
        alignment).
    L : int
        The width of the alignment (again, this
        refers to the processed alignment).
    num_symbols : int
        The number of symbols of the alphabet used.
    covariance_matrix : np.array
        Matrix of size (L * (num_symbols-1)) x (L * (num_symbols-1))
        containing the co-variation of each character pair
        in any positions.
    covariance_matrix_inv : np.array
        Inverse of covariance_matrix.
    """
    def __init__(self, alignment):
        """
        Initialize direct couplings analysis by
        processing the given alignment.

        Parameters
        ----------
        alignment : Alignment
            Alignment with lower / upper columns
            and the target sequence as first record.
        """
        # input alignment
        self._raw_alignment = alignment

        # the first sequence of an a2m alignment
        # in focus mode is the target sequence
        target_seq = self._raw_alignment[0]

        # select focus columns as alignment columns
        # that are non-gapped and a upper
        # character in the target sequence
        focus_cols = np.array([
            c.isupper() and c not in [
                self._raw_alignment._match_gap,
                self._raw_alignment._insert_gap
            ]
            for c in target_seq
        ])

        # extract focus alignment
        focus_ali = self._raw_alignment.select(
            columns=focus_cols
        )

        # extract index list of the target sequence
        # (only focus columns)
        _, start, stop = parse_header(self._raw_alignment.ids[0])
        self.index_list = np.array(range(start, stop + 1))
        self.index_list = self.index_list[focus_cols]

        # find sequences that are valid,
        # i.e. contain only alphabet symbols
        np_alphabet = np.array(list(focus_ali.alphabet))
        valid_sequences = np.array([
            np.in1d(seq, np_alphabet).all()
            for seq in focus_ali.matrix
        ])

        # remove invalid sequences
        self.alignment = focus_ali.select(
            sequences=valid_sequences
        )

        # reset pre-calculated sequence weigths
        # and frequencies of the alignment
        self._reset()

    def _reset(self):
        """
        Reset pre-computed sequence weights and
        alignment frequencies as well as the
        covariance matrix and its inverse.

        Resetting becomes important, when the
        fit function is called more than once.
        """
        # reset theta-specific weights
        self.alignment.weights = None

        # also reset frequencies since these
        # were based on the weights (and the
        # given pseudo-count)
        self.alignment._frequencies = None
        self.alignment._pair_frequencies = None
        self.regularized_frequencies = None
        self.regularized_pair_frequencies = None

        # reset covariance matrix and its inverse
        self.covariance_matrix = None
        self.covariance_matrix_inv = None

    def couplings_model(self, J, h, theta=0.8, pseudo_count=0.5):
        """
        Run mean field direct couplings analysis.

        Parameters
        ----------
        J: Inferred J_ij matrix of Potts model
        h: Inferred h_i vector of Potts model
        theta : float, optional (default: 0.8)
            Sequences with pairwise identity >= theta
            will be clustered and their sequence weights
            downweighted as 1 / num_cluster_members.
        pseudo_count : float, optional (default: 0.5)
            Applied to frequency counts to regularize
            in the case of insufficient data availability.

        Returns
        -------
        MeanFieldCouplingsModel
            Model object that holds the inferred
            fields (h_i) and couplings (J_ij).
        """
        self._reset()

        # compute sequence weights
        # using the given theta
        self.alignment.set_weights(identity_threshold=theta)

        # compute column frequencies regularized by a pseudo-count
        # (this implicitly calculates the raw frequencies as well)
        self.regularize_frequencies(pseudo_count=pseudo_count)

        # compute pairwise frequencies regularized by a pseudo-count
        # (this implicitly calculates the raw frequencies as well)
        self.regularize_pair_frequencies(pseudo_count=pseudo_count)

        # compute the covariance matrix from
        # the column and pair frequencies
        self.compute_covariance_matrix()

        # coupling parameters are inferred
        # by inverting the covariance matrix
        self.covariance_matrix_inv = -np.linalg.inv(
            self.covariance_matrix
        )

        # reshape the inverse of the covariance matrix
        # to make eijs easily accessible
        J_ij = J

        # compute fields
        h_i = h

        #print(J_ij.shape, J_ij.dtype)
        #print(h_i.shape, h_i.dtype)

        return MeanFieldCouplingsModel(
            alignment=self.alignment,
            index_list=self.index_list,
            regularized_f_i=self.regularized_frequencies,
            regularized_f_ij=self.regularized_pair_frequencies,
            h_i=h_i, J_ij=J_ij,
            theta=theta,
            pseudo_count=pseudo_count
        )

    def regularize_frequencies(self, pseudo_count=0.5):
        """
        Returns single-site frequencies
        regularized by a pseudo-count of symbols
        in alignment.

        This method sets the attribute
        self.regularized_frequencies
        and returns a reference to it.

        Parameters
        ----------
        pseudo_count : float, optional (default: 0.5)
            The value to be added as pseudo-count.

        Returns
        -------
        np.array
            Matrix of size L x num_symbols containing
            relative column frequencies of all symbols
            regularized by a pseudo-count.
        """
        self.regularized_frequencies = regularize_frequencies(
            self.alignment.frequencies,
            pseudo_count=pseudo_count
        )
        return self.regularized_frequencies

    def regularize_pair_frequencies(self, pseudo_count=0.5):
        """
        Add pseudo-count to pairwise frequencies
        to regularize in the case of insufficient
        data availability.

        This method sets the attribute
        self.regularized_pair_frequencies
        and returns a reference to it.

        Parameters
        ----------
        pseudo_count : float, optional (default: 0.5)
            The value to be added as pseudo-count.

        Returns
        -------
        np.array
            Matrix of size L x L x num_symbols x num_symbols
            containing relative pairwise frequencies of all
            symbols regularized by a pseudo-count.
        """
        self.regularized_pair_frequencies = regularize_pair_frequencies(
            self.alignment.pair_frequencies,
            pseudo_count=pseudo_count
        )
        return self.regularized_pair_frequencies

    def compute_covariance_matrix(self):
        """
        Compute the covariance matrix.

        This method sets the attribute self.covariance_matrix
        and returns a reference to it.

        Returns
        -------
        np.array
            Reference to attribute self.convariance_matrix
        """
        self.covariance_matrix = compute_covariance_matrix(
            self.regularized_frequencies,
            self.regularized_pair_frequencies
        )
        return self.covariance_matrix

    def reshape_invC_to_4d(self):
        """
        "Un-flatten" inverse of the covariance
        matrix to allow easy access to couplings
        using position and symbol indices.

        Returns
        -------
        np.array
            Matrix of size L x L x
            num_symbols x num_symbols.
        """
        return reshape_invC_to_4d(
            self.covariance_matrix_inv,
            self.alignment.L,
            self.alignment.num_symbols
        )

    def fields(self):
        """
        Compute fields.

        Returns
        -------
        np.array
            Matrix of size L x num_symbols
            containing single-site fields.
        """
        return fields(
            self.reshape_invC_to_4d(),
            self.regularized_frequencies
        )

    def char_matrix(self):
        return self.alignment.matrix

    def int_matrix(self):
        if not hasattr(self.alignment, "int_matrix"):
            self.alignment.int_matrix = map_matrix(self.alignment.matrix, self.alignment.alphabet_map)
        return self.alignment.int_matrix

    def map_inds(self):
        return set(self.alignment.alphabet_map.values())

    def oh_matrix(self):
        if not hasattr(self.alignment, "oh_matrix"):
            int_matrix = self.int_matrix()
            vals = self.map_inds()
            n_vals = len(vals)
            oh_matrix = np.zeros((int_matrix.shape[0], int_matrix.shape[1], n_vals))
            for n in range(int_matrix.shape[0]):
                for l in range(int_matrix.shape[1]):
                    oh_matrix[n, l, int_matrix[n, l]] = 1.
            self.alignment.oh_matrix = oh_matrix
        return self.alignment.oh_matrix


def load_distmap(fname):
    return DistanceMap.from_file(fname)


def get_ecs(couplings_model):
    return couplings_model._calculate_ecs()


def plot_ecs(couplings_model, distmap_intra, distmap_multimer, save_name, num_ecs=100, ecs=None):
    if ecs is None:
        ecs = get_ecs(couplings_model)
    longrange_ecs = ecs.query("abs(i - j) > 5")
    show_ecs = longrange_ecs.iloc[:num_ecs]
    #print(show_ecs)
    #1/0
    with plot_context("Arial"):
        plt.figure(figsize=(10, 10))

        plot_contact_map(
            show_ecs, distmap_intra, distmap_multimer
        )
        plt.savefig(save_name)



def add_precision(ec_table, dist_cutoff=5, score="cn",
                  min_sequence_dist=6, target_column="precision",
                  dist_column="dist"):
    """
    Compute precision of evolutionary couplings as predictor
    of 3D structure contacts

    Parameters
    ----------
    ec_table : pandas.DataFrame
        List of evolutionary couplings
    dist_cutoff : float, optional (default: 5)
        Upper distance cutoff (in Angstrom) for a
        pair to be considered a true positive contact
    score : str, optional (default: "cn")
        Column which contains coupling score. Table will
        be sorted in descending order by this score.
    min_sequence_dist : int, optional (default: 6)
        Minimal distance in primary sequence for an EC to
        be included in precision calculation
    target_column : str, optional (default: "precision")
        Name of column in which precision will be stored
    dist_column : str, optional (default: "dist")
        Name of column which contains pair distances

    Returns
    -------
    pandas.DataFrame
        EC table with added precision values as a
        function of EC rank (returned table will be
        sorted by score column)
    """
    # make sure list is sorted by score
    ec_table = ec_table.sort_values(by=score, ascending=False)

    if min_sequence_dist is not None:
        ec_table = ec_table.query("abs(i - j) >= @min_sequence_dist")

    ec_table = ec_table.copy()

    # number of true positive contacts
    true_pos_count = (ec_table.loc[:, dist_column] <= dist_cutoff).cumsum()

    # total number of contacts with specified distance
    pos_count = ec_table.loc[:, dist_column].notnull().cumsum()

    ec_table.loc[:, target_column] = true_pos_count / pos_count
    return ec_table


def ecs_marix(couplings_model, distmap_intra, distmap_multimer, save_name=None):
    ecs = get_ecs(couplings_model)
    cc = coupling_scores_compared(
        ecs, distmap_intra, distmap_multimer,
        dist_cutoff=5,
        output_file=save_name
    )
    print(cc)
    return cc