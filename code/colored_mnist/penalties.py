"""Helper functions for kernel based penalties.

Some of this code was borrowed from a repo
associated with the On Calibration and Out-of-domain Generalization
paper. The repo can be found here:

https://anonymous.4open.science/r/OOD_Calibration/wilds/code/clove_fmow_finetune.py

"""
import copy

import torch


def get_matching_map(labels):
    new_labels = convert_binary_labels(labels)
    matching_map = new_labels.mm(new_labels.t())
    matching_classes = (matching_map > 0)
    num_entries = matching_classes.long().sum()
    return matching_classes, num_entries


def get_non_matching_map(labels):
    new_labels = convert_binary_labels(labels)
    matching_map = new_labels.mm(new_labels.t())
    matching_classes = (matching_map < 0)
    num_entries = matching_classes.long().sum()
    return matching_classes, num_entries


def kernel_matrix_gaussian(X, sigma=1, use_median=False):
    """Caclulates the symmetric matrix of Gaussian kernel values"""
    # print("X: {}".format(X))
    x_distances = torch.sum(X ** 2, -1).reshape((-1, 1))
    # print("X X transpose: {}".format(torch.mm(X, X.t())))
    # print("x_distances: {}".format(x_distances))
    pairwise_distances_x = -2 * torch.mm(X, X.t()) + x_distances + x_distances.t()

    if use_median:
        sigma = torch.median(pairwise_distances_x).detach()

    gamma = -1.0 / (sigma)
    #print("min pairwise dist:{}".format(torch.min(pairwise_distances_x)))
    return torch.exp(gamma * pairwise_distances_x)


def convert_binary_labels(labels):
    """Convert 0/1 labels to -1/1 labels"""
    zero_index = labels == 0
    new_labels = copy.deepcopy(labels)
    new_labels[zero_index] = -1
    # print("New labels: {}".format(new_labels))
    return new_labels


def naive_penalty(representations, labels, sigma=0.4,
                  one_sided=None):
    """Calculates the naive representation penalty.

    Args:
        representations: torch tensor. The representations.
        labels: torch tensor. The 0/1 binary labels
            corresponding to each representation.
        sigma: float. The sigma to be used in the
            Gaussian kernel.
        one_sided: str or None. 'non-match' or 'match'
            to calculate the penalty w.r.t. only the
            non-matching or matching samples,
            respectively.

    returns: float. The calculated naive penalty.
    """
    num_samples = representations.shape[0]
    kernel = kernel_matrix_gaussian(representations, sigma=sigma)
    new_labels = convert_binary_labels(labels)

    if one_sided is not None:
        if one_sided == 'match':
            # Set all the non-matching similarities
            # in the matrix to zero.
            non_matching_map, num_entries = get_non_matching_map(labels)
            kernel = kernel.masked_fill(non_matching_map, 0)
        elif one_sided == 'non-match':
            # Set all the matching similarities
            # in the matrix to zero.
            matching_map, num_entries = get_matching_map(labels)
            kernel = kernel.masked_fill(matching_map, 0)
        else:
            raise ValueError("{} is not a valid value".format(one_sided))

    grid = new_labels.mm(new_labels.t())
    print("Grid {}".format(grid))
    print(labels)
    return torch.sum(kernel*grid) / (num_samples**2)
    # return ((new_labels.t().mm(kernel.mm(new_labels))) / (num_samples ** 2))


def env_naive_penalty(representations, labels, env_labels, sigma=0.4):
    """Calculates the naive representation penalty.

    Args:
        representations: torch tensor. The representations.
        labels: torch tensor. The 0/1 binary labels
            corresponding to each representation.
        env_labels: torch tensor. The 0/1 binary labels
            denoting which environment each env comes from.
        sigma: float. The sigma to be used in the
            Gaussian kernel.
        one_sided: str or None. 'non-match' or 'match'
            to calculate the penalty w.r.t. only the
            non-matching or matching samples,
            respectively.

    returns: float. The calculated naive penalty.
    """
    kernel = kernel_matrix_gaussian(representations, sigma=sigma)
    new_labels = convert_binary_labels(labels)

    # Zero-out any entries corresponding to similarities
    # between samples within the same environment.
    matching_map, num_entries = get_matching_map(env_labels)
    ___, other_num = get_non_matching_map(env_labels)
    kernel = kernel.masked_fill(matching_map, 0)
    print("Num entries is {}".format(num_entries))
    print("other num is {}".format(other_num))
    print(kernel[matching_map])

    grid = new_labels.mm(new_labels.t())
    print("Grid {}".format(grid))
    print(labels)
    return torch.sum(kernel*grid) / (num_entries)
    # return ((new_labels.t().mm(kernel.mm(new_labels))) / (num_entries))


def invariance_penalty(representations, labels, env_labels, sigma=0.4):
    """Calculates invariance penalty.

    Args:
        representations: torch tensor. The representations.
        labels: torch tensor. The 0/1 binary labels
            corresponding to each representation.
        env_labels: torch tensor. The 0/1 binary labels
            denoting which environment each env comes from.
        sigma: float. The sigma to be used in the
            Gaussian kernel.

    returns: float. The calculated invariance penalty.
    """
    kernel = kernel_matrix_gaussian(representations, sigma=sigma)
    new_labels = convert_binary_labels(labels)
    ident = torch.eye(labels.shape[0], dtype=bool).cuda()

    # First, calculate the "expected labels".
    # This first involves creating an intra-env
    # distribution for each sample. Zero out
    # all values corresponding to pairs crossing environments,
    # as well as the diagonal.
    non_matching_map, other_num = get_non_matching_map(env_labels)
    matching_kernel_vals = kernel.masked_fill(
                            non_matching_map, 0
                            ).masked_fill(ident, 0)
    # print("Initial kernel: {}".format(kernel))
    # print(matching_kernel_vals)

    denominators = matching_kernel_vals.sum(axis=0)
    denominators += 1e-20
    # Add a small constant: NaNs are coming out around here
    # print("Denominators: {}".format(denominators))
    #print("Min kernel: {}".format(torch.min(kernel)))
    #print("Min denom: {}".format(torch.min(denominators)))
    prob_distribution = matching_kernel_vals/denominators

    # print("Prob sum: {}".format(prob_distribution.sum()))
    # print("Labels shape: {}".format(labels.shape[0]))
    #assert torch.isclose(prob_distribution.sum(),
    #                     torch.tensor(labels.shape[0]).float(),
    #                     rtol=0, atol=1e-03)

    # Finally, use the calculated distribution to calculate
    # the "expected within environment labels" for
    # each sample.
    expected_labels = (prob_distribution*new_labels).sum(axis=0, keepdim=True)

    # print("expected labels: {}".format(expected_labels))
    # print("expected labels shape {}".format(expected_labels.shape))
    # print("Transpose array {}".format(expected_labels - expected_labels.t()))

    # Next, calculate the absolute difference of
    # expected labels for all possible pairs.
    abs_diff_expectations = torch.abs(expected_labels - expected_labels.t())

    # Finally, we'll multiply this matrix element-wise with
    # the original kernel values, this time zero-ing
    # out intra-class comparisons and the diagonal!
    matching_map, other_num = get_matching_map(env_labels)
    non_matching_kernel_vals = kernel.masked_fill(
                            matching_map, 0
                            ).masked_fill(ident, 0)

    # print("Abs diff {}".format(abs_diff_expectations))
    # print("non matching kernel values {}".format(non_matching_kernel_vals))

    return torch.sum(abs_diff_expectations*non_matching_kernel_vals)/(labels.shape[0]**2)


def new_invariance_penalty(representations, labels, env_labels, sigma=0.4):
    """Calculates invariance penalty.

    Args:
        representations: torch tensor. The representations.
        labels: torch tensor. The 0/1 binary labels
            corresponding to each representation.
        env_labels: torch tensor. The 0/1 binary labels
            denoting which environment each env comes from.
        sigma: float. The sigma to be used in the
            Gaussian kernel.

    returns: float. The calculated invariance penalty.
    """
    kernel = kernel_matrix_gaussian(representations, sigma=sigma)
    new_labels = convert_binary_labels(labels)
    ident = torch.eye(labels.shape[0], dtype=bool).cuda()

    # First, calculate the "expected labels".
    # This first involves creating an intra-env
    # distribution for each sample. Zero out
    # all values corresponding to pairs crossing environments,
    # as well as the diagonal.
    non_matching_map, other_num = get_non_matching_map(env_labels)
    matching_kernel_vals = kernel.masked_fill(
                            non_matching_map, 0
                            ).masked_fill(ident, 0)
    # print("Initial kernel: {}".format(kernel))
    # print(matching_kernel_vals)

    denominators = matching_kernel_vals.sum(axis=0)
    denominators += 1e-20
    # Add a small constant: NaNs are coming out around here
    # print("Denominators: {}".format(denominators))
    #print("Min kernel: {}".format(torch.min(kernel)))
    #print("Min denom: {}".format(torch.min(denominators)))
    prob_distribution = matching_kernel_vals/denominators

    # print("Prob sum: {}".format(prob_distribution.sum()))
    # print("Labels shape: {}".format(labels.shape[0]))
    #assert torch.isclose(prob_distribution.sum(),
    #                     torch.tensor(labels.shape[0]).float(),
    #                     rtol=0, atol=1e-03)

    # Finally, use the calculated distribution to calculate
    # the "expected within environment labels" for
    # each sample.
    expected_labels = (prob_distribution*new_labels).sum(axis=0, keepdim=True)
    abs_diff_normalizers = torch.abs(expected_labels) + torch.abs(expected_labels).t()
    # print("expected labels: {}".format(expected_labels))
    # print("expected labels shape {}".format(expected_labels.shape))
    # print("Transpose array {}".format(expected_labels - expected_labels.t()))

    # Next, calculate the absolute difference of
    # expected labels for all possible pairs.
    abs_diff_expectations = torch.abs(expected_labels - expected_labels.t())

    # Each term is 1 - the normalized diff
    new_terms = 1 - (abs_diff_expectations/abs_diff_normalizers)

    # Finally, we'll multiply this matrix element-wise with
    # the original kernel values, this time zero-ing
    # out intra-class comparisons and the diagonal!
    matching_map, other_num = get_matching_map(env_labels)
    non_matching_kernel_vals = kernel.masked_fill(
                            matching_map, 0
                            ).masked_fill(ident, 0)

    # print("Abs diff {}".format(abs_diff_expectations))
    # print("non matching kernel values {}".format(non_matching_kernel_vals))

    #return torch.sum(abs_diff_expectations*non_matching_kernel_vals)
    return torch.sum(new_terms*non_matching_kernel_vals)/(labels.shape[0]**2)


def representation_analysis(representations, labels, sigma=0.4):
    """Calculates the naive representation penalty.

    Args:
        representations: torch tensor. The representations.
        labels: torch tensor. The 0/1 binary labels
            corresponding to each representation.
        sigma: float. The sigma to be used in the
            Gaussian kernel.
    """
    # Create map of where the within-class
    # distances will be for each class individually
    one_samps = (labels.mm(labels.t()) == 1)
    zero_samps = ((labels - 1).mm((labels - 1).t()) == 1)
    new_labels = convert_binary_labels(labels)
    matching_map = new_labels.mm(new_labels.t())
    kernel = kernel_matrix_gaussian(representations, sigma=sigma)
    matching_classes = (matching_map == 1)
    non_matching_classes = (matching_map == -1)

    print("Matching distances: {}".format(
        kernel[matching_classes].sum()
    ))
    print("Non-matching distances: {}".format(
        kernel[non_matching_classes].sum()
    ))
    print("Class 0 distances: {}".format(
        kernel[zero_samps].sum()
    ))
    print("Class 1 distances: {}".format(
        kernel[one_samps].sum()
    ))

    return kernel[matching_classes].sum(), \
        kernel[non_matching_classes].sum(),\
        kernel[zero_samps].sum(),\
        kernel[one_samps].sum()
