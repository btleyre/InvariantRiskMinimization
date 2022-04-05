"""Helper functions for kernel based penalties.

Some of this code was borrowed from a repo
associated with the On Calibration and Out-of-domain Generalization
paper. The repo can be found here:

https://anonymous.4open.science/r/OOD_Calibration/wilds/code/clove_fmow_finetune.py

"""
import copy

import torch
import numpy as np
import matplotlib.pyplot as plt

# Global variables
KERNEL_SCALER = 1e10


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

    # Multiply by a constant for numerical stability.
    matching_kernel_vals = matching_kernel_vals*(KERNEL_SCALER)
    denominators = matching_kernel_vals.sum(axis=0)
    #denominators += 1e-20
    # denominators = torch.where(denominators == 0, denominators + 1e-20, denominators)
    #print(denominators)
    # denominators += 1e-20
    # Add a small constant: NaNs are coming out around here
    # print("Denominators: {}".format(denominators))
    #print("Min kernel: {}".format(torch.min(kernel)))
    #print("Min denom: {}".format(torch.min(denominators)))
    denominators = torch.where(denominators == 0, denominators + 1e-20, denominators)
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

    # Determine the number to divide by
    num_env_0 = (env_labels == 0).sum()
    num_env_1 = (env_labels == 1).sum()

    return torch.sum(abs_diff_expectations*non_matching_kernel_vals)/(num_env_0*num_env_1*2)


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

    # Multiply by a constant for numerical stability.
    matching_kernel_vals = matching_kernel_vals*(KERNEL_SCALER)
    denominators = matching_kernel_vals.sum(axis=0)
    #denominators += 1e-20
    # denominators = torch.where(denominators == 0, denominators + 1e-20, denominators)
    #print(denominators)
    # denominators += 1e-20
    # Add a small constant: NaNs are coming out around here
    # print("Denominators: {}".format(denominators))
    #print("Min kernel: {}".format(torch.min(kernel)))
    #print("Min denom: {}".format(torch.min(denominators)))
    denominators = torch.where(denominators == 0, denominators + 1e-20, denominators)
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
    # Where this array is zero, replace with a small constant, as we'll be dividing with it.
    abs_diff_normalizers = torch.where(abs_diff_normalizers == 0, abs_diff_normalizers + 1e-20, abs_diff_normalizers)
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

    # Determine the number to divide by
    num_env_0 = (env_labels == 0).sum()
    num_env_1 = (env_labels == 1).sum()

    #return torch.sum(abs_diff_expectations*non_matching_kernel_vals)
    return torch.sum(new_terms*non_matching_kernel_vals)/(num_env_0*num_env_1*2)


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


def kernel_analysis(representations, labels, env_labels,
                            shape_labels, colour_labels, sigma=0.4,
                            use_label=True, exper_name='exper'):
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
                            ).masked_fill(ident,0)

    # Multiply by constant to get stability
    matching_kernel_vals = matching_kernel_vals*(KERNEL_SCALER)
    denominators = matching_kernel_vals.sum(axis=0)
    # denominators += 1e-20
    # Add a small constant: NaNs are coming out around here
    denominators = torch.where(denominators == 0, denominators + 1e-20, denominators)
    prob_distribution = matching_kernel_vals/denominators

    # Now that we have the probability distributions, we want to see
    # what kinds of samples each sample puts high probability on.
    legend_dict = {}
    for label_val in [0, 1]:
        for env_val in [0, 1]:
            for shape_val in [0, 1]:
                for colour_val in [0, 1]:

                    if use_label:
                        legend = torch.logical_and(shape_labels == shape_val, 
                                    torch.logical_and(colour_labels == colour_val,
                                                torch.logical_and(env_labels.squeeze() == env_val, labels.squeeze() == label_val)))
                    else:
                        legend = torch.logical_and(shape_labels == shape_val, 
                                    torch.logical_and(colour_labels == colour_val, env_labels.squeeze() == env_val))

                    if use_label:
                        combo_str = "label_{}_env_{}_shape_{}_colour_{}".format(
                            label_val,
                            env_val,
                            shape_val,
                            colour_val
                        )
                    else:
                        combo_str = "env_{}_shape_{}_colour_{}".format(
                            env_val,
                            shape_val,
                            colour_val
                        )                        
                    legend_dict[combo_str] = legend

    expected_labels = (prob_distribution*new_labels).sum(axis=0, keepdim=True)

    intra_dict = intra_env_analysis(legend_dict, prob_distribution, expected_labels,
                            use_label=use_label, exper_name=exper_name)

    between_dict = between_env_analysis(legend_dict, kernel, expected_labels, labels, env_labels,
                            use_label=use_label, exper_name=exper_name)

    return intra_dict, between_dict


def intra_env_analysis(legend_dict, prob_distribution, expected_labels,
                            use_label=False, exper_name='exper'):

    # For each type of example, store a dictionary containing the metrics we
    # care about
    metric_dicts_dict = {}

    # First, we'll just print out the average expected label for each class of example.
    for sample_name, sample_legend in legend_dict.items():
        print(sample_name)
        # Expected labels is 1xbatch_size
        sample_expected_labels = expected_labels[:, sample_legend]
        print("mean: {}, std: {}".format(torch.mean(sample_expected_labels), torch.std(sample_expected_labels)))

        # Next, see which colour/shape combinations receive the largest
        # portion of the probability distribution
        sample_prob_dist = prob_distribution[:, sample_legend]
        env_str = 'env_0' if 'env_0' in sample_name else 'env_1'
        env_conditioned_dict = {}
        for other_sample_name, other_sample_legend in legend_dict.items():
            # Only consider samples in the same environment
            if env_str in other_sample_name:
                # Get the total probability for this sample type.
                other_sample_total_probs = sample_prob_dist[other_sample_legend, :].sum(axis=0)

                # Print out the mean and variance for both
                print(other_sample_name)
                print("Mean total prob: {}, Std total prob: {}".format(torch.mean(other_sample_total_probs), torch.std(other_sample_total_probs)))

                # Save both sets of values
                env_conditioned_dict[other_sample_name] = (torch.mean(other_sample_total_probs).detach().item(), torch.std(other_sample_total_probs).detach().item())
        
        # Finally, we'll make some bar graphs.
        # Loop over dictionary items this way to make sure everything is in the same order.
        group_names = []
        group_means = []
        group_stds = []
        for group_name, group_tup in env_conditioned_dict.items():
            group_names.append(group_name)
            group_means.append(group_tup[0])
            group_stds.append(group_tup[1])

        # Create a dictionary containing each group's metrics, and add that
        # to the dictionary containing the metrics for each sample type.
        metrics_dict = {}
        for idx, name in enumerate(group_names):
            metrics_dict[name] = {
                'mean': group_means[idx],
                'std': group_stds[idx],
            }

        metric_dicts_dict[sample_name] = metrics_dict
        
        """
        if not use_label:
            print(group_names)
            print(group_means)
            print(group_stds)
            fig, ax = plt.subplots()
            ax.bar(np.arange(len(group_names)), np.array(group_means), yerr=np.array(group_stds), align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylabel('Avg Total Prob')
            ax.set_xticks(np.arange(len(group_names)))
            ax.set_xticklabels(group_names)
            ax.set_title('{} Intra-Environment Probabilities'.format(sample_name))
            ax.yaxis.grid(True)

            # Save the figure and show
            plt.tight_layout()
            plt.savefig('./{}_{}_intra_env.png'.format(exper_name, sample_name))
        """

    return metric_dicts_dict


def between_env_analysis(legend_dict, kernel, expected_labels, labels, env_labels,
                         use_label=True, exper_name='exper'):
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
    ident = torch.eye(labels.shape[0], dtype=bool).cuda()

    # Zero out all kernel values that aren't
    # between environments
    matching_map, other_num = get_matching_map(env_labels)
    non_matching_kernel_vals = kernel.masked_fill(
                            matching_map, 0
                            ).masked_fill(ident, 0)

    # For each type of example, store a dictionary containing the metrics we
    # care about
    metric_dicts_dict = {}

    for sample_name, sample_legend in legend_dict.items():
        print(sample_name)
        # Expected labels is 1xbatch_size
        sample_expected_labels = expected_labels[:, sample_legend]
        # First, we'll just print out the average expected label for each class of example.
        print("mean: {}, std: {}".format(torch.mean(sample_expected_labels), torch.std(sample_expected_labels)))

        # Next, see which colour/shape combinations receive the largest
        # kernel values on average
        sample_kernel_val = non_matching_kernel_vals[:, sample_legend]
        env_str = 'env_1' if 'env_0' in sample_name else 'env_0'
        env_conditioned_kernel_dict = {}
        env_conditioned_exp_dict = {}
        for other_sample_name, other_sample_legend in legend_dict.items():
            # Only consider samples in the same environment
            if env_str in other_sample_name:
                # Get the average kernel value for this sample type.
                other_sample_kernel_means = sample_kernel_val[other_sample_legend, :].mean(axis=0)

                # Given that expected labels is 1xbatch_size, we can just use the original expected labels array.
                # Also don't need to calculate a mean over the first axis, as it's just a vector.
                other_sample_exp_means = expected_labels[:, other_sample_legend]

                # Print out the mean and variance for both
                print(other_sample_name)
                print("Mean kernel val: {}, Std kernel val: {}".format(torch.mean(other_sample_kernel_means), torch.std(other_sample_kernel_means)))
                print("Mean exp label: {}, Std exp label: {}".format(torch.mean(other_sample_exp_means), torch.std(other_sample_exp_means)))

                # Save both sets of values
                env_conditioned_kernel_dict[other_sample_name] = (torch.mean(other_sample_kernel_means).detach().item(), torch.std(other_sample_kernel_means).detach().item())
                env_conditioned_exp_dict[other_sample_name] = (torch.mean(other_sample_exp_means).detach().item(), torch.std(other_sample_exp_means).detach().item())
        
        # Finally, we'll make some bar graphs.
        # Loop over dictionary items this way to make sure everything is in the same order.
        group_names = []
        group_ker_means = []
        group_ker_stds = []
        group_exp_means = []
        group_exp_stds = []

        for ((group_ker_name, group_ker_tup), (group_exp_name, group_exp_tup)) in \
                zip(env_conditioned_kernel_dict.items(), env_conditioned_exp_dict.items()):
            
            # Make sure these two have the same name
            assert group_ker_name == group_exp_name

            group_names.append(group_ker_name)
            group_ker_means.append(group_ker_tup[0])
            group_ker_stds.append(group_ker_tup[1])

            group_exp_means.append(group_exp_tup[0])
            group_exp_stds.append(group_exp_tup[1])
        
        # Create a dictionary containing each group's metrics, and add that
        # to the dictionary containing the metrics for each sample type.
        metrics_dict = {}
        for idx, name in enumerate(group_names):
            metrics_dict[name] = {
                'ker_mean': group_ker_means[idx],
                'ker_std': group_ker_stds[idx],
                'exp_mean': group_exp_means[idx],
                'exp_std': group_exp_stds[idx],
            }

        metrics_dict['sample_expected_labels_mean'] = torch.mean(sample_expected_labels).detach().item()
        metric_dicts_dict[sample_name] = metrics_dict

        """
        if not use_label:
            print(group_names)
            fig, ax = plt.subplots()
            ax.bar(np.arange(len(group_names)), np.array(group_ker_means), align='center', alpha=0.5, ecolor='black', capsize=10)

            plt.ylim(0, 1.3*np.max(group_ker_means))
            # Write the expected labels below each bar
            for index, (exp_label, ker_mean) in enumerate(zip(group_exp_means, group_ker_means)):
                plt.text(index - 0.25, ker_mean, '{:.4g}'.format(exp_label), color='red', fontweight='bold')

            # Write the sample's own expected label somewhere in the middle.
            plt.text(1.25, 1.15*np.max(group_ker_means), 'Sample Exp Label {:.4g}'.format(torch.mean(sample_expected_labels)), color='green', fontweight='bold')

            ax.set_ylabel('Avg Kernel Val')
            ax.set_xticks(np.arange(len(group_names)))
            ax.set_xticklabels(group_names)
            ax.set_title('{} Between Env Kernel Vals'.format(sample_name))
            ax.yaxis.grid(True)

            # Save the figure and show
            plt.tight_layout()
            plt.savefig('./{}_{}_between_env.png'.format(exper_name, sample_name))
        """

    return metric_dicts_dict


def plot_within_env(group_names,
                    group_means,
                    group_stds,
                    exper_name,
                    sample_name,
                    use_label=False):

    # First, we'll replace the 'env_0' or 'env_1' in
    # the group names to make them fit on the x-axis.
    group_names = [name.replace('env_1', '').replace('env_0', '') for name in group_names]

    if not use_label:
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(group_names)), np.array(group_means), yerr=np.array(group_stds), align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Avg Total Prob')
        ax.set_xticks(np.arange(len(group_names)))
        ax.set_xticklabels(group_names)
        ax.set_title('{} Intra-Environment Probabilities'.format(sample_name))
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig('{}_{}_intra_env.png'.format(exper_name, sample_name))
        plt.clf()


def plot_between_env(group_names,
                     group_ker_means,
                     group_ker_stds,
                     group_exp_means,
                     group_exp_stds,
                     exper_name,
                     sample_name,
                     sample_expected_labels_mean,
                     use_label=False):
    
    # First, we'll replace the 'env_0' or 'env_1' in
    # the group names to make them fit on the x-axis.
    group_names = [name.replace('env_1', '').replace('env_0', '') for name in group_names]

    if not use_label:
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(group_names)), np.array(group_ker_means), align='center', alpha=0.5, ecolor='black', capsize=10)

        plt.ylim(0, 1.3*np.max(group_ker_means))
        # Write the expected labels below each bar
        for index, (exp_label, ker_mean) in enumerate(zip(group_exp_means, group_ker_means)):
            plt.text(index - 0.25, ker_mean, '{:.4g}'.format(exp_label), color='red', fontweight='bold')

        # Write the sample's own expected label somewhere in the middle.
        plt.text(1.25, 1.15*np.max(group_ker_means), 'Sample Exp Label {:.4g}'.format(sample_expected_labels_mean), color='green', fontweight='bold')

        ax.set_ylabel('Avg Kernel Val')
        ax.set_xticks(np.arange(len(group_names)))
        ax.set_xticklabels(group_names)
        ax.set_title('{} Between Env Kernel Vals'.format(sample_name))
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig('{}_{}_between_env.png'.format(exper_name, sample_name))
        plt.clf()
