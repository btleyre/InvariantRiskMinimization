# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd

import penalties

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('-b', type=int, default=5000)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', action='store_true')
parser.add_argument('--new_invar_penalty', action='store_true')
parser.add_argument('--invar_penalty', action='store_true')
parser.add_argument('--sigma', type=float, default=16.0)
flags = parser.parse_args()

print('Flags:')
for k,v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

final_train_accs = []
final_test_accs = []
for restart in range(flags.n_restarts):
    print("Restart", restart)

    # Load MNIST, make train/val splits, and shuffle train set examples

    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    # Build environments

    def make_environment(images, labels, e):
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()
        def torch_xor(a, b):
            return (a-b).abs() # Assumes both inputs are either 0 or 1
        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (labels < 5).float()
        labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
        # Assign a color based on the label; flip the color with probability e
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
        return {
            'images': (images.float() / 255.).cuda(),
            'labels': labels[:, None].cuda()
        }

    envs = [
        make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
        make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
        make_environment(mnist_val[0], mnist_val[1], 0.9)
    ]

    # Our penalties won't allow us to process the
    # entire dataset in a single batch, so we'll
    # need to implement some kind of batching.
    env_0_dataset = torch.utils.data.TensorDataset(
      envs[0]['images'],
      envs[0]['labels'], torch.zeros(envs[0]['labels'].shape).cuda())

    env_0_loader = torch.utils.data.DataLoader(
        env_0_dataset, batch_size=int(flags.b), shuffle=True
        )

    env_1_dataset = torch.utils.data.TensorDataset(
      envs[1]['images'],
      envs[1]['labels'], torch.ones(envs[1]['labels'].shape).cuda())

    env_1_loader = torch.utils.data.DataLoader(
        env_1_dataset, batch_size=int(flags.b), shuffle=True
        )

    # Define and instantiate the model

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            if flags.grayscale_model:
                lin1 = nn.Linear(14 * 14, flags.hidden_dim)
            else:
                lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
            lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
            lin3 = nn.Linear(flags.hidden_dim, 1)
            for lin in [lin1, lin2, lin3]:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True))
            self._classifier = lin3
        def forward(self, input):
            if flags.grayscale_model:
                out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
            else:
                out = input.view(input.shape[0], 2 * 14 * 14)
            rep = self._main(out)
            out = self._classifier(rep)
            return out, rep

    mlp = MLP().cuda()

    # Define loss function helpers

    def mean_nll(logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y)

    def mean_accuracy(logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    def penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    # Train loop

    def pretty_print(*values):
        col_width = 13
        def format_val(v):
            if not isinstance(v, str):
                v = np.array2string(v, precision=5, floatmode='fixed')
            return v.ljust(col_width)
        str_values = [format_val(v) for v in values]
        print("     ".join(str_values))

    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

    pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')


    for step in range(flags.steps):
        for env_0_batch, env_1_batch in zip(env_0_loader, env_1_loader):
            env_0_images, env_0_labels, env_0_envs = env_0_batch
            env_1_images, env_1_labels, env_1_envs = env_1_batch

            # We'll divide our training into the original
            # IRM training code and our new code. The reason
            # for the division is that our penalty considers samples
            # examples from multiple environments.
            if flags.new_invar_penalty or flags.invar_penalty:
                # Calculate nll for both envs here and avg them.
                env_0_logits, env_0_reps = mlp(env_0_images)
                env_0_nll = mean_nll(env_0_logits, env_0_labels)
                env_0_acc = mean_accuracy(env_0_logits, env_0_labels)

                env_1_logits, env_1_reps = mlp(env_1_images)
                env_1_nll = mean_nll(env_1_logits, env_1_labels)
                env_1_acc = mean_accuracy(env_1_logits, env_1_labels)

                # Calculate averages, do backprop, etc.
                train_nll = torch.stack([env_0_nll, env_1_nll]).mean()
                train_acc = torch.stack([env_0_acc, env_1_acc]).mean()

                # Group representations, labels, and labels
                # for potential use with an invariance penalty.
                all_reps = torch.cat([env_0_reps, env_1_reps])
                all_labels= torch.cat([env_0_labels, env_1_labels])
                all_envs = torch.cat([env_0_envs, env_1_envs])


                penalty_val = None            
                # Invariance penalties: these can be composed
                if flags.new_invar_penalty:
                    penalty_val = -penalties.new_invariance_penalty(
                        all_reps, all_labels, all_envs, flags.sigma
                        )
                if flags.invar_penalty:
                    if penalty_val is None:
                        penalty_val = penalties.invariance_penalty(
                            all_reps, all_labels, all_envs, flags.sigma
                            )
                    else:
                        # Note that when we're composing this with the new
                        # penalty, we want it to be ADDED to the loss, so
                        # we'll subtract it from the current penalty.
                        penalty_val += penalties.invariance_penalty(
                            all_reps, all_labels, all_envs, flags.sigma
                        )
                train_penalty = penalty_val

            else:
                # Calculate the per-environment loss, acc, penalty
                env_0_logits, reps = mlp(env_0_images)
                env_0_nll = mean_nll(env_0_logits, env_0_labels)
                env_0_acc = mean_accuracy(env_0_logits, env_0_labels)
                env_0_penalty = penalty(env_0_logits, env_0_labels)

                env_1_logits, reps = mlp(env_1_images)
                env_1_nll = mean_nll(env_1_logits, env_1_labels)
                env_1_acc = mean_accuracy(env_1_logits, env_1_labels)
                env_1_penalty = penalty(env_1_logits, env_1_labels)

                # Calculate averages, do backprop, etc.
                train_nll = torch.stack([env_0_nll, env_1_nll]).mean()
                train_acc = torch.stack([env_0_acc, env_1_acc]).mean()
                train_penalty = torch.stack([env_0_penalty, env_1_penalty]).mean()

            # Also do some evaluation on the full test set.
            test_logits, reps = mlp(envs[2]['images'])
            envs[2]['nll'] = mean_nll(test_logits, envs[2]['labels'])
            envs[2]['acc'] = mean_accuracy(test_logits,envs[2]['labels'])
            envs[2]['penalty'] = penalty(test_logits, envs[2]['labels'])

            weight_norm = torch.tensor(0.).cuda()
            for w in mlp.parameters():
                weight_norm += w.norm().pow(2)

            loss = train_nll.clone()
            loss += flags.l2_regularizer_weight * weight_norm
            penalty_weight = (flags.penalty_weight
                              if step >= flags.penalty_anneal_iters else 1.0)
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                loss /= penalty_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            test_acc = envs[2]['acc']
            if step % 100 == 0:
                pretty_print(
                    np.int32(step),
                    train_nll.detach().cpu().numpy(),
                    train_acc.detach().cpu().numpy(),
                    train_penalty.detach().cpu().numpy(),
                    test_acc.detach().cpu().numpy()
                )

    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))

    """
    for step in range(flags.steps):
        for env in envs:
            logits = mlp(env['images'])
            env['nll'] = mean_nll(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            env['penalty'] = penalty(logits, env['labels'])

        train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
        train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
        train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm
        penalty_weight = (flags.penalty_weight 
                if step >= flags.penalty_anneal_iters else 1.0)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        test_acc = envs[2]['acc']
        if step % 100 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                train_penalty.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy()
            )

    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
    """