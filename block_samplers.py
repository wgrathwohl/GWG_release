import torch
import torch.nn as nn
import torch.distributions as dists
import utils
import numpy as np
import itertools


def all_binary_choices(n):
    b = [0., 1.]
    it = list(itertools.product(b * n))
    return torch.tensor(it).float()


def hamming_ball(n, k):
    ball = [np.zeros((n,))]
    for i in range(k + 1)[1:]:
        it = itertools.combinations(range(n), i)
        for tup in it:
            vec = np.zeros((n,))
            for ind in tup:
                vec[ind] = 1.
            ball.append(vec)
    return ball


class BlockGibbsSampler(nn.Module):
    def __init__(self, dim, block_size, hamming_dist=None, fixed_order=False):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.hamming_dist = hamming_dist
        self.fixed_order = fixed_order
        self._inds = self._init_inds()

    def _init_inds(self):
        inds = list(range(self.dim))
        if not self.fixed_order:
            np.random.shuffle(inds)
        return inds

    def step(self, x, model):
        if len(self._inds) == 0:  # ran out of inds
            self._inds = self._init_inds()

        inds = self._inds[:self.block_size]
        self._inds = self._inds[self.block_size:]
        logits = []
        xs = []
        for c in itertools.product(*([[0., 1.]] * len(inds))):
            xc = x.clone()
            c = torch.tensor(c).float().to(xc.device)
            xc[:, inds] = c
            l = model(xc).squeeze()
            xs.append(xc[:, :, None])
            logits.append(l[:, None])

        logits = torch.cat(logits, 1)
        xs = torch.cat(xs, 2)
        dist = dists.OneHotCategorical(logits=logits)
        choices = dist.sample()

        x_new = (xs * choices[:, None, :]).sum(-1)
        return x_new


class HammingBallSampler(BlockGibbsSampler):
    def __init__(self, dim, block_size, hamming_dist, fixed_order=False):
        super().__init__(dim, block_size, hamming_dist, fixed_order=fixed_order)
        self.dim = dim
        self.block_size = block_size
        self.hamming_dist = hamming_dist
        self.fixed_order = fixed_order

    def step(self, x, model):
        if len(self._inds) == 0:  # ran out of inds
            self._inds = self._init_inds()

        inds = self._inds[:self.block_size]
        self._inds = self._inds[self.block_size:]
        # bit flips in the hamming ball
        H = torch.tensor(hamming_ball(len(inds), min(self.hamming_dist, len(inds)))).float().to(x.device)
        H_inds = list(range(H.size(0)))
        chosen_H_inds = np.random.choice(H_inds, x.size(0))
        changes = H[chosen_H_inds]
        u = x.clone()
        u[:, inds] = changes * (1. - u[:, inds]) + (1. - changes) * u[:, inds]  # apply sampled changes U ~ p(U | X)

        logits = []
        xs = []
        for c in H:
            xc = u.clone()
            c = torch.tensor(c).float().to(xc.device)[None]
            xc[:, inds] = c * (1. - xc[:, inds]) + (1. - c) * xc[:, inds]  # apply all changes
            l = model(xc).squeeze()
            xs.append(xc[:, :, None])
            logits.append(l[:, None])

        logits = torch.cat(logits, 1)
        xs = torch.cat(xs, 2)
        dist = dists.OneHotCategorical(logits=logits)
        choices = dist.sample()

        x_new = (xs * choices[:, None, :]).sum(-1)
        return x_new


class DiffSampler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / 2.

    def step(self, x, model):
        x_cur = x
        forward_delta = self.diff_fn(x_cur, model)
        cd_forward = dists.OneHotCategorical(logits=forward_delta.detach())
        changes = cd_forward.sample()

        lp_forward = cd_forward.log_prob(changes)

        x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)

        reverse_delta = self.diff_fn(x_delta.detach(), model)
        cd_reverse = dists.OneHotCategorical(logits=reverse_delta.detach())

        lp_reverse = cd_reverse.log_prob(changes)

        m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
        la = m_term + lp_reverse - lp_forward
        a = (la.exp() > torch.rand_like(la)).float()
        x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
        return x_cur


class HammingDiffSampler(nn.Module):
    def __init__(self, dim, hamming_dist=2):
        super().__init__()
        self.dim = dim
        self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / 2.
        self.H = torch.tensor(hamming_ball(dim, hamming_dist)).float()[1:]  # remove option to do nothing

    def step(self, x, model):
        H = self.H.to(x.device)
        x_cur = x
        forward_delta = self.diff_fn(x_cur, model)
        forward_logits = forward_delta @ H.t()

        cd_forward = dists.Categorical(logits=forward_logits.detach())
        changes = cd_forward.sample()

        lp_forward = cd_forward.log_prob(changes)

        x_changes = H[changes]
        x_delta = (1. - x_cur) * x_changes + x_cur * (1. - x_changes)

        reverse_delta = self.diff_fn(x_delta.detach(), model)
        reverse_logits = reverse_delta @ H.t()
        cd_reverse = dists.OneHotCategorical(logits=reverse_logits.detach())

        lp_reverse = cd_reverse.log_prob(changes)

        m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
        la = m_term + lp_reverse - lp_forward
        a = (la.exp() > torch.rand_like(la)).float()
        x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
        return x_cur