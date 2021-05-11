import torch
import torch.nn as nn
import torch.distributions as dists
import utils
import numpy as np


# Gibbs-With-Gradients for binary data
class DiffSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., step_size=1.0):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp


    def step(self, x, model, mask):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        if self.multi_hop:
            if self.fixed_proposal:
                delta = self.diff_fn(x, model)
                cd = dists.Bernoulli(probs=delta.sigmoid() * self.step_size)
                for i in range(self.n_steps):
                    changes = cd.sample()
                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
                    la = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
                self._ar = np.mean(a_s)
            else:
                for i in range(self.n_steps):
                    forward_delta = self.diff_fn(x_cur, model)
                    cd_forward = dists.Bernoulli(logits=(forward_delta * 2 / self.temp))
                    changes = cd_forward.sample()

                    lp_forward = cd_forward.log_prob(changes).sum(-1)

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)


                    reverse_delta = self.diff_fn(x_delta, model)
                    cd_reverse = dists.Bernoulli(logits=(reverse_delta * 2 / self.temp))

                    lp_reverse = cd_reverse.log_prob(changes).sum(-1)

                    m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    la = m_term + lp_reverse - lp_forward
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
                    m_terms.append(m_term.mean().item())
                    prop_terms.append((lp_reverse - lp_forward).mean().item())
                self._ar = np.mean(a_s)
                self._mt = np.mean(m_terms)
                self._pt = np.mean(prop_terms)
        else:
            if self.fixed_proposal:
                delta = self.diff_fn(x, model)
                cd = dists.OneHotCategorical(logits=delta)
                for i in range(self.n_steps):
                    changes = cd.sample()

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
                    la = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
                self._ar = np.mean(a_s)
            else:
                for i in range(self.n_steps):
                    forward_delta = self.diff_fn(x_cur, model) - 1e9 * mask
                    cd_forward = dists.OneHotCategorical(logits=forward_delta)
                    changes = cd_forward.sample()

                    lp_forward = cd_forward.log_prob(changes)

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)

                    reverse_delta = self.diff_fn(x_delta, model) - 1e9 * mask
                    cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

                    lp_reverse = cd_reverse.log_prob(changes)

                    m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    la = m_term + lp_reverse - lp_forward
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

        return x_cur


# Gibbs-With-Gradients variant which proposes multiple flips per step
class MultiDiffSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, temp=1., n_samples=1):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.temp = temp
        self.n_samples = n_samples
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp


    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            cd_forward = dists.OneHotCategorical(logits=forward_delta)
            changes_all = cd_forward.sample((self.n_samples,))

            lp_forward = cd_forward.log_prob(changes_all).sum(0)

            changes = (changes_all.sum(0) > 0.).float()

            x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
            self._phops = (x_delta != x).float().sum(-1).mean().item()

            reverse_delta = self.diff_fn(x_delta, model)
            cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

            lp_reverse = cd_reverse.log_prob(changes_all).sum(0)

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            a_s.append(a.mean().item())
            m_terms.append(m_term.mean().item())
            prop_terms.append((lp_reverse - lp_forward).mean().item())
        self._ar = np.mean(a_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)

        self._hops = (x != x_cur).float().sum(-1).mean().item()
        return x_cur


class PerDimGibbsSampler(nn.Module):
    def __init__(self, dim, rand=False):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 1.
        self.rand = rand

    def step(self, x, model):
        sample = x.clone()
        lp_keep = model(sample).squeeze()
        if self.rand:
            changes = dists.OneHotCategorical(logits=torch.zeros((self.dim,))).sample((x.size(0),)).to(x.device)
        else:
            changes = torch.zeros((x.size(0), self.dim)).to(x.device)
            changes[:, self._i] = 1.

        sample_change = (1. - changes) * sample + changes * (1. - sample)

        lp_change = model(sample_change).squeeze()

        lp_update = lp_change - lp_keep
        update_dist = dists.Bernoulli(logits=lp_update)
        updates = update_dist.sample()
        sample = sample_change * updates[:, None] + sample * (1. - updates[:, None])
        self.changes[self._i] = updates.mean()
        self._i = (self._i + 1) % self.dim
        self._hops = (x != sample).float().sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.


class PerDimMetropolisSampler(nn.Module):
    def __init__(self, dim, n_out, rand=False):
        super().__init__()
        self.dim = dim
        self.n_out = n_out
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._j = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 0.
        self.rand = rand

    def step(self, x, model):
        if self.rand:
            i = np.random.randint(0, self.dim)
        else:
            i = self._i

        logits = []
        ndim = x.size(-1)

        for k in range(ndim):
            sample = x.clone()
            sample_i = torch.zeros((ndim,))
            sample_i[k] = 1.
            sample[:, i, :] = sample_i
            lp_k = model(sample).squeeze()
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        dist = dists.OneHotCategorical(logits=logits)
        updates = dist.sample()
        sample = x.clone()
        sample[:, i, :] = updates
        self._i = (self._i + 1) % self.dim
        self._hops = ((x != sample).float().sum(-1) / 2.).sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.


# Gibbs-With-Gradients for categorical data
class DiffSamplerMultiDim(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, temp=1.):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.temp = temp
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function_multi_dim(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function_multi_dim(x, m) / self.temp

    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []


        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            # make sure we dont choose to stay where we are!
            forward_logits = forward_delta - 1e9 * x_cur
            #print(forward_logits)
            cd_forward = dists.OneHotCategorical(logits=forward_logits.view(x_cur.size(0), -1))
            changes = cd_forward.sample()

            # compute probability of sampling this change
            lp_forward = cd_forward.log_prob(changes)
            # reshape to (bs, dim, nout)
            changes_r = changes.view(x_cur.size())
            # get binary indicator (bs, dim) indicating which dim was changed
            changed_ind = changes_r.sum(-1)
            # mask out cuanged dim and add in the change
            x_delta = x_cur.clone() * (1. - changed_ind[:, :, None]) + changes_r

            reverse_delta = self.diff_fn(x_delta, model)
            reverse_logits = reverse_delta - 1e9 * x_delta
            cd_reverse = dists.OneHotCategorical(logits=reverse_logits.view(x_delta.size(0), -1))
            reverse_changes = x_cur * changed_ind[:, :, None]

            lp_reverse = cd_reverse.log_prob(reverse_changes.view(x_delta.size(0), -1))

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None, None] + x_cur * (1. - a[:, None, None])
            a_s.append(a.mean().item())
            m_terms.append(m_term.mean().item())
            prop_terms.append((lp_reverse - lp_forward).mean().item())
        self._ar = np.mean(a_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)

        self._hops = (x != x_cur).float().sum(-1).sum(-1).mean().item()
        return x_cur


class GibbsSampler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))

    def step(self, x, model):
        sample = x.clone()
        for i in range(self.dim):
            lp_keep = model(sample).squeeze()

            xi_keep = sample[:, i]
            xi_change = 1. - xi_keep
            sample_change = sample.clone()
            sample_change[:, i] = xi_change

            lp_change = model(sample_change).squeeze()

            lp_update = lp_change - lp_keep
            update_dist = dists.Bernoulli(logits=lp_update)
            updates = update_dist.sample()
            sample = sample_change * updates[:, None] + sample * (1. - updates[:, None])
            self.changes[i] = updates.mean()
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.
