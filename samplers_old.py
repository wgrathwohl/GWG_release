import torch
import torch.nn as nn
import torch.distributions as dists
import utils
import numpy as np



def _ebm_helper(netEBM, x):
    x = x.clone().detach().requires_grad_(True)
    E_x = netEBM(x)
    logjoint_vect = E_x.squeeze()
    logjoint = torch.sum(logjoint_vect)
    logjoint.backward()
    grad_logjoint = x.grad
    return logjoint_vect, logjoint, grad_logjoint


def get_ebm_samples(netEBM, x_init, burn_in, num_samples_posterior,
                    leapfrog_steps, stepsize,
                    flag_adapt=1, hmc_learning_rate=.02, hmc_opt_accept=.67, acceptEBM=None):
    if type(stepsize) != float:
        assert flag_adapt == 0
        stepsize = stepsize[:, None]
    device = x_init.device
    bsz, x_size = x_init.size(0), x_init.size()[1:]
    n_steps = burn_in + num_samples_posterior
    acceptHist = torch.zeros(bsz, n_steps).to(device)
    logjointHist = torch.zeros(bsz, n_steps).to(device)
    samples = torch.zeros(bsz*num_samples_posterior, *x_size).to(device)
    current_x = x_init
    cnt = 0
    for i in range(n_steps):
        x = current_x
        p = torch.randn_like(current_x)
        current_p = p
        logjoint_vect, logjoint, grad_logjoint = _ebm_helper(netEBM, current_x)
        current_U = -logjoint_vect.view(-1, 1)
        if acceptEBM is None:
            current_U_A = current_U
        else:
            logjoint_vect_A, _, _ = _ebm_helper(acceptEBM, current_x)
            current_U_A = -logjoint_vect_A.view(-1, 1)
        grad_U = -grad_logjoint
        p = p - stepsize * grad_U / 2.0
        for j in range(leapfrog_steps):
            x = x + stepsize * p
            if j < leapfrog_steps - 1:
                logjoint_vect, logjoint, grad_logjoint = _ebm_helper(netEBM, x)
                proposed_U = -logjoint_vect
                grad_U = -grad_logjoint
                p = p - stepsize * grad_U
        logjoint_vect, logjoint, grad_logjoint = _ebm_helper(netEBM, x)
        proposed_U = -logjoint_vect.view(-1, 1)
        if acceptEBM is None:
            proposed_U_A = proposed_U
        else:
            logjoint_vect_A, _, _ = _ebm_helper(acceptEBM, x)
            proposed_U_A = -logjoint_vect_A.view(-1, 1)
        grad_U = -grad_logjoint

        p = p - stepsize * grad_U / 2.0
        p = -p
        current_K = 0.5 * (current_p**2).flatten(start_dim=1).sum(dim=1)
        current_K = current_K.view(-1, 1)       # should be size of B x 1
        proposed_K = 0.5 * (p**2).flatten(start_dim=1).sum(dim=1)
        proposed_K = proposed_K.view(-1, 1)     # should be size of B x 1
        unif = torch.rand(bsz).view(-1, 1).to(device)
        accept = unif.lt(torch.exp(current_U_A - proposed_U_A + current_K - proposed_K))
        accept = accept.float().squeeze()       # should be B x 1
        acceptHist[:, i] = accept
        ind = accept.nonzero().squeeze()
        try:
            len(ind) > 0
            current_x[ind, :] = x[ind, :]
            current_U[ind] = proposed_U[ind]
        except:
            print('Samples were all rejected...skipping')
            continue
        if i < burn_in and flag_adapt == 1:
            stepsize = stepsize + hmc_learning_rate * (accept.float().mean() - hmc_opt_accept) * stepsize
        elif i >= burn_in:
            samples[cnt*bsz: (cnt+1)*bsz, :] = current_x
            cnt += 1
        logjointHist[:, i] = -current_U.squeeze()
    acceptRate = acceptHist.mean(dim=1)
    return samples, acceptRate, stepsize



def threshold(x):
    return (x > 0.).float()


def soft_threshold(x, t=1.):
    return (x / t).sigmoid()


class RBF(nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()
        self.sigma = sigma

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY

class SVGD(nn.Module):
    def __init__(self, optim, kernel=RBF()):
        super().__init__()
        self.K = kernel
        self.optim = optim
        self._ess = 0.0

    def phi(self, X, log_prob):
        X = X.detach().requires_grad_(True)

        log_prob = log_prob(X)
        score_func = torch.autograd.grad(log_prob.sum(), X)[0]

        K_XX = self.K(X, X.detach())
        grad_K = -torch.autograd.grad(K_XX.sum(), X)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)
        return phi

    def discrete_phi(self, X, log_prob_t, log_prob_s):
        X = X.detach().requires_grad_(True)

        log_prob_t = log_prob_t(X)
        log_prob_s = log_prob_s(X)
        log_w = log_prob_s - log_prob_t
        w = log_w.softmax(0).detach()
        score_func = torch.autograd.grad(log_prob_s.sum(), X)[0] * w[:, None]
        self._ess = 1./(w**2).sum()


        K_XX = self.K(X, X.detach())
        Kw = K_XX * w[None, :]
        grad_K1 = -torch.autograd.grad(Kw.sum(), X, create_graph=True)[0]

        # Kw = K_XX * w[:, None]
        # grad_K2 = -torch.autograd.grad(Kw.sum(), X)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K1) #/ X.size(0)
        return phi

    def step(self, X, log_prob):
        self.optim.zero_grad()
        X.grad = -self.phi(X, log_prob)
        self.optim.step()

    def discrete_step(self, X, log_prob_t, log_prob_s):
        self.optim.zero_grad()
        X.grad = -self.discrete_phi(X, log_prob_t, log_prob_s)
        self.optim.step()


def update_logp(u, u_mu, std):
    return dists.Normal(u_mu, std).log_prob(u).flatten(start_dim=1).sum(1)


class RelaxationSampler(nn.Module):
    def __init__(self, dim, model, base="gaussian",
                 optimizer_fn=lambda p: torch.optim.Adam([p], lr=.01),
                 kernel=RBF(), n_particles=100):
        super().__init__()
        if base == "gaussian":
            self.base_dist = dists.Normal(0., 1.)
        else:
            raise ValueError

        self.optimizer_fn = optimizer_fn
        self.model = model
        self.kernel = kernel
        self.X = self.base_dist.sample((n_particles, dim)).to(model.W)
        self.optimizer = optimizer_fn(self.X)
        self.n_particles = n_particles
        self.dim = dim

    def threshold(self, x):
        return (x > 0.).float()

    def soft_threshold(self, x):
        return x.sigmoid()

    def target(self, x):
        base = self.base_dist.log_prob(x).sum(1)
        x_proj = self.threshold(x)
        m_cont = self.model(x_proj)#.squeeze()
        return base + m_cont

    def surrogate(self, x):
        base = self.base_dist.log_prob(x).sum(1)
        x_proj = self.soft_threshold(x)
        m_cont = self.model(x_proj)#.squeeze()
        return base + m_cont


    # def _update_grad(self):
    #     X = self.X.detach().requires_grad_(True)
    #     surr_log_prob = self.surrogate(X)
    #     score_func = torch.autograd.grad(surr_log_prob.sum(), X)[0]
    #     targ_log_prob = self.target(X)
    #     # compute weights
    #     log_w = surr_log_prob - targ_log_prob
    #     w = log_w.softmax(0).detach()
    #     # print(w)
    #     # print(log_w)
    #     # 1/0
    #
    #     K_XX = self.kernel(X, X.detach())# * w[None]
    #     grad_K = -torch.autograd.grad((K_XX.sum(1)).sum(), X)[0]
    #     print(K_XX)
    #     1/0
    #
    #     #print(K_XX.size(), score_func.size(), (K_XX @ score_func).size())
    #     #1/0
    #
    #     phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)
    #     #phi = score_func
    #     return phi

    def _update_grad(self):
        X = self.X.detach().requires_grad_(True)

        log_prob = self.surrogate(X)
        t_log_prob = self.target(X)
        log_w = log_prob - t_log_prob
        w = log_w.softmax(0).detach()
        score_func = torch.autograd.grad(log_prob.sum(), X)[0]

        K_XX = self.kernel(X, X.detach()) * w[None]
        grad_K = -torch.autograd.grad(K_XX.sum(), X)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) #/ X.size(0)

        return phi

    def step(self, k=1):
        for i in range(k):
            phi = self._update_grad()
            self.optimizer.zero_grad()
            self.X.grad = -phi
            self.optimizer.step()

    def reset(self, x=None):
        if x is None:
            x = self.base_dist.sample((self.n_particles, self.dim)).to(self.model.W)
        self.X = x
        self.optimizer = self.optimizer_fn(self.X)

    def output(self):
        return self.threshold(self.X)


class RelaxedMALA(nn.Module):
    def __init__(self, dim, model, base="gaussian", lr=.02, temp=1.):
        super().__init__()
        if base == "gaussian":
            self.base_dist = dists.Normal(0., 1.)
        else:
            raise ValueError

        self.lr = lr
        self.step_size = 1. / dim
        self.concrete_temp = 1.0
        self._ar = []
        self.dim = dim
        self.model = model
        self.temp = temp
        self._accept_rate = 0.

    def threshold(self, x):
        return (x > 0.).float()

    def soft_threshold(self, x):
        return (x * .1).sigmoid()

    def logp_target(self, x):
        base = self.base_dist.log_prob(x).sum(1)
        x_proj = self.threshold(x)
        m_cont = self.model(x_proj).squeeze()#.squeeze()
        return (base + m_cont) * self.temp

    def logp_surrogate(self, x):
        base = self.base_dist.log_prob(x).sum(1)
        x_proj = self.soft_threshold(x)
        m_cont = self.model(x_proj).squeeze()#.squeeze()
        return (base + m_cont) * self.temp

    def init(self, n):
        return self.base_dist.sample((n, self.dim))

    def init_from_data(self, x):
        x_c = self.base_dist.sample((x.size(0), self.dim)).to(x.device)
        x_c = x_c * torch.sign(x_c)  # make positive
        # [0, 1] --> [-1, 1]
        xp = 2 * x - 1
        x_out = xp * x_c
        return x_out

    def step(self, x, accept_dist="target"):
        x = x.detach().requires_grad_()
        logp_s = self.logp_surrogate(x)
        grad = torch.autograd.grad(logp_s.sum(), x)[0]
        step_std = (2 * self.step_size) ** .5

        update_mu = x + self.step_size * grad
        update = update_mu + step_std * torch.randn_like(update_mu)

        logp_updates = self.logp_surrogate(update)
        reverse_grad = torch.autograd.grad(logp_updates.sum(), update)[0]
        reverse_update_mu = update + self.step_size * reverse_grad

        logp_forward = update_logp(update, update_mu, step_std)
        logp_backward = update_logp(x, reverse_update_mu, step_std)
        if accept_dist == "surrogate":
            logp_accept = logp_updates + logp_backward - logp_s - logp_forward
        else:
            logp_t = self.logp_target(x)
            logp_updates_t = self.logp_target(update)
            logp_accept = logp_updates_t + logp_backward - logp_t - logp_forward

        p_accept = logp_accept.exp()
        accept = (torch.rand_like(p_accept) < p_accept).float()
        next_x = accept[:, None] * update + (1 - accept[:, None]) * x
        self._ar.append(accept.mean().item())
        return next_x

    def accept_rate(self):
        return self._accept_rate

    def update_step_size(self):
        ar = np.mean(self._ar)
        self.step_size = self.step_size + self.lr * (ar - .57) * self.step_size
        self._ar = []
        self._accept_rate = ar


class AnnealedSGLD(nn.Module):
    def __init__(self, dim, model, base="gaussian", t=1.):
        super().__init__()
        if base == "gaussian":
            self.base_dist = dists.Normal(0., 1.)
        else:
            raise ValueError

        self.step_size = 1. / dim
        self.concrete_temp = 1.0
        self._ar = []
        self.dim = dim
        self.model = model
        self._accept_rate = 0.
        self.t = t

    def threshold(self, x):
        return (x > 0.).float()

    def soft_threshold(self, x):
        return (x * .1).sigmoid()

    def logp_target(self, x):
        base = self.base_dist.log_prob(x).sum(1)
        x_proj = self.threshold(x)
        m_cont = self.model(x_proj).squeeze()#.squeeze()
        return (base + m_cont) * self.t

    def logp_surrogate(self, x, lam):
        base = self.base_dist.log_prob(x).sum(1)
        x_proj = self.soft_threshold(x / lam)
        m_cont = self.model(x_proj).squeeze()#.squeeze()
        return (base + m_cont) * self.t

    def init(self, n):
        return self.base_dist.sample((n, self.dim))

    def init_from_data(self, x):
        x_c = self.base_dist.sample((x.size(0), self.dim)).to(x.device)
        x_c = x_c * torch.sign(x_c)  # make positive
        # [0, 1] --> [-1, 1]
        xp = 2 * x - 1
        x_out = xp * x_c
        return x_out

    def step(self, x, step_size, lam):
        x = x.detach().requires_grad_()
        logp_s = self.logp_surrogate(x, lam)
        grad = torch.autograd.grad(logp_s.sum(), x)[0]
        step_std = (2 * step_size) ** .5

        update_mu = x + step_size * grad
        update = update_mu + step_std * torch.randn_like(update_mu)
        return update

    def steps(self, x, step_size, n_steps, max_lam):
        lams = np.linspace(0, max_lam, n_steps + 1)[1:][::-1]
        for lam in lams:
            x = self.step(x, step_size, lam)
        return x


class BinaryRelaxedModel(nn.Module):
    def __init__(self, dim, model, base="gaussian"):
        super().__init__()
        self.model = model
        if base == "gaussian":
            self.base_dist = dists.Normal(0., 1.)
        else:
            raise ValueError

        self.dim = dim

    def init(self, n):
        return self.base_dist.sample((n, self.dim))

    def init_from_data(self, x):
        x_c = self.base_dist.sample((x.size(0), self.dim)).to(x.device)
        x_c = x_c * torch.sign(x_c)  # make positive
        # [0, 1] --> [-1, 1]
        xp = 2 * x - 1
        x_out = xp * x_c
        return x_out

    def logp_target(self, x):
        base = self.base_dist.log_prob(x).sum(1)
        x_proj = threshold(x)
        m_cont = self.model(x_proj).squeeze()#.squeeze()
        return base + m_cont

    def logp_surrogate(self, x, t=1.):
        base = self.base_dist.log_prob(x).sum(1)
        x_proj = soft_threshold(x, t=t)
        m_cont = self.model(x_proj).squeeze()  # .squeeze()
        return base + m_cont

    def logp_accept_obj(self, x, step_size, t):
        x = x.detach().requires_grad_()
        logp_s = self.logp_surrogate(x, t=t)
        grad = torch.autograd.grad(logp_s.sum(), x, retain_graph=True, create_graph=True)[0]
        step_std = (2 * step_size) ** .5

        update_mu = x + step_size * grad
        update = update_mu + step_std * torch.randn_like(update_mu)

        logp_updates = self.logp_surrogate(update, t=t)
        reverse_grad = torch.autograd.grad(logp_updates.sum(), update, retain_graph=True, create_graph=True)[0]
        reverse_update_mu = update + step_size * reverse_grad

        logp_forward = update_logp(update, update_mu, step_std)
        logp_forward2 = update_logp(update.detach(), update_mu, step_std)
        logp_backward = update_logp(x, reverse_update_mu, step_std)

        logp_target_update = self.logp_target(update)
        logp_surrogate_update = self.logp_surrogate(update, t)
        rebar = (logp_target_update - logp_surrogate_update).detach() * logp_forward2 + logp_surrogate_update
        return rebar + logp_backward - logp_forward, (rebar,
                                                      logp_target_update, logp_surrogate_update,
                                                      logp_backward, logp_forward)

    def step(self, x, step_size, t=1., tt=1., accept_dist="target"):
        x = x.detach().requires_grad_()
        logp_s = self.logp_surrogate(x, t=t) * tt
        grad = torch.autograd.grad(logp_s.sum(), x)[0]
        step_std = (2 * step_size) ** .5

        update_mu = x + step_size * grad
        update = update_mu + step_std * torch.randn_like(update_mu)

        logp_updates = self.logp_surrogate(update, t=t) * tt
        reverse_grad = torch.autograd.grad(logp_updates.sum(), update)[0]
        reverse_update_mu = update + step_size * reverse_grad

        logp_forward = update_logp(update, update_mu, step_std)
        logp_backward = update_logp(x, reverse_update_mu, step_std)
        if accept_dist == "surrogate":
            logp_accept = logp_updates + logp_backward - logp_s - logp_forward
        else:
            logp_t = self.logp_target(x) * tt
            logp_updates_t = self.logp_target(update) * tt
            logp_accept = logp_updates_t + logp_backward - logp_t - logp_forward

        p_accept = logp_accept.exp()
        accept = (torch.rand_like(p_accept) < p_accept).float()
        next_x = accept[:, None] * update + (1 - accept[:, None]) * x
        return next_x, accept.mean().item()

    def hmc_step(self, x, step_size, n_steps, t=1., accept_dist="target"):
        if accept_dist == "surrogate":
            x, ar, step_size = get_ebm_samples(lambda x: self.logp_surrogate(x, t=t).squeeze(),
                                               x, n_steps, 1, 5, step_size)
        else:
            x, ar, step_size = get_ebm_samples(lambda x: self.logp_surrogate(x, t=t).squeeze(),
                                               x.detach().requires_grad_(), n_steps, 1, 5, step_size,
                                               acceptEBM=lambda x: self.logp_target(x).squeeze())
        return x, ar, step_size

    def annealed_hmc(self, x, step_sizes, n_steps, max_lam):
        n_lam = len(step_sizes)
        lams = np.linspace(0., max_lam, n_lam + 1)[1:][::-1]
        n_steps_per_lam = n_steps // n_lam
        ars = []
        for i in range(len(step_sizes)):
            lam = lams[i]
            step_size = step_sizes[i]
            if i < len(step_sizes) - 1:
                next_lam = lams[i + 1]
                x, ar, step_size = get_ebm_samples(lambda x: self.logp_surrogate(x, t=lam).squeeze(),
                                                   x, n_steps_per_lam, 1, 5, step_size,
                                                   acceptEBM=lambda x: self.logp_surrogate(x, t=next_lam).squeeze())
            else:
                x, ar, step_size = get_ebm_samples(lambda x: self.logp_surrogate(x, t=lam).squeeze(),
                                                   x, n_steps_per_lam, 1, 5, step_size,
                                                   acceptEBM=lambda x: self.logp_target(x).squeeze())
            step_sizes[i] = step_size
            ars.append(ar.mean().item())
        return x, ars, step_sizes