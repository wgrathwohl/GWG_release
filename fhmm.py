import torch
import torch.nn as nn


class FHMM(nn.Module):
    def __init__(self, N, K, W, W0, out_sigma, p, v,
                 learn_W=False, learn_W0=False, learn_p=False, learn_v=False, learn_obs=False, alt_logpx=False):
        super().__init__()
        self.logit_v = nn.Parameter(v.log() - (1. - v).log(), requires_grad=learn_v)
        self.logit_p = nn.Parameter(p.log() - (1. - p).log(), requires_grad=learn_p)
        self.W = nn.Parameter(W, requires_grad=learn_W)
        self.W0 = nn.Parameter(W0, requires_grad=learn_W0)
        self.out_logsigma = nn.Parameter(torch.tensor(out_sigma).log(), requires_grad=learn_obs)
        self.K = K
        self.N = N
        self.alt_logpx = alt_logpx

    @property
    def out_sigma(self):
        return self.out_logsigma.exp()

    def p_X0(self):
        return torch.distributions.Bernoulli(logits=self.logit_v)

    def p_XC(self):
        return torch.distributions.Bernoulli(logits=-self.logit_p)

    def log_p_X(self, X):
        X = X.view(X.size(0), self.N, self.K)
        X0 = X[:, 0]
        X_cur = X[:, :-1]
        X_next = X[:, 1:]
        X_change = (1. - X_cur) * X_next + (1. - X_next) * X_cur
        log_px0 = self.p_X0().log_prob(X0).sum(-1)
        log_pxC = self.p_XC().log_prob(X_change).sum(-1).sum(-1)
        return log_px0 + log_pxC

    def log_p_X2(self, X):
        X = X.view(X.size(0), self.N, self.K)
        X = 2 * X - 1
        X0 = X[:, 0]
        log_px0 = (X0 * self.logit_v / 2).sum(-1)
        X_cur = X[:, :-1]
        X_next = X[:, 1:]
        X_change = X_cur * X_next
        log_pxc = (X_change * self.logit_p / 2).sum(-1).sum(-1)
        return log_px0 + log_pxc

    def p_y_given_x(self, X, sigma=None):
        X = X.view(X.size(0), self.N, self.K)
        xw = (self.W[None, None] * X).sum(-1)
        mu = xw + self.W0
        if sigma is None:
            sigma = self.out_sigma
        out_dist = torch.distributions.Normal(mu, sigma)
        return out_dist

    def log_p_y_given_x(self, y, X, sigma=None):
        out_dist = self.p_y_given_x(X, sigma=sigma)
        if len(y.size()) == 1:
            return out_dist.log_prob(y[None]).sum(-1)
        else:
            return out_dist.log_prob(y).sum(-1)

    def log_p_joint(self, y, X, sigma=None):
        logp_y = self.log_p_y_given_x(y, X, sigma=sigma)
        if self.alt_logpx:
            logp_X = self.log_p_X2(X)
        else:
            logp_X = self.log_p_X(X)
        return logp_y + logp_X

    def sample_X(self, n=1):
        X0 = self.p_X0().sample((n,))
        XNs = [X0[:, None, :]]
        for i in range(self.N - 1):
            XC = self.p_XC().sample((n,))[:, None, :]
            X_cur = XNs[-1]
            X_next = (1. - XC) * X_cur + XC * (1. - X_cur)
            XNs.append(X_next)
        return torch.cat(XNs, 1)








