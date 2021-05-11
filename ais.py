import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class AISModel(nn.Module):
    def __init__(self, model, init_dist):
        super().__init__()
        self.model = model
        self.init_dist = init_dist

    def forward(self, x, beta):
        logpx = self.model(x).squeeze()
        logpi = self.init_dist.log_prob(x).sum(-1)
        return logpx * beta + logpi * (1. - beta)


def evaluate(model, init_dist, sampler,
             train_loader, val_loader, test_loader,
             preprocess, device,
             n_iters, n_samples, steps_per_iter=1, viz_every=100):

    model = AISModel(model, init_dist)

    # move to cuda
    model.to(device)

    # annealing weights
    betas = np.linspace(0., 1., n_iters)

    samples = init_dist.sample((n_samples,))
    log_w = torch.zeros((n_samples,)).to(device)

    gen_samples = []
    for itr, beta_k in tqdm(enumerate(betas)):
        if itr == 0:
            continue  # skip 0

        beta_km1 = betas[itr - 1]

        # udpate importance weights
        with torch.no_grad():
            log_w = log_w + model(samples, beta_k) - model(samples, beta_km1)
        # update samples
        model_k = lambda x: model(x, beta=beta_k)
        for d in range(steps_per_iter):
            samples = sampler.step(samples.detach(), model_k).detach()

        if (itr + 1) % viz_every == 0:
            gen_samples.append(samples.cpu().detach())

    logZ_final = log_w.logsumexp(0) - np.log(n_samples)
    print("Final log(Z) = {:.4f}".format(logZ_final))

    model = model.model

    logps = []
    for x, _ in train_loader:
        x = preprocess(x.to(device))
        logp_x = model(x).squeeze().detach()
        logps.append(logp_x)

    logps = torch.cat(logps)
    train_ll = logps.mean() - logZ_final

    logps = []
    for x, _ in val_loader:
        x = preprocess(x.to(device))
        logp_x = model(x).squeeze().detach()
        logps.append(logp_x)

    logps = torch.cat(logps)
    val_ll = logps.mean() - logZ_final

    logps = []
    for x, _ in test_loader:
        x = preprocess(x.to(device))
        logp_x = model(x).squeeze().detach()
        logps.append(logp_x)

    logps = torch.cat(logps)
    test_ll = logps.mean() - logZ_final
    return logZ_final, train_ll, val_ll, test_ll, gen_samples