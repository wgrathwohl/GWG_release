import argparse
import rbm
import torch
import numpy as np
import samplers
import matplotlib.pyplot as plt
import os
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def main(args):
    makedirs(args.save_dir)
    logger = open("{}/log.txt".format(args.save_dir), 'w')

    def my_print(s):
        print(s)
        logger.write(str(s) + '\n')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load existing data
    if args.model == "lattice_potts":
        model = rbm.LatticePottsModel(int(args.dim), int(args.n_out), 0., 0., learn_sigma=True)
    if args.model == "dense_potts":
        model = rbm.DensePottsModel(args.dim, args.n_out, learn_J=True, learn_bias=True)
    else:
        raise ValueError

    model.to(device)

    if args.sampler == "gibbs":
        sampler = samplers.PerDimMetropolisSampler(args.dim, int(args.n_out), rand=False)
    elif args.sampler == "rand_gibbs":
        sampler = samplers.PerDimMetropolisSampler(args.dim, int(args.n_out), rand=True)
    elif args.sampler == "gwg":
        sampler = samplers.DiffSamplerMultiDim(args.dim, 1, approx=True, temp=2.)
    else:
        raise ValueError

    my_print(device)
    my_print(model)
    my_print(sampler)

    # load ckpt
    my_print("Loading...")
    if args.ckpt_path is not None:
        d = torch.load(args.ckpt_path)
        model.load_state_dict(d['model'])
    my_print("Loaded!")
    betas = np.linspace(0., 1., args.n_iters)

    samples = model.init_sample(args.n_samples)
    log_w = torch.zeros((args.n_samples,)).to(device)
    log_w += model.bias.logsumexp(-1).sum()

    logZs = []
    for itr, beta_k in enumerate(betas):
        if itr == 0:
            continue  # skip 0

        beta_km1 = betas[itr - 1]

        # udpate importance weights
        with torch.no_grad():
            log_w = log_w + model(samples, beta=beta_k) - model(samples, beta_km1)
        # update samples
        model_k = lambda x: model(x, beta=beta_k)
        for d in range(args.steps_per_iter):
            samples = sampler.step(samples.detach(), model_k).detach()


        if itr % args.print_every == 0:
            logZ = log_w.logsumexp(0) - np.log(args.n_samples)
            logZs.append(logZ.item())
            my_print("({}) beta = {}, log Z = {:.4f}".format(itr, beta_k, logZ.item()))
            logger.flush()


        if itr % args.viz_every == 0:
            plt.clf()
            plt.plot(logZs, label="log(Z)")
            plt.legend()
            plt.savefig("{}/logZ.png".format(args.save_dir))

    logZ_final = log_w.logsumexp(0) - np.log(args.n_samples)
    my_print("Final log(Z) = {:.4f}".format(logZ_final))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--save_dir', type=str, default="/tmp/test_discrete")
    parser.add_argument('--ckpt_path', type=str, default=None)
    # data generation
    parser.add_argument('--n_samples', type=int, default=1000)
    # models
    parser.add_argument('--model', choices=['lattice_potts', 'dense_potts'],
                        type=str, default='dense_potts')
    # mcmc
    parser.add_argument('--sampler', type=str, default='gibbs')
    parser.add_argument('--seed', type=int, default=1234567)

    #
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--dim', type=int, default=48)
    parser.add_argument('--steps_per_iter', type=int, default=48)
    parser.add_argument('--n_out', type=int, default=21)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--viz_every', type=int, default=1000)


    args = parser.parse_args()
    args.device = device
    main(args)