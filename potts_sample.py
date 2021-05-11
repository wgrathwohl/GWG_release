import argparse
import toy_data
import rbm
import torch
import numpy as np
import samplers
import mmd
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import torchvision
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import utils
import tensorflow_probability as tfp
import block_samplers
import time
import pickle


def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_ess(chain, burn_in):
    print(chain.shape)
    c = chain
    l = c.shape[0]
    bi = int(burn_in * l)
    c = c[bi:]
    cv = tfp.mcmc.effective_sample_size(c).numpy()
    print(cv)
    cv[np.isnan(cv)] = 1.
    return cv


def main(args):
    makedirs(args.save_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = rbm.LatticePottsModel(int(args.dim), int(args.n_out), args.sigma, args.bias)
    model.to(device)
    print(device)

    if args.n_out == 3:
        plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), args.dim, args.dim, 3).transpose(3, 1),
                                                         p, normalize=False, nrow=int(x.size(0) ** .5))
    else:
        plot = None

    ess_samples = model.init_sample(args.n_samples).to(device)

    hops = {}
    ess = {}
    times = {}
    chains = {}


    temps = ['dim-gibbs', 'rand-gibbs', 'gwg']
    for temp in temps:
        if temp == 'dim-gibbs':
            sampler = samplers.PerDimMetropolisSampler(args.dim ** 2, args.n_out)
        elif temp == "rand-gibbs":
            sampler = samplers.PerDimMetropolisSampler(args.dim ** 2, args.n_out, rand=True)
        elif "bg-" in temp:
            block_size = int(temp.split('-')[1])
            sampler = block_samplers.BlockGibbsSampler(model.data_dim, block_size)
        elif "hb-" in temp:
            block_size, hamming_dist = [int(v) for v in temp.split('-')[1:]]
            sampler = block_samplers.HammingBallSampler(model.data_dim, block_size, hamming_dist)
        elif temp == "gwg":
            sampler = samplers.DiffSamplerMultiDim(args.dim ** 2, 1, approx=True, temp=2.)
        elif "gwg-" in temp:
            n_hops = int(temp.split('-')[1])
            sampler = samplers.MultiDiffSampler(model.data_dim, 1,
                                                approx=True, temp=2., n_samples=n_hops)
        else:
            raise ValueError("Invalid sampler...")

        x = model.init_dist.sample((args.n_test_samples,)).to(device)

        times[temp] = []
        hops[temp] = []
        chain = []
        cur_time = 0.
        for i in range(args.n_steps):
            # do sampling and time it
            st = time.time()
            xhat = sampler.step(x.detach(), model).detach()
            cur_time += time.time() - st

            # compute hamming dist
            cur_hops = (x != xhat).float().view(x.size(0), -1).sum(-1).mean().item()

            # update trajectory
            x = xhat

            if i % args.subsample == 0:
                if args.ess_statistic == "dims":
                    chain.append(x.cpu()[0].view(-1).numpy()[None])
                else:
                    xc = x[0][None]
                    h = (xc != ess_samples).float().view(ess_samples.size(0), -1).sum(-1)
                    chain.append(h.detach().cpu().numpy()[None])

            if i % args.viz_every == 0 and plot is not None:
                plot("/{}/temp_{}_samples_{}.png".format(args.save_dir, temp, i), x)

            if i % args.print_every == 0:
                times[temp].append(cur_time)
                hops[temp].append(cur_hops)
                print("temp {}, itr = {}, hop-dist = {:.4f}".format(temp, i, cur_hops))

        chain = np.concatenate(chain, 0)
        chains[temp] = chain
        ess[temp] = get_ess(chain, args.burn_in)
        print("ess = {} +/- {}".format(ess[temp].mean(), ess[temp].std()))

    ess_temps = temps
    plt.clf()
    plt.boxplot([ess[temp] for temp in ess_temps], labels=ess_temps, showfliers=False)
    plt.savefig("{}/ess.png".format(args.save_dir))

    plt.clf()
    plt.boxplot([ess[temp] / times[temp][-1] / (1. - args.burn_in) for temp in ess_temps], labels=ess_temps, showfliers=False)
    plt.savefig("{}/ess_per_sec.png".format(args.save_dir))

    plt.clf()
    for temp in temps:
        plt.plot(hops[temp], label="{}".format(temp))

    plt.legend()
    plt.savefig("{}/hops.png".format(args.save_dir))

    for temp in temps:
        plt.clf()
        plt.plot(chains[temp][:, 0])
        plt.savefig("{}/trace_{}.png".format(args.save_dir, temp))

    with open("{}/results.pkl".format(args.save_dir), 'wb') as f:
        results = {
            'ess': ess,
            'hops': hops,
            'chains': chains
        }
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="/tmp/test_discrete")
    parser.add_argument('--data', choices=['mnist', 'random'], type=str, default='random')
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_test_samples', type=int, default=100)
    parser.add_argument('--gt_steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=8008135)
    # model def
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--n_out', type=int, default=3)
    parser.add_argument('--sigma', type=float, default=.1)
    parser.add_argument('--bias', type=float, default=0.)
    # logging
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--viz_every', type=int, default=100)
    # for rbm training
    parser.add_argument('--rbm_lr', type=float, default=.001)
    parser.add_argument('--cd', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--batch_size', type=int, default=100)
    # for ess
    parser.add_argument('--subsample', type=int, default=1)
    parser.add_argument('--burn_in', type=float, default=.1)
    parser.add_argument('--ess_statistic', type=str, default="dims", choices=["hamming", "dims"])
    args = parser.parse_args()

    main(args)
