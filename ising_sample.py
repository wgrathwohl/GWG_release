import argparse
import rbm
import torch
import numpy as np
import samplers
import matplotlib.pyplot as plt
import os
import torchvision
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
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
    c = chain
    l = c.shape[0]
    bi = int(burn_in * l)
    c = c[bi:]
    cv = tfp.mcmc.effective_sample_size(c).numpy()
    cv[np.isnan(cv)] = 1.
    return cv

def get_log_rmse(x):
    x = 2. * x - 1.
    x2 = (x ** 2).mean(-1)
    return x2.log10().detach().cpu().numpy()


def main(args):
    makedirs(args.save_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = rbm.LatticeIsingModel(args.dim, args.sigma, args.bias)
    model.to(device)
    print(device)


    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, args.dim, args.dim),
                                                     p, normalize=False, nrow=int(x.size(0) ** .5))
    ess_samples = model.init_sample(args.n_samples).to(device)

    hops = {}
    ess = {}
    times = {}
    chains = {}
    means = {}

    temps = ['bg-1', 'bg-2', 'hb-10-1', 'gwg', 'gwg-3', 'gwg-5']
    for temp in temps:
        if temp == 'dim-gibbs':
            sampler = samplers.PerDimGibbsSampler(model.data_dim)
        elif temp == "rand-gibbs":
            sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=True)
        elif "bg-" in temp:
            block_size = int(temp.split('-')[1])
            sampler = block_samplers.BlockGibbsSampler(model.data_dim, block_size)
        elif "hb-" in temp:
            block_size, hamming_dist = [int(v) for v in temp.split('-')[1:]]
            sampler = block_samplers.HammingBallSampler(model.data_dim, block_size, hamming_dist)
        elif temp == "gwg":
            sampler = samplers.DiffSampler(model.data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
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
        mean = torch.zeros_like(x)
        for i in range(args.n_steps):
            # do sampling and time it
            st = time.time()
            xhat = sampler.step(x.detach(), model).detach()
            cur_time += time.time() - st

            # compute hamming dist
            cur_hops = (x != xhat).float().sum(-1).mean().item()

            # update trajectory
            x = xhat

            mean = mean + x
            if i % args.subsample == 0:
                if args.ess_statistic == "dims":
                    chain.append(x.cpu().numpy()[0][None])
                else:
                    xc = x#[0][None]
                    h = (xc != ess_samples[0][None]).float().sum(-1)
                    chain.append(h.detach().cpu().numpy()[None])

            if i % args.viz_every == 0 and plot is not None:
                plot("/{}/temp_{}_samples_{}.png".format(args.save_dir, temp, i), x)

            if i % args.print_every == 0:
                times[temp].append(cur_time)
                hops[temp].append(cur_hops)
                print("temp {}, itr = {}, hop-dist = {:.4f}".format(temp, i, cur_hops))

        means[temp] = mean / args.n_steps
        chain = np.concatenate(chain, 0)
        chains[temp] = chain
        if not args.no_ess:
            ess[temp] = get_ess(chain, args.burn_in)
            print("ess = {} +/- {}".format(ess[temp].mean(), ess[temp].std()))

    ess_temps = temps
    plt.clf()
    plt.boxplot([get_log_rmse(means[temp]) for temp in ess_temps], labels=ess_temps, showfliers=False)
    plt.savefig("{}/log_rmse.png".format(args.save_dir))

    if not args.no_ess:
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
            'chains': chains,
            'means': means
        }
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="/tmp/test_discrete")
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_test_samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234567)
    # model def
    parser.add_argument('--dim', type=int, default=10)
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
    parser.add_argument('--no_ess', action="store_true")
    args = parser.parse_args()

    main(args)
