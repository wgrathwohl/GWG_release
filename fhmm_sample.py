import argparse
import torch
import numpy as np
import samplers
import fhmm
import matplotlib.pyplot as plt
import os
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import time
import block_samplers
import pickle


def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def main(args):
    makedirs("{}/sources".format(args.save_dir))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    W = args.W_init_sigma * torch.randn((args.K,))
    W0 = args.W_init_sigma * torch.randn((1,))
    p = args.X_keep_prob * torch.ones((args.K,))
    v = args.X0_mean * torch.ones((args.K,))

    model = fhmm.FHMM(args.N, args.K, W, W0, args.obs_sigma, p, v, alt_logpx=args.alt)
    model.to(device)
    print("device is", device)

    # generate data
    Xgt = model.sample_X(1)
    p_y_given_Xgt = model.p_y_given_x(Xgt)

    mu = p_y_given_Xgt.loc
    mu_true = mu[0]
    plt.clf()
    plt.plot(mu_true.detach().cpu().numpy(), label="mean")
    ygt = p_y_given_Xgt.sample()[0]
    plt.plot(ygt.detach().cpu().numpy(), label='sample')
    plt.legend()
    plt.savefig("{}/data.png".format(args.save_dir))
    ygt = ygt.to(device)

    for k in range(args.K):
        plt.clf()
        plt.plot(Xgt[0, :, k].detach().cpu().numpy())
        plt.savefig("{}/sources/x_{}.png".format(args.save_dir, k))


    logp_joint_real = model.log_p_joint(ygt, Xgt).item()
    print("joint likelihood of real data is {}".format(logp_joint_real))

    log_joints = {}
    diffs = {}
    times = {}
    recons = {}
    ars = {}
    hops = {}
    phops = {}
    mus = {}

    dim = args.K * args.N
    x_init = model.sample_X(args.n_test_samples).to(device)
    samp_model = lambda _x: model.log_p_joint(ygt, _x)

    temps = ['bg-1', 'bg-2', 'hb-10-1', 'gwg', 'gwg-3', 'gwg-5']
    for temp in temps:
        makedirs("{}/{}".format(args.save_dir, temp))
        if temp == 'dim-gibbs':
            sampler = samplers.PerDimGibbsSampler(dim)
        elif temp == "rand-gibbs":
            sampler = samplers.PerDimGibbsSampler(dim, rand=True)
        elif "bg-" in temp:
            block_size = int(temp.split('-')[1])
            sampler = block_samplers.BlockGibbsSampler(dim, block_size)
        elif "hb-" in temp:
            block_size, hamming_dist = [int(v) for v in temp.split('-')[1:]]
            sampler = block_samplers.HammingBallSampler(dim, block_size, hamming_dist)
        elif temp == "gwg":
            sampler = samplers.DiffSampler(dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
        elif "gwg-" in temp:
            n_hops = int(temp.split('-')[1])
            sampler = samplers.MultiDiffSampler(dim, 1,
                                                approx=True, temp=2., n_samples=n_hops)
        else:
            raise ValueError("Invalid sampler...")
        
        x = x_init.clone().view(x_init.size(0), -1)

        diffs[temp] = []

        log_joints[temp] = []
        ars[temp] = []
        hops[temp] = []
        phops[temp] = []
        recons[temp] = []
        start_time = time.time()
        for i in range(args.n_steps + 1):
            if args.anneal is None:
                sm = samp_model
            else:
                s = np.linspace(args.anneal, args.obs_sigma, args.n_steps + 1)[i]
                sm = lambda _x: model.log_p_joint(ygt, _x, sigma=s)
            xhat = sampler.step(x.detach(), sm).detach()

            # compute hamming dist
            cur_hops = (x != xhat).float().sum(-1).mean().item()
            # update trajectory
            x = xhat

            if i % 1000 == 0:
                p_y_given_x = model.p_y_given_x(x)
                mu = p_y_given_x.loc
                plt.clf()
                plt.plot(mu_true.detach().cpu().numpy(), label="true")
                plt.plot(mu[0].detach().cpu().numpy() + .01, label='mu0')
                plt.plot(mu[1].detach().cpu().numpy() - .01, label='mu1')
                plt.legend()
                plt.savefig("{}/{}/mean_{}.png".format(args.save_dir, temp, i))
                mus[temp] = mu[0].detach().cpu().numpy()

            if i % 10 == 0:
                p_y_given_x = model.p_y_given_x(x)
                mu = p_y_given_x.loc
                err = ((mu - ygt[None]) ** 2).sum(1).mean()
                recons[temp].append(err.item())

                log_j = model.log_p_joint(ygt, x)
                diff = (x.view(x.size(0), args.N, args.K) != Xgt).float().view(x.size(0), -1).mean(1)
                log_joints[temp].append(log_j.mean().item())
                diffs[temp].append(diff.mean().item())
                hops[temp].append(cur_hops)
                print("temp {}, itr = {}, log-joint = {:.4f}, "
                      "hop-dist = {:.4f}, recons = {:.4f}".format(temp, i, log_j.mean().item(), cur_hops, err.item()))

        for k in range(args.K):
            plt.clf()
            xr = x.view(x.size(0), args.N, args.K)
            plt.plot(xr[0, :, k].detach().cpu().numpy())
            plt.savefig("{}/{}/source_{}.png".format(args.save_dir, temp, k))

        times[temp] = time.time() - start_time


    plt.clf()
    for temp in temps:
        plt.plot(log_joints[temp], label=temp)
    plt.plot([logp_joint_real for _ in log_joints[temp]], label="true")
    plt.legend()
    plt.savefig("{}/joints.png".format(args.save_dir))

    plt.clf()
    for temp in temps:
        plt.plot(recons[temp], label=temp)
    plt.legend()
    plt.savefig("{}/recons.png".format(args.save_dir))

    plt.clf()
    for temp in temps:
        plt.plot(diffs[temp], label=temp)
    plt.legend()
    plt.savefig("{}/errs.png".format(args.save_dir))

    plt.clf()
    for i, temp in enumerate(temps):
        plt.plot(mus[temp] + float(i) * .01, label=temp)
    plt.plot(mu_true.detach().cpu().numpy(), label="true")
    plt.legend()
    plt.savefig("{}/mean.png".format(args.save_dir))

    plt.clf()
    for temp in temps:
        plt.plot(hops[temp], label="{}".format(temp))

    plt.legend()
    plt.savefig("{}/hops.png".format(args.save_dir))

    with open("{}/results.pkl".format(args.save_dir), 'wb') as f:
        results = {
            'hops': hops,
            'recons': recons,
            'joints': log_joints,
        }
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="/tmp/test_discrete")
    parser.add_argument('--data', choices=['random'], type=str, default='random')
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_test_samples', type=int, default=100)
    parser.add_argument('--n_multi_sample', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--multi_sample', action="store_true")
    parser.add_argument('--approx', action="store_true")
    parser.add_argument('--alt', action="store_true")
    parser.add_argument('--anneal', type=float, default=None)


    parser.add_argument('--W_init_sigma', type=float, default=1.)
    parser.add_argument('--obs_sigma', type=float, default=.5)
    parser.add_argument('--X0_mean', type=float, default=.1)
    parser.add_argument('--X_keep_prob', type=float, default=.95)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--N', type=int, default=100)


    parser.add_argument('--sigma', type=float, default=.1)
    parser.add_argument('--bias', type=float, default=0.)
    parser.add_argument('--n_hidden', type=int, default=25)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--viz_batch_size', type=int, default=1000)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--viz_every', type=int, default=1000)
    parser.add_argument('--n_toy_data', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--rbm_lr', type=float, default=.001)
    parser.add_argument('--mcmc_lr', type=float, default=.003)
    parser.add_argument('--temp', type=float, default=1.)
    parser.add_argument('--tt', type=float, default=1.)
    parser.add_argument('--weight_decay', type=float, default=.0)
    parser.add_argument('--cd', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=28)

    args = parser.parse_args()

    main(args)

