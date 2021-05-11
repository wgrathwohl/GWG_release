import argparse
import rbm
import torch
import numpy as np
import samplers
import mmd
import matplotlib.pyplot as plt
import os
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
    c = chain
    l = c.shape[0]
    bi = int(burn_in * l)
    c = c[bi:]
    cv = tfp.mcmc.effective_sample_size(c).numpy()
    cv[np.isnan(cv)] = 1.
    return cv


def main(args):
    makedirs(args.save_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = rbm.BernoulliRBM(args.n_visible, args.n_hidden)
    model.to(device)
    print(device)

    if args.data == "mnist":
        assert args.n_visible == 784
        train_loader, test_loader, plot, viz = utils.get_data(args)

        init_data = []
        for x, _ in train_loader:
            init_data.append(x)
        init_data = torch.cat(init_data, 0)
        init_mean = init_data.mean(0).clamp(.01, .99)

        model = rbm.BernoulliRBM(args.n_visible, args.n_hidden, data_mean=init_mean)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.rbm_lr)

        # train!
        itr = 0
        for x, _ in train_loader:
            x = x.to(device)
            xhat = model.gibbs_sample(v=x, n_steps=args.cd)

            d = model.logp_v_unnorm(x)
            m = model.logp_v_unnorm(xhat)

            obj = d - m
            loss = -obj.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if itr % args.print_every == 0:
                print("{} | log p(data) = {:.4f}, log p(model) = {:.4f}, diff = {:.4f}".format(itr,d.mean(), m.mean(),
                                                                                               (d - m).mean()))

    else:
        model.W.data = torch.randn_like(model.W.data) * (.05 ** .5)
        model.b_v.data = torch.randn_like(model.b_v.data) * 1.0
        model.b_h.data = torch.randn_like(model.b_h.data) * 1.0
        viz = plot = None

    gt_samples = model.gibbs_sample(n_steps=args.gt_steps, n_samples=args.n_samples + args.n_test_samples, plot=True)
    kmmd = mmd.MMD(mmd.exp_avg_hamming, False)
    gt_samples, gt_samples2 = gt_samples[:args.n_samples], gt_samples[args.n_samples:]
    if plot is not None:
        plot("{}/ground_truth.png".format(args.save_dir), gt_samples2)
    opt_stat = kmmd.compute_mmd(gt_samples2, gt_samples)
    print("gt <--> gt log-mmd", opt_stat, opt_stat.log10())

    new_samples = model.gibbs_sample(n_steps=0, n_samples=args.n_test_samples)

    log_mmds = {}
    log_mmds['gibbs'] = []
    ars = {}
    hops = {}
    ess = {}
    times = {}
    chains = {}
    chain = []

    times['gibbs'] = []
    start_time = time.time()
    for i in range(args.n_steps):
        if i % args.print_every == 0:
            stat = kmmd.compute_mmd(new_samples, gt_samples)
            log_stat = stat.log10().item()
            log_mmds['gibbs'].append(log_stat)
            print("gibbs", i, stat, stat.log10())
            times['gibbs'].append(time.time() - start_time)
        new_samples = model.gibbs_sample(new_samples, 1)
        if i % args.subsample == 0:
            if args.ess_statistic == "dims":
                chain.append(new_samples.cpu().numpy()[0][None])
            else:
                xc = new_samples[0][None]
                h = (xc != gt_samples).float().sum(-1)
                chain.append(h.detach().cpu().numpy()[None])

    chain = np.concatenate(chain, 0)
    chains['gibbs'] = chain
    ess['gibbs'] = get_ess(chain, args.burn_in)
    print("ess = {} +/- {}".format(ess['gibbs'].mean(), ess['gibbs'].std()))

    temps = ['bg-1', 'bg-2', 'hb-10-1', 'gwg', 'gwg-3', 'gwg-5']
    for temp in temps:
        if temp == 'dim-gibbs':
            sampler = samplers.PerDimGibbsSampler(args.n_visible)
        elif temp == "rand-gibbs":
            sampler = samplers.PerDimGibbsSampler(args.n_visible, rand=True)
        elif "bg-" in temp:
            block_size = int(temp.split('-')[1])
            sampler = block_samplers.BlockGibbsSampler(args.n_visible, block_size)
        elif "hb-" in temp:
            block_size, hamming_dist = [int(v) for v in temp.split('-')[1:]]
            sampler = block_samplers.HammingBallSampler(args.n_visible, block_size, hamming_dist)
        elif temp == "gwg":
            sampler = samplers.DiffSampler(args.n_visible, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
        elif "gwg-" in temp:
            n_hops = int(temp.split('-')[1])
            sampler = samplers.MultiDiffSampler(args.n_visible, 1,
                                                approx=True, temp=2., n_samples=n_hops)
        else:
            raise ValueError("Invalid sampler...")

        x = model.init_dist.sample((args.n_test_samples,)).to(device)

        log_mmds[temp] = []
        ars[temp] = []
        hops[temp] = []
        times[temp] = []
        chain = []
        cur_time = 0.
        for i in range(args.n_steps):
            # do sampling and time it
            st = time.time()
            xhat = sampler.step(x.detach(), model).detach()
            cur_time += time.time() - st

            # compute hamming dist
            cur_hops = (x != xhat).float().sum(-1).mean().item()

            # update trajectory
            x = xhat

            if i % args.subsample == 0:
                if args.ess_statistic == "dims":
                    chain.append(x.cpu().numpy()[0][None])
                else:
                    xc = x[0][None]
                    h = (xc != gt_samples).float().sum(-1)
                    chain.append(h.detach().cpu().numpy()[None])

            if i % args.viz_every == 0 and plot is not None:
                plot("/{}/temp_{}_samples_{}.png".format(args.save_dir, temp, i), x)

            if i % args.print_every == 0:
                hard_samples = x
                stat = kmmd.compute_mmd(hard_samples, gt_samples)
                log_stat = stat.log10().item()
                log_mmds[temp].append(log_stat)
                times[temp].append(cur_time)
                hops[temp].append(cur_hops)
                print("temp {}, itr = {}, log-mmd = {:.4f}, hop-dist = {:.4f}".format(temp, i, log_stat, cur_hops))
        chain = np.concatenate(chain, 0)
        ess[temp] = get_ess(chain, args.burn_in)
        chains[temp] = chain
        print("ess = {} +/- {}".format(ess[temp].mean(), ess[temp].std()))

    ess_temps = temps
    plt.clf()
    plt.boxplot([ess[temp] for temp in ess_temps], labels=ess_temps, showfliers=False)
    plt.savefig("{}/ess.png".format(args.save_dir))

    plt.clf()
    plt.boxplot([ess[temp] / times[temp][-1] / (1. - args.burn_in) for temp in ess_temps], labels=ess_temps, showfliers=False)
    plt.savefig("{}/ess_per_sec.png".format(args.save_dir))

    plt.clf()
    for temp in temps + ['gibbs']:
        plt.plot(log_mmds[temp], label="{}".format(temp))

    plt.legend()
    plt.savefig("{}/results.png".format(args.save_dir))

    plt.clf()
    for temp in temps:
        plt.plot(ars[temp], label="{}".format(temp))

    plt.legend()
    plt.savefig("{}/ars.png".format(args.save_dir))

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
            'log_mmds': log_mmds,
            'chains': chains,
            'times': times
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
    parser.add_argument('--seed', type=int, default=1234567)
    # rbm def
    parser.add_argument('--n_hidden', type=int, default=25)
    parser.add_argument('--n_visible', type=int, default=100)
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
