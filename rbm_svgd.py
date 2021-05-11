import argparse
import toy_data
import rbm
import torch
import numpy as np
import samplers_old as samplers
import mmd
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import torchvision
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import utils


def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


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

    gt_samples = model.gibbs_sample(n_steps=args.mcmc_steps, n_samples=args.n_samples + args.n_test_samples, plot=True)
    kmmd = mmd.MMD(mmd.exp_avg_hamming, False)
    gt_samples, gt_samples2 = gt_samples[:args.n_samples], gt_samples[args.n_samples:]
    if plot is not None:
        plot("{}/ground_truth.png".format(args.save_dir), gt_samples2)
    opt_stat = kmmd.compute_mmd(gt_samples2, gt_samples)
    print("gt <--> gt log-mmd", opt_stat, opt_stat.log10())

    new_samples = model.gibbs_sample(n_steps=0, n_samples=args.n_test_samples)

    log_mmds = {}
    log_mmds['gibbs'] = []
    for i in range(args.n_steps):
        if i % 10 == 0:
            stat = kmmd.compute_mmd(new_samples, gt_samples)
            log_stat = stat.log10().item()
            log_mmds['gibbs'].append(log_stat)
            print("gibbs", i, stat, stat.log10())
        new_samples = model.gibbs_sample(new_samples, 1)


    r_model = samplers.BinaryRelaxedModel(args.n_visible, model)
    r_model.to(device)


    if args.n_visible == 2:
        import visualize_flow
        def viz(p, t):
            plt.clf()
            visualize_flow.plt_flow_density(lambda x: r_model.logp_surrogate(x, t), plt.gca(), npts=200)
            plt.savefig(p)
        def plot(p, x):
            plt.clf()
            visualize_flow.plt_samples(x.detach().cpu().numpy(), plt.gca(), 200)
            plt.savefig(p)


    temps = [.5, 1., 2.]
    for temp in temps:
        log_mmds[temp] = []
        target = lambda x: r_model.logp_surrogate(x, temp)
        x = nn.Parameter(r_model.base_dist.sample((args.n_test_samples, args.n_visible)).to(device))
        optim = torch.optim.Adam(params=[x], lr=args.lr)
        svgd = samplers.SVGD(optim)
        if viz is not None:
            viz("{}/target_{}.png".format(args.save_dir, temp), temp)
        for i in range(args.n_steps):
            svgd.discrete_step(x, r_model.logp_target, target)

            if i % 100 == 0 and plot is not None:
                if args.data == "mnist":
                    hx = samplers.threshold(x)
                else:
                    hx = x
                plot("/{}/samples_temp_{}_{}.png".format(args.save_dir, temp, i), hx)

            if i % 10 == 0:
                hard_samples = samplers.threshold(x)
                stat = kmmd.compute_mmd(hard_samples, gt_samples)
                log_stat = stat.log10().item()
                log_mmds[temp].append(log_stat)
                print("temp = {}, itr = {}, log-mmd = {:.4f}, ess = {:.4f}".format(temp, i, log_stat, svgd._ess))

    plt.clf()
    for temp in temps + ['gibbs']:
        plt.plot(log_mmds[temp], label="{}".format(temp))

    plt.legend()
    plt.savefig("{}/results.png".format(args.save_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="/tmp/test_discrete")
    parser.add_argument('--data', choices=['mnist', 'random'], type=str, default='random')
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_test_samples', type=int, default=100)
    parser.add_argument('--mcmc_steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--adapt', action="store_true")
    parser.add_argument('--hmc', action="store_true")
    parser.add_argument('--mdim', action="store_true")
    parser.add_argument('--ss', type=float, default=.01)


    parser.add_argument('--n_anneal', type=int, default=10)
    parser.add_argument('--sgld_steps', type=int, default=100)
    parser.add_argument('--sgld_sigma', type=float, default=.01)
    parser.add_argument('--lam', type=float, default=1)
    parser.add_argument('--max_lam', type=float, default=1.)
    parser.add_argument('--n_hidden', type=int, default=25)
    parser.add_argument('--n_visible', type=int, default=100)
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
