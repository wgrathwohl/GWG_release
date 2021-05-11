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
from tqdm import tqdm
import pickle


def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def l1(module):
    loss = 0.
    for p in module.parameters():
        loss += p.abs().sum()
    return loss


def main(args):
    makedirs(args.save_dir)
    logger = open("{}/log.txt".format(args.save_dir), 'w')

    def my_print(s):
        print(s)
        logger.write(str(s) + '\n')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load existing data
    if args.data == "mnist" or args.data_file is not None:
        train_loader, test_loader, plot, viz = utils.get_data(args)
    # generate the dataset
    else:
        data, data_model = utils.generate_data(args)
        my_print("we have created your data, but what have you done for me lately?????")
        with open("{}/data.pkl".format(args.save_dir), 'wb') as f:
            pickle.dump(data, f)
        if args.data_model == "er_ising":
            ground_truth_J = data_model.J.detach().cpu()
            with open("{}/J.pkl".format(args.save_dir), 'wb') as f:
                pickle.dump(ground_truth_J, f)
        quit()


    if args.model == "lattice_potts":
        model = rbm.LatticePottsModel(int(args.dim), int(args.n_state), 0., 0., learn_sigma=True)
        buffer = model.init_sample(args.buffer_size)
    elif args.model == "lattice_ising":
        model = rbm.LatticeIsingModel(int(args.dim), 0., 0., learn_sigma=True)
        buffer = model.init_sample(args.buffer_size)
    elif args.model == "lattice_ising_3d":
        model = rbm.LatticeIsingModel(int(args.dim), .2, learn_G=True, lattice_dim=3)
        ground_truth_J = model.J.clone().to(device)
        model.G.data = torch.randn_like(model.G.data) * .01
        model.sigma.data = torch.ones_like(model.sigma.data)
        buffer = model.init_sample(args.buffer_size)
        plt.clf()
        plt.matshow(ground_truth_J.detach().cpu().numpy())
        plt.savefig("{}/ground_truth.png".format(args.save_dir))
    elif args.model == "lattice_ising_2d":
        model = rbm.LatticeIsingModel(int(args.dim), args.sigma, learn_G=True, lattice_dim=2)
        ground_truth_J = model.J.clone().to(device)
        model.G.data = torch.randn_like(model.G.data) * .01
        model.sigma.data = torch.ones_like(model.sigma.data)
        buffer = model.init_sample(args.buffer_size)
        plt.clf()
        plt.matshow(ground_truth_J.detach().cpu().numpy())
        plt.savefig("{}/ground_truth.png".format(args.save_dir))
    elif args.model == "er_ising":
        model = rbm.ERIsingModel(int(args.dim), 2, learn_G=True)
        model.G.data = torch.randn_like(model.G.data) * .01
        buffer = model.init_sample(args.buffer_size)
        with open(args.graph_file, 'rb') as f:
            ground_truth_J = pickle.load(f)
            plt.clf()
            plt.matshow(ground_truth_J.detach().cpu().numpy())
            plt.savefig("{}/ground_truth.png".format(args.save_dir))
        ground_truth_J = ground_truth_J.to(device)
    elif args.model == "rbm":
        model = rbm.BernoulliRBM(args.dim, args.n_hidden)
        buffer = model.init_dist.sample((args.buffer_size,))
    elif args.model == "dense_potts":
        raise ValueError
    elif args.model == "dense_ising":
        raise ValueError
    elif args.model == "mlp":
        raise ValueError

    model.to(device)
    buffer = buffer.to(device)

    # make G symmetric
    def get_J():
        j = model.J
        return (j + j.t()) / 2

    if args.sampler == "gibbs":
        if "potts" in args.model:
            sampler = samplers.PerDimMetropolisSampler(model.data_dim, int(args.n_state), rand=False)
        else:
            sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=False)
    elif args.sampler == "rand_gibbs":
        if "potts" in args.model:
            sampler = samplers.PerDimMetropolisSampler(model.data_dim, int(args.n_state), rand=True)
        else:
            sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=True)
    elif args.sampler == "gwg":
        if "potts" in args.model:
            sampler = samplers.DiffSamplerMultiDim(model.data_dim, 1, approx=True, temp=2.)
        else:
            sampler = samplers.DiffSampler(model.data_dim, 1, approx=True, fixed_proposal=False, temp=2.)
    else:
        assert "gwg-" in args.sampler
        n_hop = int(args.sampler.split('-')[1])
        if "potts" in args.model:
            raise ValueError
        else:
            sampler = samplers.MultiDiffSampler(model.data_dim, 1, approx=True, temp=2., n_samples=n_hop)

    my_print(device)
    my_print(model)
    my_print(buffer.size())
    my_print(sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    itr = 0
    sigmas = []
    sq_errs = []
    rmses = []
    while itr < args.n_iters:
        for x in train_loader:
            x = x[0].to(device)

            for k in range(args.sampling_steps):
                buffer = sampler.step(buffer.detach(), model).detach()

            logp_real = model(x).squeeze().mean()
            logp_fake = model(buffer).squeeze().mean()

            obj = logp_real - logp_fake
            loss = -obj
            loss += args.l1 * get_J().abs().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.G.data *= (1. - torch.eye(model.G.data.size(0))).to(model.G)

            if itr % args.print_every == 0:
                my_print("({}) log p(real) = {:.4f}, log p(fake) = {:.4f}, diff = {:.4f}, hops = {:.4f}".format(itr,
                                                                                              logp_real.item(),
                                                                                              logp_fake.item(),
                                                                                              obj.item(),
                                                                                              sampler._hops))
                if args.model in ("lattice_potts", "lattice_ising"):
                    my_print("\tsigma true = {:.4f}, current sigma = {:.4f}".format(args.sigma,
                                                                                    model.sigma.data.item()))
                else:
                    sq_err = ((ground_truth_J - get_J()) ** 2).sum()
                    rmse = ((ground_truth_J - get_J()) ** 2).mean().sqrt()
                    my_print("\t err^2 = {:.4f}, rmse = {:.4f}".format(sq_err, rmse))
                    print(ground_truth_J)
                    print(get_J())


            if itr % args.viz_every == 0:
                if args.model in ("lattice_potts", "lattice_ising"):
                    sigmas.append(model.sigma.data.item())
                    plt.clf()
                    plt.plot(sigmas, label="model")
                    plt.plot([args.sigma for s in sigmas], label="gt")
                    plt.legend()
                    plt.savefig("{}/sigma.png".format(args.save_dir))
                else:
                    sq_err = ((ground_truth_J - get_J()) ** 2).sum()
                    sq_errs.append(sq_err.item())
                    plt.clf()
                    plt.plot(sq_errs, label="sq_err")
                    plt.legend()
                    plt.savefig("{}/sq_err.png".format(args.save_dir))

                    rmse = ((ground_truth_J - get_J()) ** 2).mean().sqrt()
                    rmses.append(rmse.item())
                    plt.clf()
                    plt.plot(rmses, label="rmse")
                    plt.legend()
                    plt.savefig("{}/rmse.png".format(args.save_dir))

                    plt.clf()
                    plt.matshow(get_J().detach().cpu().numpy())
                    plt.savefig("{}/model_{}.png".format(args.save_dir, itr))

                plot("{}/data_{}.png".format(args.save_dir, itr), x.detach().cpu())
                plot("{}/buffer_{}.png".format(args.save_dir, itr), buffer[:args.batch_size].detach().cpu())

            itr += 1

            if itr > args.n_iters:
                if args.model in ("lattice_potts", "lattice_ising"):
                    final_sigma = model.sigma.data.item()
                    with open("{}/sigma.txt".format(args.save_dir), 'w') as f:
                        f.write(str(final_sigma))
                else:
                    sq_err = ((ground_truth_J - get_J()) ** 2).sum().item()
                    rmse = ((ground_truth_J - get_J()) ** 2).mean().sqrt().item()
                    with open("{}/sq_err.txt".format(args.save_dir), 'w') as f:
                        f.write(str(sq_err))
                    with open("{}/rmse.txt".format(args.save_dir), 'w') as f:
                        f.write(str(rmse))

                quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--save_dir', type=str, default="/tmp/test_discrete")
    parser.add_argument('--data', type=str, default='random')
    parser.add_argument('--data_file', type=str, help="location of pkl containing data")
    parser.add_argument('--graph_file', type=str, help="location of pkl containing graph") # ER
    # data generation
    parser.add_argument('--gt_steps', type=int, default=1000000)
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--sigma', type=float, default=.1)  # ising and potts
    parser.add_argument('--bias', type=float, default=0.)   # ising and potts
    parser.add_argument('--n_out', type=int, default=3)     # potts
    parser.add_argument('--degree', type=int, default=2)  # ER
    parser.add_argument('--data_model', choices=['rbm', 'lattice_ising', 'lattice_potts', 'lattice_ising_3d',
                                                 'er_ising'],
                        type=str, default='lattice_ising')
    # models
    parser.add_argument('--model', choices=['rbm', 'lattice_ising', 'lattice_potts', 'lattice_ising_3d',
                                            'lattice_ising_2d', 'er_ising'],
                        type=str, default='lattice_ising')
    # mcmc
    parser.add_argument('--sampler', type=str, default='gibbs')
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--approx', action="store_true")
    parser.add_argument('--sampling_steps', type=int, default=100)
    parser.add_argument('--buffer_size', type=int, default=100)

    #
    parser.add_argument('--n_iters', type=int, default=100000)

    parser.add_argument('--n_hidden', type=int, default=25)
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--n_state', type=int, default=3)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--viz_batch_size', type=int, default=1000)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--viz_every', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--weight_decay', type=float, default=.0)
    parser.add_argument('--l1', type=float, default=.0)

    args = parser.parse_args()
    args.device = device
    main(args)
