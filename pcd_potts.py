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
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import utils


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


def norm_J(J):
    return J.norm(dim=(2, 3))


def matsave(M, path):
    plt.clf()
    plt.matshow(M.detach().cpu().numpy())
    plt.savefig(path)


def main(args):
    makedirs(args.save_dir)
    logger = open("{}/log.txt".format(args.save_dir), 'w')

    def my_print(s):
        print(s)
        logger.write(str(s) + '\n')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load existing data
    if args.data == "synthetic":
        train_loader, test_loader, data, ground_truth_J, ground_truth_h, ground_truth_C = utils.load_synthetic(
            args.data_file, args.batch_size)
        dim, n_out = data.size()[1:]
        ground_truth_J_norm = norm_J(ground_truth_J).to(device)
        matsave(ground_truth_J.abs().transpose(2, 1).reshape(dim * n_out, dim * n_out),
                "{}/ground_truth_J.png".format(args.save_dir))
        matsave(ground_truth_C, "{}/ground_truth_C.png".format(args.save_dir))
        matsave(ground_truth_J_norm, "{}/ground_truth_J_norm.png".format(args.save_dir))
        num_ecs = 120
        dm_indices = torch.arange(ground_truth_J_norm.size(0)).long()
    # generate the dataset
    elif args.data == "PF00018":
        train_loader, test_loader, data, num_ecs, ground_truth_J_norm, ground_truth_C = utils.load_ingraham(args)
        dim, n_out = data.size()[1:]
        ground_truth_J_norm = ground_truth_J_norm.to(device)
        matsave(ground_truth_C, "{}/ground_truth_C.png".format(args.save_dir))
        matsave(ground_truth_J_norm, "{}/ground_truth_dists.png".format(args.save_dir))
        dm_indices = torch.arange(ground_truth_J_norm.size(0)).long()

    else:
        train_loader, test_loader, data, num_ecs, ground_truth_J_norm, ground_truth_C, dm_indices = utils.load_real_protein(args)
        dim, n_out = data.size()[1:]
        ground_truth_J_norm = ground_truth_J_norm.to(device)
        matsave(ground_truth_C, "{}/ground_truth_C.png".format(args.save_dir))
        matsave(ground_truth_J_norm, "{}/ground_truth_dists.png".format(args.save_dir))

    if args.model == "lattice_potts":
        model = rbm.LatticePottsModel(int(args.dim), int(n_out), 0., 0., learn_sigma=True)
        buffer = model.init_sample(args.buffer_size)
    if args.model == "dense_potts":
        model = rbm.DensePottsModel(dim, n_out, learn_J=True, learn_bias=True)
        buffer = model.init_sample(args.buffer_size)
    elif args.model == "dense_ising":
        raise ValueError
    elif args.model == "mlp":
        raise ValueError

    model.to(device)

    # make G symmetric
    def get_J():
        j = model.J
        jt = j.transpose(0, 1).transpose(2, 3)
        return (j + jt) / 2

    def get_J_sub():
        j = get_J()
        j_sub = j[dm_indices, :][:, dm_indices]
        return j_sub

    if args.sampler == "gibbs":
        if "potts" in args.model:
            sampler = samplers.PerDimMetropolisSampler(dim, int(n_out), rand=False)
        else:
            sampler = samplers.PerDimGibbsSampler(dim, rand=False)
    elif args.sampler == "plm":
        sampler = samplers.PerDimMetropolisSampler(dim, int(n_out), rand=False)
    elif args.sampler == "rand_gibbs":
        if "potts" in args.model:
            sampler = samplers.PerDimMetropolisSampler(dim, int(n_out), rand=True)
        else:
            sampler = samplers.PerDimGibbsSampler(dim, rand=True)
    elif args.sampler == "gwg":
        if "potts" in args.model:
            sampler = samplers.DiffSamplerMultiDim(dim, 1, approx=True, temp=2.)
        else:
            sampler = samplers.DiffSampler(dim, 1, approx=True, fixed_proposal=False, temp=2.)
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

    # load ckpt
    if args.ckpt_path is not None:
        d = torch.load(args.ckpt_path)
        model.load_state_dict(d['model'])
        optimizer.load_state_dict(d['optimizer'])
        sampler.load_state_dict(d['sampler'])


    # mask matrix for PLM
    L, D = model.J.size(0), model.J.size(2)
    num_node = L * D
    J_mask = torch.ones((num_node, num_node)).to(device)
    for i in range(L):
        J_mask[D * i:D * i + D, D * i:D * i + D] = 0


    itr = 0
    sq_errs = []
    rmses = []
    all_inds = list(range(args.buffer_size))
    while itr < args.n_iters:
        for x in train_loader:
            if args.data == "synthetic":
                x = x[0].to(device)
                weights = torch.ones((x.size(0),)).to(device)
            else:
                weights = x[1].to(device)
                if args.unweighted:
                    weights = torch.ones_like(weights)
                x = x[0].to(device)

            if args.sampler == "plm":
                plm_J = model.J.transpose(2, 1).reshape(dim * n_out, dim * n_out)
                logits = torch.matmul(x.view(x.size(0), -1), plm_J * J_mask) + model.bias.view(-1)[None]
                x_inds = (torch.arange(x.size(-1))[None, None].to(x.device) * x).sum(-1)
                cross_entropy = nn.functional.cross_entropy(
                    input=logits.reshape((-1, D)),
                    target=x_inds.view(-1).long(),
                    reduce=False)
                cross_entropy = torch.sum(cross_entropy.reshape((-1, L)), -1)
                loss = (cross_entropy * weights).mean()

            else:
                buffer_inds = np.random.choice(all_inds, args.batch_size, replace=False)
                x_fake = buffer[buffer_inds].to(device)
                for k in range(args.sampling_steps):
                    x_fake = sampler.step(x_fake.detach(), model).detach()

                buffer[buffer_inds] = x_fake.detach().cpu()

                logp_real = (model(x).squeeze() * weights).mean()
                logp_fake = model(x_fake).squeeze().mean()

                obj = logp_real - logp_fake
                loss = -obj

            # add l1 reg
            loss += args.l1 * norm_J(get_J()).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if itr % args.print_every == 0:
                if args.sampler == "plm":
                    my_print("({}) loss = {:.4f}".format(itr, loss.item()))
                else:
                    my_print("({}) log p(real) = {:.4f}, log p(fake) = {:.4f}, diff = {:.4f}, hops = {:.4f}".format(itr,
                                                                                                  logp_real.item(),
                                                                                                  logp_fake.item(),
                                                                                                  obj.item(),
                                                                                                  sampler._hops))

                sq_err = ((ground_truth_J_norm - norm_J(get_J_sub())) ** 2).sum()
                rmse = ((ground_truth_J_norm - norm_J(get_J_sub())) ** 2).mean().sqrt()
                inds = torch.triu_indices(ground_truth_C.size(0), ground_truth_C.size(1), 1)
                C_inds = ground_truth_C[inds[0], inds[1]]
                J_inds = norm_J(get_J_sub())[inds[0], inds[1]]
                J_inds_sorted = torch.sort(J_inds, descending=True).indices
                C_inds_sorted = C_inds[J_inds_sorted]
                C_cumsum = C_inds_sorted.cumsum(0)
                arange = torch.arange(C_cumsum.size(0)) + 1
                acc_at = C_cumsum.float() / arange.float()
                my_print("\t err^2 = {:.4f}, rmse = {:.4f}, acc @ 50 = {:.4f}, acc @ 75 = {:.4f}, acc @ 100 = {:.4f}".format(sq_err, rmse,
                                                                                                         acc_at[50],
                                                                                                         acc_at[75],
                                                                                                         acc_at[100]))
                logger.flush()


            if itr % args.viz_every == 0:
                sq_err = ((ground_truth_J_norm - norm_J(get_J_sub())) ** 2).sum()
                rmse = ((ground_truth_J_norm - norm_J(get_J_sub())) ** 2).mean().sqrt()

                sq_errs.append(sq_err.item())
                plt.clf()
                plt.plot(sq_errs, label="sq_err")
                plt.legend()
                plt.savefig("{}/sq_err.png".format(args.save_dir))

                rmses.append(rmse.item())
                plt.clf()
                plt.plot(rmses, label="rmse")
                plt.legend()
                plt.savefig("{}/rmse.png".format(args.save_dir))


                matsave(get_J_sub().abs().transpose(2, 1).reshape(dm_indices.size(0) * n_out,
                                                                  dm_indices.size(0) * n_out),
                        "{}/model_J_{}_sub.png".format(args.save_dir, itr))
                matsave(norm_J(get_J_sub()), "{}/model_J_norm_{}_sub.png".format(args.save_dir, itr))

                matsave(get_J().abs().transpose(2, 1).reshape(dim * n_out, dim * n_out),
                        "{}/model_J_{}.png".format(args.save_dir, itr))
                matsave(norm_J(get_J()), "{}/model_J_norm_{}.png".format(args.save_dir, itr))

                inds = torch.triu_indices(ground_truth_C.size(0), ground_truth_C.size(1), 1)
                C_inds = ground_truth_C[inds[0], inds[1]]
                J_inds = norm_J(get_J_sub())[inds[0], inds[1]]
                J_inds_sorted = torch.sort(J_inds, descending=True).indices
                C_inds_sorted = C_inds[J_inds_sorted]
                C_cumsum = C_inds_sorted.cumsum(0)
                arange = torch.arange(C_cumsum.size(0)) + 1
                acc_at = C_cumsum.float() / arange.float()

                plt.clf()
                plt.plot(acc_at[:num_ecs].detach().cpu().numpy())
                plt.savefig("{}/acc_at_{}.png".format(args.save_dir, itr))

            if itr % args.ckpt_every == 0:
                my_print("Saving checkpoint to {}/ckpt.pt".format(args.save_dir))
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sampler": sampler.state_dict()
                }, "{}/ckpt.pt".format(args.save_dir))


            itr += 1

            if itr > args.n_iters:
                sq_err = ((ground_truth_J_norm - norm_J(get_J_sub())) ** 2).sum()
                rmse = ((ground_truth_J_norm - norm_J(get_J_sub())) ** 2).mean().sqrt()
                with open("{}/sq_err.txt".format(args.save_dir), 'w') as f:
                    f.write(str(sq_err))
                with open("{}/rmse.txt".format(args.save_dir), 'w') as f:
                    f.write(str(rmse))

                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sampler": sampler.state_dict()
                }, "{}/ckpt.pt".format(args.save_dir))

                quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--save_dir', type=str, default="/tmp/test_discrete")
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--data', type=str, default='synthetic')
    parser.add_argument('--data_file', type=str, help="location of pkl containing data")
    parser.add_argument('--data_root', type=str, default="./data")
    parser.add_argument('--graph_file', type=str, help="location of pkl containing graph") # ER
    # data generation
    parser.add_argument('--gt_steps', type=int, default=1000000)
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--sigma', type=float, default=.1)  # ising and potts
    parser.add_argument('--bias', type=float, default=0.)   # ising and potts
    parser.add_argument('--degree', type=int, default=2)  # ER
    parser.add_argument('--data_model', choices=['rbm', 'lattice_ising', 'lattice_potts', 'lattice_ising_3d',
                                                 'er_ising'],
                        type=str, default='lattice_ising')
    # models
    parser.add_argument('--model', choices=['rbm', 'lattice_ising', 'lattice_potts', 'lattice_ising_3d',
                                            'lattice_ising_2d', 'er_ising', 'dense_potts'],
                        type=str, default='lattice_ising')
    # mcmc
    parser.add_argument('--sampler', type=str, default='gibbs')
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--approx', action="store_true")
    parser.add_argument('--unweighted', action="store_true")
    parser.add_argument('--sampling_steps', type=int, default=100)
    parser.add_argument('--buffer_size', type=int, default=100)

    #
    parser.add_argument('--n_iters', type=int, default=100000)

    parser.add_argument('--n_hidden', type=int, default=25)
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--viz_batch_size', type=int, default=1000)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--viz_every', type=int, default=1000)
    parser.add_argument('--ckpt_every', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--weight_decay', type=float, default=.0)
    parser.add_argument('--l1', type=float, default=.0)
    parser.add_argument('--contact_cutoff', type=float, default=5.)

    args = parser.parse_args()
    args.device = device
    main(args)
