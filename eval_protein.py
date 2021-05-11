import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import utils
import pickle


def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def norm_J(J):
    return J.norm(dim=(2, 3))


def matsave(M, path):
    plt.clf()
    plt.matshow(M.detach().cpu().numpy())
    plt.colorbar()
    plt.savefig(path)


def top_k_mat(M, k):
    inds = torch.triu_indices(M.size(0), M.size(1), 1)
    M_inds = M[inds[0], inds[1]]
    Ms = torch.sort(M_inds, 0, descending=True).values
    kth = Ms[k - 1]

    out = M.clone()
    out[M < kth] = 0.
    out[M >= kth] = 1.
    return out


def main(args):
    makedirs(args.save_dir)
    logger = open("{}/log.txt".format(args.save_dir), 'w')

    def my_print(s):
        print(s)
        logger.write(str(s) + '\n')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    datasets = "PF00018 OPSD_BOVIN CHEY_ECOLI"
    datasets = datasets.split()
    for dataset in datasets:
        print("Loading {}".format(dataset))
        makedirs("{}/{}".format(args.save_dir, dataset))
        args.data = dataset
        if args.data == "PF00018":
            train_loader, test_loader, data, num_ecs, ground_truth_J_norm, ground_truth_C = utils.load_ingraham(args)
            dim, n_out = data.size()[1:]
            ground_truth_J_norm = ground_truth_J_norm.to(device)
            dm_indices = torch.arange(ground_truth_J_norm.size(0)).long()

        else:
            train_loader, test_loader, data, num_ecs, ground_truth_J_norm, ground_truth_C, dm_indices = utils.load_real_protein(args)
            dim, n_out = data.size()[1:]
            ground_truth_J_norm = ground_truth_J_norm.to(device)

        num_ecs_real = int(ground_truth_C.sum().item() / 2)
        print(num_ecs, num_ecs_real, "ECS!!!")

        matsave(ground_truth_C, "{}/{}/ground_truth_C.png".format(args.save_dir, dataset))
        matsave(ground_truth_J_norm, "{}/{}/ground_truth_dists.png".format(args.save_dir, dataset))

        all_acc_ats = {}
        samplers = ["gibbs", "gwg", "plm"]
        l1s = [".01", ".03"]
        for l1 in l1s:
            out_dir = "{}/{}/l1_{}".format(args.save_dir, dataset, l1)
            makedirs(out_dir)
            acc_ats = {}
            for sampler in samplers:
                if sampler == "plm":
                    base_dir = "/scratch/gobi2/gwohl/GWG_EXPERIMENTS/protein"
                    ckpt_file = "{}/{}/sampler_{}_steps_20_l1_{}_weighted/ckpt.pt".format(base_dir, dataset, sampler, l1)
                else:
                    base_dir = "/scratch/gobi2/gwohl/GWG_EXPERIMENTS/protein_big_buffer"
                    ckpt_file = "{}/{}/sampler_{}_steps_20_l1_{}/ckpt.pt".format(base_dir, dataset, sampler, l1)
                print("Loading ckpt {}".format(ckpt_file))
                ckpt = torch.load(ckpt_file)
                model = ckpt['model']
                J = model['J']

                # make G symmetric
                def get_J():
                    j = J
                    jt = j.transpose(0, 1).transpose(2, 3)
                    return (j + jt) / 2

                def get_J_sub():
                    j = get_J()
                    j_sub = j[dm_indices, :][:, dm_indices]
                    return j_sub

                print("Visualize matrices")
                matsave(get_J_sub().abs().transpose(2, 1).reshape(dm_indices.size(0) * n_out,
                                                                  dm_indices.size(0) * n_out),
                        "{}/J_sub_{}.png".format(out_dir, sampler))
                matsave(norm_J(get_J_sub()), "{}/J_norm_{}_sub.png".format(out_dir, sampler))

                matsave(get_J().abs().transpose(2, 1).reshape(dim * n_out, dim * n_out),
                        "{}/J_{}.png".format(out_dir, sampler))
                matsave(norm_J(get_J()), "{}/J_norm_{}.png".format(out_dir, sampler))

                # get top vals
                sub_norms = norm_J(get_J_sub())
                sub_norms_top_k = top_k_mat(sub_norms, num_ecs)
                matsave(sub_norms_top_k, "{}/J_sub_norm_top_{}_{}.png".format(out_dir, num_ecs, sampler))
                sub_norms_top_l = top_k_mat(sub_norms, sub_norms.size(0))
                matsave(sub_norms_top_l, "{}/J_sub_norm_top_{}_{}.png".format(out_dir, sub_norms.size(0), sampler))
                sub_norms_top_l = top_k_mat(sub_norms, 2 * sub_norms.size(0))
                matsave(sub_norms_top_l, "{}/J_sub_norm_top_{}_{}.png".format(out_dir, 2 * sub_norms.size(0), sampler))
                sub_norms_top_l = top_k_mat(sub_norms, 4 * sub_norms.size(0))
                matsave(sub_norms_top_l, "{}/J_sub_norm_top_{}_{}.png".format(out_dir, 4 * sub_norms.size(0), sampler))

                norms = norm_J(get_J())
                norms_top_l = top_k_mat(norms, norms.size(0))
                matsave(norms_top_l, "{}/J_norm_top_{}_{}.png".format(out_dir, norms.size(0), sampler))
                norms_top_l = top_k_mat(norms, 2 * norms.size(0))
                matsave(norms_top_l, "{}/J_norm_top_{}_{}.png".format(out_dir, 2 * norms.size(0), sampler))
                norms_top_l = top_k_mat(norms, 4 * norms.size(0))
                matsave(norms_top_l, "{}/J_norm_top_{}_{}.png".format(out_dir, 4 * norms.size(0), sampler))


                print("Get acc at values")
                inds = torch.triu_indices(ground_truth_C.size(0), ground_truth_C.size(1), 1)
                C_inds = ground_truth_C[inds[0], inds[1]]
                J_inds = norm_J(get_J_sub())[inds[0], inds[1]]
                J_inds_sorted = torch.sort(J_inds, descending=True).indices
                C_inds_sorted = C_inds[J_inds_sorted]
                C_cumsum = C_inds_sorted.cumsum(0)
                arange = torch.arange(C_cumsum.size(0)) + 1
                acc_at = C_cumsum.float() / arange.float()
                acc_ats[sampler] = acc_at.detach().cpu().numpy()

            # save acc ats
            with open("{}/acc_ats.pkl".format(out_dir), 'wb') as f:
                pickle.dump(acc_ats, f)


            plt.clf()
            for sampler in samplers:
                plt.plot(acc_ats[sampler][:num_ecs_real], label=sampler)
            plt.legend()
            plt.savefig("{}/acc_at.png".format(out_dir))

            all_acc_ats[l1] = acc_ats

        plt.clf()
        for l1 in l1s:
            for sampler in samplers:
                plt.plot(all_acc_ats[l1][sampler][:num_ecs_real], label="{}-{}".format(sampler, l1))
        plt.legend()
        out_dir = "{}/{}".format(args.save_dir, dataset)
        plt.savefig("{}/acc_at_all.png".format(out_dir))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--save_dir', type=str, default="/tmp/test_discrete")
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
    parser.add_argument('--seed', type=int, default=347455)
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
