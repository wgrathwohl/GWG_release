import argparse
import torch
import numpy as np
import os
import torchvision
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import vamp_utils
import mlp
from pcd_ebm_ema_cat import get_sampler, EBM, MyOneHotCategorical
import ais


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

    my_print("Loading data")
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)

    def plot(p, x):
        ar = torch.arange(x.size(-1)).to(x.device)
        x_int = (x * ar[None, None, :]).sum(-1)
        x_int = x_int.view(x.size(0), args.input_size[0], args.input_size[1], args.input_size[2])
        torchvision.utils.save_image(x_int, p, normalize=True, nrow=int(x.size(0) ** .5))

    def bits_per_dim(ll):
        nll = -ll
        num_pixels = np.prod(args.input_size)
        return (nll / num_pixels) / np.log(2)

    def preprocess(data):
        x_int = (data * 256).long()
        x_oh = torch.nn.functional.one_hot(x_int, 256).float()
        return x_oh

    # make model
    my_print("Making Model")
    # make model
    if args.model.startswith("mlp-"):
        nint = int(args.model.split('-')[1])
        net = mlp.MLPEBM_cat(np.prod(args.input_size), args.proj_dim, 256, nint)
    elif args.model.startswith("resnet-"):
        nint = int(args.model.split('-')[1])
        net = mlp.ResNetEBM_cat(args.input_size, args.proj_dim, 256, nint)
    else:
        raise ValueError("invalid model definition")

    # get data mean and initialize buffer
    my_print("Getting init batch")
    init_batch = []
    for x, _ in train_loader:
        init_batch.append(preprocess(x))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2 / 256
    init_mean = init_batch.mean(0) + eps
    init_mean = init_mean / init_mean.sum(-1)[:, None]


    if args.base_dist:
        model = EBM(net, init_mean)
    else:
        model = EBM(net)

    if args.ckpt_path is not None:
        d = torch.load(args.ckpt_path)
        if args.ema:
            model.load_state_dict(d['ema_model'])
        else:
            model.load_state_dict(d['model'])
        buffer = d['buffer']

    # wrap model for annealing
    init_dist = MyOneHotCategorical(init_mean.to(device))

    # get sampler
    sampler = get_sampler(args)

    my_print(device)
    my_print(model)
    my_print(sampler)

    logZ, train_ll, val_ll, test_ll, ais_samples = ais.evaluate(model, init_dist, sampler,
                                                                train_loader, val_loader, test_loader,
                                                                preprocess, device,
                                                                args.eval_sampling_steps,
                                                                args.n_samples, viz_every=args.viz_every)

    my_print("Train log-likelihood: {}, {} bpd".format(train_ll.item(), bits_per_dim(train_ll.item())))
    my_print("Valid log-likelihood: {}, {} bpd".format(val_ll.item(), bits_per_dim(val_ll.item())))
    my_print("Test log-likelihood: {}, {} bpd".format(test_ll.item(), bits_per_dim(test_ll.item())))
    for _i, _x in enumerate(ais_samples):
        plot("{}/EMA_sample_{}.png".format(args.save_dir, _i), _x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--save_dir', type=str, default="/tmp/test_discrete")
    parser.add_argument('--dataset_name', type=str, default='static_mnist')
    parser.add_argument('--ckpt_path', type=str, default=None)
    # data generation
    parser.add_argument('--n_out', type=int, default=3)  # potts
    # models
    parser.add_argument('--model', type=str, default='mlp-256')
    parser.add_argument('--base_dist', action='store_true')
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--p_control', type=float, default=0.0)
    # mcmc
    parser.add_argument('--sampler', type=str, default='gibbs')
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--sampling_steps', type=int, default=100)
    parser.add_argument('--steps_per_iter', type=int, default=1)
    parser.add_argument('--eval_sampling_steps', type=int, default=100)
    parser.add_argument('--buffer_size', type=int, default=1000)
    parser.add_argument('--buffer_init', type=str, default='mean')
    # training
    parser.add_argument('--n_iters', type=int, default=100000)
    parser.add_argument('--warmup_iters', type=int, default=-1)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--viz_every', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight_decay', type=float, default=.0)
    parser.add_argument('--proj_dim', type=int, default=4)

    args = parser.parse_args()
    args.device = device
    main(args)
