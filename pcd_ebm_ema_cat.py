import argparse
import mlp
import torch
import torch.nn.functional
import numpy as np
import samplers
import block_samplers
import torch.nn as nn
import os
import torchvision
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import vamp_utils
import ais
import copy
from mlp import MLPEBM_cat  # Assuming this is the module containing the EBM architecture
from tqdm import tqdm


def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_sampler(args):
    data_dim = np.prod(args.input_size)
    sampler = samplers.DiffSamplerMultiDim(data_dim, 1, approx=True, temp=2.)
    return sampler


class MyOneHotCategorical:
    def __init__(self, mean):
        self.dist = torch.distributions.OneHotCategorical(probs=mean)

    def sample(self, x):
        return self.dist.sample(x)

    def log_prob(self, x):
        logits = self.dist.logits
        lp = torch.log_softmax(logits, -1)
        return (x * lp[None]).sum(-1)


class EBM(nn.Module):
    def __init__(self, net, mean=None):
        super().__init__()
        self.net = net
        if mean is None:
            self.mean = None
        else:
            self.mean = nn.Parameter(mean, requires_grad=False)

    def forward(self, x):
        if self.mean is None:
            bd = 0.
        else:
            base_dist = MyOneHotCategorical(self.mean)
            bd = base_dist.log_prob(x).sum(-1)

        logp = self.net(x).squeeze()
        return logp + bd


def main(args):
    makedirs(args.save_dir)
    logger = open("{}/log.txt".format(args.save_dir), 'w')

    def my_print(s):
        print(s)
        logger.write(str(s) + '\n')
        logger.flush()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load data
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
    def plot(p, x):
        ar = torch.arange(x.size(-1)).to(x.device)
        x_int = (x * ar[None, None, :]).sum(-1)
        x_int = x_int.view(x.size(0), args.input_size[0], args.input_size[1], args.input_size[2])
        torchvision.utils.save_image(x_int, p, normalize=True, nrow=int(x.size(0) ** .5))

    def preprocess(data):
        x_int = (data * 255).long()
        x_oh = torch.nn.functional.one_hot(x_int, 256).float()
        return x_oh

    def bits_per_dim(ll):
        nll = -ll
        num_pixels = np.prod(args.input_size)
        return (nll / num_pixels) / np.log(2)


    # make model
    # data in form [batch size, n dims, n outcomes]
    if args.model.startswith("mlp-"):
        nint = int(args.model.split('-')[1])
        net = mlp.MLPEBM_cat(np.prod(args.input_size), args.proj_dim, 256, nint)
    elif args.model.startswith("resnet-"):
        nint = int(args.model.split('-')[1])
        net = mlp.ResNetEBM_cat(args.input_size, args.proj_dim, 256, nint)
    else:
        raise ValueError("invalid model definition")


    # get data mean and initialize buffer
    init_batch = []
    for x, _ in train_loader:
        init_batch.append(preprocess(x))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2 / 256
    init_mean = init_batch.mean(0) + eps
    init_mean = init_mean / init_mean.sum(-1)[:, None]

    if args.buffer_init == "mean":
        init_dist = MyOneHotCategorical(init_mean)
        buffer = init_dist.sample((args.buffer_size,))
    else:
        raise ValueError("Invalid init")

    if args.base_dist:
        model = EBM(net, init_mean)
    else:
        model = EBM(net, init_mean)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ema_model = copy.deepcopy(model)

    if args.ckpt_path is not None:
        d = torch.load(args.ckpt_path)
        model.load_state_dict(d['model'])
        ema_model.load_state_dict(d['ema_model'])
        buffer = d['buffer']



    # move to cuda
    model.to(device)
    ema_model.to(device)

    # get sampler
    sampler = get_sampler(args)

    my_print(device)
    my_print(model)
    my_print(buffer.size())
    my_print(sampler)

    itr = 0
    best_val_ll = -np.inf
    hop_dists = []
    all_inds = list(range(args.buffer_size))
    lr = args.lr
    init_dist = MyOneHotCategorical(init_mean.to(device))
    rand_img = torch.randint(low=0, high=256, size=(100,) + (784,)).to(device) / 255.
    rand_img = preprocess(rand_img)
    i = 0

    while itr < args.n_iters:
        for x in tqdm(train_loader):
            if itr < args.warmup_iters:
                lr = args.lr * float(itr) / args.warmup_iters
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            x = preprocess(x[0].to(device).requires_grad_())

            # choose random inds from buffer
            buffer_inds = sorted(np.random.choice(all_inds, args.batch_size, replace=False))
            x_buffer = buffer[buffer_inds].to(device)
            x_fake = x_buffer

            hops = []  # keep track of how much the sampelr moves particles around
            for k in range(args.sampling_steps):
                x_fake_new = sampler.step(x_fake.detach(), model).detach()
                h = (x_fake_new != x_fake).float().view(x_fake_new.size(0), -1).sum(-1).mean().item()
                hops.append(h)
                x_fake = x_fake_new
            hop_dists.append(np.mean(hops))

            # update buffer
            buffer[buffer_inds] = x_fake.detach().cpu()

            logp_real = model(x).squeeze()
            if args.p_control > 0:
                grad_ld = torch.autograd.grad(logp_real.sum(), x,
                                              create_graph=True)[0].flatten(start_dim=1).norm(2, 1)
                grad_reg = (grad_ld ** 2. / 2.).mean() * args.p_control
            else:
                grad_reg = 0.0

            logp_fake = model(x_fake).squeeze()

            obj = logp_real.mean() - logp_fake.mean()
            loss = -obj + grad_reg + args.l2 * ((logp_real ** 2.).mean() + (logp_fake ** 2.).mean())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k in range(10):
                rand_img = sampler.step(rand_img.detach(), model).detach()  

            # update ema_model
            for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

            if itr % args.print_every == 0:
                my_print("({}) | cur lr = {:.4f} |log p(real) = {:.4f}, "
                         "log p(fake) = {:.4f}, diff = {:.4f}, hops = {:.4f}".format(itr, lr, logp_real.mean().item(),
                                                                                     logp_fake.mean().item(), obj.item(),
                                                                                     hop_dists[-1]))

            if itr % args.viz_every == 0:
                print("#############PLOT BUFFER#############")
                plot("output_img/data_{}.png".format(itr), x.detach().cpu())
                plot("output_img/buffer_{}.png".format(itr), x_fake)
                plot("output_img/gen_{}.png".format(i), rand_img.detach().cpu())
                i += 1
                
            if (itr + 1) % args.eval_every == 0:
                logZ, train_ll, val_ll, test_ll, ais_samples = ais.evaluate(ema_model, init_dist, sampler,
                                                                            train_loader, val_loader, test_loader,
                                                                            preprocess, device,
                                                                            args.eval_sampling_steps,
                                                                            args.test_batch_size)
                my_print("EMA Train log-likelihood ({}): {}, {} bpd".format(itr, train_ll.item(),
                                                                            bits_per_dim(train_ll.item())))
                my_print("EMA Valid log-likelihood ({}): {}, {} bpd".format(itr, val_ll.item(),
                                                                            bits_per_dim(val_ll.item())
                                                                            ))
                my_print("EMA Test log-likelihood ({}): {}, {} bpd".format(itr, test_ll.item(),
                                                                           bits_per_dim(test_ll.item())
                                                                           ))
                for _i, _x in enumerate(ais_samples):
                    plot("{}/EMA_sample_{}_{}.png".format(args.save_dir, itr, _i), _x)

                model.cpu()
                d = {}
                d['model'] = model.state_dict()
                d['ema_model'] = ema_model.state_dict()
                d['buffer'] = buffer
                d['optimizer'] = optimizer.state_dict()

                if val_ll.item() > best_val_ll:
                    best_val_ll = val_ll.item()
                    my_print("Best valid likelihood")

                model.to(device)

            itr += 1
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--save_dir', type=str, default="tmp/test_discrete")
    parser.add_argument('--dataset_name', type=str, default='cat', choices=["cat", "freyfaces", "histopathology"])
    parser.add_argument('--ckpt_path', type=str, default=None)
    # models
    parser.add_argument('--model', type=str, default='mlp-256')
    parser.add_argument('--base_dist', action='store_true')
    parser.add_argument('--p_control', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--ema', type=float, default=0.999)
    parser.add_argument('--proj_dim', type=int, default=4)
    # mcmc
    parser.add_argument('--sampler', type=str, default='gwg')
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--sampling_steps', type=int, default=100)
    parser.add_argument('--reinit_freq', type=float, default=0.0)
    parser.add_argument('--eval_sampling_steps', type=int, default=100)
    parser.add_argument('--buffer_size', type=int, default=1000)
    parser.add_argument('--buffer_init', type=str, default='mean')
    # training
    parser.add_argument('--n_iters', type=int, default=3001)
    parser.add_argument('--warmup_iters', type=int, default=-1)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--viz_every', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight_decay', type=float, default=.0)

    args = parser.parse_args()
    args.device = device
    os.makedirs('output_img', exist_ok=True)
    main(args)