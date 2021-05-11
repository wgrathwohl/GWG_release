import torch
import torch.nn as nn
import toy_data
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import visualize_flow
import matplotlib.pyplot as plt
import pickle
import rbm
import samplers
from tqdm import tqdm
import os


def difference_function(x, model):
    d = torch.zeros_like(x)
    orig_out = model(x).squeeze()
    for i in range(x.size(1)):
        x_pert = x.clone()
        x_pert[:, i] = 1. - x[:, i]
        delta = model(x_pert).squeeze() - orig_out
        d[:, i] = delta
    return d

def approx_difference_function(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    wx = gx * -(2. * x - 1)
    return wx.detach()


def difference_function_multi_dim(x, model):
    d = torch.zeros_like(x)
    orig_out = model(x).squeeze()
    for i in range(x.size(1)):
        for j in range(x.size(2)):
            x_pert = x.clone()
            x_pert[:, i] = 0.
            x_pert[:, i, j] = 1.
            delta = model(x_pert).squeeze() - orig_out
            d[:, i, j] = delta
    return d


def approx_difference_function_multi_dim(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    gx_cur = (gx * x).sum(-1)[:, :, None]
    return gx - gx_cur


def short_run_mcmc(logp_net, x_init, k, sigma, step_size=None):
    x_k = torch.autograd.Variable(x_init, requires_grad=True)
    # sgld
    if step_size is None:
        step_size = (sigma ** 2.) / 2.
    for i in range(k):
        f_prime = torch.autograd.grad(logp_net(x_k).sum(), [x_k], retain_graph=True)[0]
        x_k.data += step_size * f_prime + sigma * torch.randn_like(x_k)

    return x_k


def get_data(args):
    if args.data == "mnist":
        transform = tr.Compose([tr.Resize(args.img_size), tr.ToTensor(), lambda x: (x > .5).float().view(-1)])
        train_data = torchvision.datasets.MNIST(root="../data", train=True, transform=transform, download=True)
        test_data = torchvision.datasets.MNIST(root="../data", train=False, transform=transform, download=True)
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, args.batch_size, shuffle=True, drop_last=True)
        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, args.img_size, args.img_size),
                                                         p, normalize=True, nrow=sqrt(x.size(0)))
        encoder = None
        viz = None
    elif args.data in toy_data.TOY_DSETS:
        data = []
        seen = 0
        while seen < args.n_toy_data:
            x = toy_data.inf_train_gen(args.data, batch_size=args.batch_size)
            data.append(x)
            seen += x.shape[0]
        data = np.concatenate(data, 0)
        m, M = data.min(), data.max()
        delta = M - m
        buffer = delta / 8.
        encoder = toy_data.Int2Gray(min=m - buffer, max=M + buffer)

        def plot(p, x):
            plt.clf()
            x = x.cpu().detach().numpy()
            x = encoder.decode_batch(x)
            visualize_flow.plt_samples(x, plt.gca())
            plt.savefig(p)

        def viz(p, model):
            plt.clf()
            visualize_flow.plt_flow_density(lambda x: model(encoder.encode_batch(x)), plt.gca(), npts=200)
            plt.savefig(p)

        data = torch.from_numpy(data).float()
        e_data = encoder.encode_batch(data)
        y = torch.zeros_like(data[:, 0])
        train_data = TensorDataset(e_data, y)
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
        test_loader = train_loader

    elif args.data_file is not None:
        with open(args.data_file, 'rb') as f:
            x = pickle.load(f)
        x = torch.tensor(x).float()
        train_data = TensorDataset(x)
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
        test_loader = train_loader
        viz = None
        if args.model == "lattice_ising" or args.model == "lattice_ising_2d":
            plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, args.dim, args.dim),
                                                             p, normalize=False, nrow=int(x.size(0) ** .5))
        elif args.model == "lattice_potts":
            plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), args.dim, args.dim, 3).transpose(3, 1),
                                                             p, normalize=False, nrow=int(x.size(0) ** .5))
        else:
            plot = lambda p, x: None
    else:
        raise ValueError

    return train_loader, test_loader, plot, viz


def generate_data(args):
    if args.data_model == "lattice_potts":
        model = rbm.LatticePottsModel(args.dim, args.n_state, args.sigma)
        sampler = samplers.PerDimMetropolisSampler(model.data_dim, args.n_out, rand=False)
    elif args.data_model == "lattice_ising":
        model = rbm.LatticeIsingModel(args.dim, args.sigma)
        sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=False)
    elif args.data_model == "lattice_ising_3d":
        model = rbm.LatticeIsingModel(args.dim, args.sigma, lattice_dim=3)
        sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=False)
        print(model.sigma)
        print(model.G)
        print(model.J)
    elif args.data_model == "er_ising":
        model = rbm.ERIsingModel(args.dim, args.degree, args.sigma)
        sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=False)
        print(model.G)
        print(model.J)
    else:
        raise ValueError

    model = model.to(args.device)
    samples = model.init_sample(args.n_samples).to(args.device)
    print("Generating {} samples from:".format(args.n_samples))
    print(model)
    for _ in tqdm(range(args.gt_steps)):
        samples = sampler.step(samples, model).detach()

    return samples.detach().cpu(), model


def load_synthetic(mat_file, batch_size):
    import scipy.io
    mat = scipy.io.loadmat(mat_file)
    ground_truth_J = mat['eij']
    ground_truth_h = mat['hi']
    ground_truth_C = mat['C']
    q = mat['q']
    n_out = q[0, 0]

    x_int = mat['sample']
    n_samples, dim = x_int.shape

    x_int = torch.tensor(x_int).long() - 1
    x_oh = torch.nn.functional.one_hot(x_int, n_out)
    assert x_oh.size() == (n_samples, dim, n_out)

    x = torch.tensor(x_oh).float()
    train_data = TensorDataset(x)
    train_loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    test_loader = train_loader

    J = torch.tensor(ground_truth_J)
    j = J
    jt = j.transpose(0, 1).transpose(2, 3)
    ground_truth_J = (j + jt) / 2
    return train_loader, test_loader, x, \
           torch.tensor(ground_truth_J), torch.tensor(ground_truth_h), torch.tensor(ground_truth_C)



def load_real_protein(args):
    from data_utils import Alignment, load_distmap, MyDCA, map_matrix
    a2m = {
        "BPT1_BOVIN": f"{args.data_root}/BPT1_BOVIN/BPT1_BOVIN_full_b03.a2m",
        "CADH1_HUMAN": f"{args.data_root}/CADH1_HUMAN/CADH1_HUMAN_full_b02.a2m",
        "CHEY_ECOLI": f"{args.data_root}/CHEY_ECOLI/CHEY_ECOLI_full_b09.a2m",
        "ELAV4_HUMAN": f"{args.data_root}/ELAV4_HUMAN/ELAV4_HUMAN_full_b02.a2m",
        "O45418_CAEEL": f"{args.data_root}/O45418_CAEEL/O45418_CAEEL_full_b02.a2m",
        "OMPR_ECOLI": f"{args.data_root}/OMPR_ECOLI/OMPR_ECOLI_full_b08.a2m",
        "OPSD_BOVIN": f"{args.data_root}/OPSD_BOVIN/OPSD_BOVIN_full_b03.a2m",
        "PCBP1_HUMAN": f"{args.data_root}/PCBP1_HUMAN/PCBP1_HUMAN_full_b01.a2m",
        "RNH_ECOLI": f"{args.data_root}/RNH_ECOLI/RNH_ECOLI_full_b04.a2m",
        "THIO_ALIAC": f"{args.data_root}/THIO_ALIAC/THIO_ALIAC_full_b09.a2m",
        "TRY2_RAT": f"{args.data_root}/TRY2_RAT/TRY2_RAT_full_b02.a2m",
    }[args.data]
    print("Loading alignment...")
    with open(a2m, "r") as infile:
        aln = Alignment.from_file(infile, format="fasta")
    print("Done")
    print("Loading distmap(s)")
    intra_file = {
        "BPT1_BOVIN": f"{args.data_root}/BPT1_BOVIN/BPT1_BOVIN_full_b03_distance_map_monomer",
        "CADH1_HUMAN": f"{args.data_root}/CADH1_HUMAN/CADH1_HUMAN_full_b02_distance_map_monomer",
        "CHEY_ECOLI": f"{args.data_root}/CHEY_ECOLI/CHEY_ECOLI_full_b09_distance_map_monomer",
        "ELAV4_HUMAN": f"{args.data_root}/ELAV4_HUMAN/ELAV4_HUMAN_full_b02_distance_map_monomer",
        "O45418_CAEEL": f"{args.data_root}/O45418_CAEEL/O45418_CAEEL_full_b02_distance_map_monomer",
        "OMPR_ECOLI": f"{args.data_root}/OMPR_ECOLI/OMPR_ECOLI_full_b08_distance_map_monomer",
        "OPSD_BOVIN": f"{args.data_root}/OPSD_BOVIN/OPSD_BOVIN_full_b03_distance_map_monomer",
        "PCBP1_HUMAN": f"{args.data_root}/PCBP1_HUMAN/PCBP1_HUMAN_full_b01_distance_map_monomer",
        "RNH_ECOLI": f"{args.data_root}/RNH_ECOLI/RNH_ECOLI_full_b04_distance_map_monomer",
        "THIO_ALIAC": f"{args.data_root}/THIO_ALIAC/THIO_ALIAC_full_b09_distance_map_monomer",
        "TRY2_RAT": f"{args.data_root}/TRY2_RAT/TRY2_RAT_full_b02_distance_map_monomer",
    }[args.data]
    distmap_intra = load_distmap(intra_file)

    multimer_file = {
        "BPT1_BOVIN": f"{args.data_root}/BPT1_BOVIN/BPT1_BOVIN_full_b03_distance_map_multimer",
        "CADH1_HUMAN": f"{args.data_root}/CADH1_HUMAN/CADH1_HUMAN_full_b02_distance_map_multimer",
        "CHEY_ECOLI": f"{args.data_root}/CHEY_ECOLI/CHEY_ECOLI_full_b09_distance_map_multimer",
        "ELAV4_HUMAN": f"{args.data_root}/ELAV4_HUMAN/ELAV4_HUMAN_full_b02_distance_map_multimer",
        "O45418_CAEEL": f"{args.data_root}/O45418_CAEEL/O45418_CAEEL_full_b02_distance_map_multimer",
        "OMPR_ECOLI": f"{args.data_root}/OMPR_ECOLI/OMPR_ECOLI_full_b08_distance_map_multimer",
        "OPSD_BOVIN": None,
        "PCBP1_HUMAN": f"{args.data_root}/PCBP1_HUMAN/PCBP1_HUMAN_full_b01_distance_map_multimer",
        "RNH_ECOLI": f"{args.data_root}/RNH_ECOLI/RNH_ECOLI_full_b04_distance_map_multimer",
        "THIO_ALIAC": f"{args.data_root}/THIO_ALIAC/THIO_ALIAC_full_b09_distance_map_multimer",
        "TRY2_RAT": None,
    }[args.data]
    if multimer_file is None:
        distmap_multimer = None
    else:
        distmap_multimer = load_distmap(multimer_file)

    num_ecs = {
        "BPT1_BOVIN": 57,
        "CADH1_HUMAN": 324,
        "CHEY_ECOLI": 114,
        "ELAV4_HUMAN": 140,
        "O45418_CAEEL": 234,
        "OMPR_ECOLI": 220,
        "OPSD_BOVIN": 266,
        "PCBP1_HUMAN": 190,
        "RNH_ECOLI": 133,
        "THIO_ALIAC": 95,
        "TRY2_RAT": 209,
    }[args.data]
    print("Done")
    print("Pulling data")
    L = aln.L
    D = len(aln.alphabet)
    print("Raw Data size {}".format((L, D)))


    dca = MyDCA(aln)
    #dca.alignment.set_weights()
    #print(dca.alignment.weights.sum(), "MY DCA SUM")
    L = dca.alignment.L
    D = len(dca.alignment.alphabet)
    x_int = torch.from_numpy(dca.int_matrix()).float()
    x_oh = torch.nn.functional.one_hot(x_int.long(), D).float()
    print("Filtered Data size {}".format((L, D)))


    J = -distmap_intra.dist_matrix
    J = J + args.contact_cutoff
    J[J < 0] = 0.
    J[np.isnan(J)] = 0.  # treat unobserved values as just having no contact
    ind = np.diag_indices(J.shape[0])
    J[ind] = 0.
    C = np.copy(J)
    C[C > 0] = 1.
    C[C <= 0] = 0.
    print("Done")
    print("J size = {}".format(J.shape))

    weight_file = f"{args.data_root}/{args.data}/weights.pkl"
    if not os.path.exists(weight_file):
        print("Generating weights")
        dca.alignment.set_weights()
        weights = dca.alignment.weights
        with open(weight_file, 'wb') as f:
            pickle.dump(weights, f)
    else:
        print("Loading weights")
        with open(weight_file, 'rb') as f:
            weights = pickle.load(f)

    weights = torch.tensor(weights).float()
    print("Done")
    print("Dataset has {} examples, sum weights are {}".format(weights.size(0), weights.sum()))
    print("Scaling up by {}".format(float(weights.size(0)) / weights.sum()))
    weights *= float(weights.size(0)) / weights.sum()
    print("Distmap size {}".format(J.shape))

    train_data = TensorDataset(x_oh, weights)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
    test_loader = train_loader

    # pull indices from distance map
    #dm_indices = list(torch.tensor(np.array(distmap_intra.residues_j.id).astype(int) - 1).numpy())
    dm_indices = list(torch.tensor(np.array(distmap_intra.residues_j.id).astype(int)).numpy())
    print(dm_indices)
    dca_indices = dca.index_list
    print(dca_indices)
    int_indices = list(set(dm_indices).intersection(set(dca_indices)))
    dm_int_indices = []
    for i, ind in enumerate(dm_indices):
        if ind in int_indices:
            dm_int_indices.append(i)

    dca_int_indices = []
    for i, ind in enumerate(dca_indices):
        if ind in int_indices:
            dca_int_indices.append(i)

    print(dm_int_indices)
    print(dca_int_indices)

    print(len(dm_int_indices))
    print(len(dca_int_indices))

    print("Removing indices from C and J")
    print("Old size: {}".format(C.shape))
    C = C[dm_int_indices][:, dm_int_indices]
    J = J[dm_int_indices][:, dm_int_indices]
    print("New size: {}".format(C.shape))
    dca_int_indices = torch.tensor(dca_int_indices).long()
    print(dca_int_indices)
    return train_loader, test_loader, x_oh, num_ecs, torch.tensor(J), torch.tensor(C), dca_int_indices

def load_ingraham(args):
    from data_utils import Alignment, map_matrix
    with open("{}/PF00018.a2m".format(args.data_root), "r") as infile:
        aln = Alignment.from_file(infile, format="fasta")

    L = aln.L
    D = len(aln.alphabet)
    x_int = torch.from_numpy(map_matrix(aln.matrix, aln.alphabet_map))
    n_out = D
    x_oh = torch.nn.functional.one_hot(x_int.long(), n_out).float()

    print(L, D, x_oh.size())

    aln.set_weights()
    weights = torch.tensor(aln.weights).float()
    print("Dataset has {} examples, sum weights are {}".format(weights.size(0), weights.sum()))
    print("Scaling up by {}".format(float(weights.size(0)) / weights.sum()))
    weights *= float(weights.size(0)) / weights.sum()

    with open("{}/PF00018_summary.txt".format(args.data_root), 'r') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
        X = [int(line[0]) for line in lines]
        Y = [int(line[1]) for line in lines]
        M = [float(line[5]) for line in lines]
        D = np.zeros((48, 48))
        for x, y, m in zip(X, Y, M):
            D[x - 1, y - 1] = m
            J = -(D + D.T)
            J = J + args.contact_cutoff
            J[J < 0] = 0.
            ind = np.diag_indices(J.shape[0])
            J[ind] = 0.
            C = np.copy(J)
            C[C > 0] = 1.
            C[C <= 0] = 0.

    print("Distmap size {}".format(J.shape))

    train_data = TensorDataset(x_oh, weights)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
    test_loader = train_loader

    return train_loader, test_loader, x_oh, 200, torch.tensor(J), torch.tensor(C)

