"""
Toy data utilities.
"""

import numpy as np
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
import torch

TOY_DSETS = ("moons", "circles", "8gaussians", "pinwheel", "2spirals", "checkerboard", "rings", "swissroll")


class Int2Gray:
    def __init__(self, nbits=16, nint=10**4, min=0., max=1.):
        self.int2gray = []
        for i in range(0, 1 << nbits):
            gray = i ^ (i >> 1)
            self.int2gray.append(gray)
        self.gray2int = {g:i for i, g in enumerate(self.int2gray)}
        self.min = min
        self.max = max
        self.nint = nint
        self.nbits = nbits

    def encode(self, x):
        assert x > self.min and x < self.max
        x = (x - self.min) / (self.max - self.min)
        xi = int(x * self.nint)
        g = self.int2gray[xi]
        bs = "{0:0{1}b}".format(g, self.nbits)
        ba = np.zeros((self.nbits,))
        for i, b in enumerate(bs):
            ba[i] = float(b)
        return ba

    def decode(self, g):
        g = g.astype('int').astype('str')
        gs = "".join(list(g))
        g = int(gs, 2)
        xi = self.gray2int[g]
        x = xi / float(self.nint)
        x = x * (self.max - self.min) + self.min
        return x

    def encode_batch(self, x):
        if isinstance(x, np.ndarray):
            xx, xy = x[:, 0], x[:, 1]
            xx = np.array([self.encode(_x) for _x in xx])
            xy = np.array([self.encode(_x) for _x in xy])
            x = np.concatenate([xx, xy], 1)
            return x
        elif torch.is_tensor(x):
            x = x.numpy()
            x = self.encode_batch(x)
            return torch.from_numpy(x).float()
        else:
            raise ValueError("only works for torch tensor or np array, given {}".format(type(x)))

    def decode_batch(self, g):
        if isinstance(g, np.ndarray):
            gx, gy = g[:, :self.nbits], g[:, self.nbits:]
            xx = np.array([self.decode(_x) for _x in gx])
            xy = np.array([self.decode(_x) for _x in gy])
            x = np.concatenate([xx[:, None], xy[:, None]], 1)
            return x
        elif torch.is_tensor(g):
            g = g.numpy()
            x = self.decode_batch(g)
            return torch.from_numpy(x).float()
        else:
            raise ValueError("only works for torch tensor or np array, given {}".format(type(x)))


class RingDist:
    def __init__(self, r, std):
        self.r = r
        self.std = std
        self.dist = torch.distributions.Normal(torch.tensor(r).float(), torch.tensor(std).float())

    def logpx(self, x):
        r = x.norm(2, 1)
        return self.dist.log_prob(r)

class RingsDist:
    def __init__(self, rs, stds):
        self.dists = [RingDist(r, std) for r, std in zip(rs, stds)]

    def logpx(self, x):
        lps = torch.cat([dist.logpx(x)[:, None] for dist in self.dists], 1).logsumexp(1)[:, None]
        return lps

class MOGCross:
    def __init__(self, std=.5):
        mus = [
            [-3, 0],
            [-2, 0],
            [-1, 0],
            [-0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [0, -3],
            [0, -2],
            [0, -1],
            [0, 1],
            [0, 2],
            [0, 3],
        ]
        mus = [torch.tensor(mu).float() for mu in mus]

        self.dists = [torch.distributions.Normal(loc=mu, scale=std) for mu in mus]

    def logpx(self, x):
        lps = [dist.log_prob(x).sum(1)[:, None] for dist in self.dists]
        lps = torch.cat(lps, 1)
        lp = lps.logsumexp(1) - np.log(float(len(self.dists)))
        return lp


def data_density(data):
    """
    Returns function which outputs energy of data (e.g. unnormalized density).
    """
    if data == "8gaussians":
        # output squared distance from all the centers.
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = torch.Tensor([(scale * x, scale * y) for x, y in centers])

        class Energy(torch.nn.Module):
            """
            The energy for the 8gaussians.
            """
            def __init__(self):
                super(Energy, self).__init__()
                self.centers = torch.nn.Parameter(centers[None], requires_grad=False)

            def forward(self, x):
                x = x * 1.414
                log_p = -0.5 * (x.unsqueeze(1) - self.centers).square() + -0.5 * np.log(2 * np.pi)
                return log_p.sum(2).logsumexp(1).unsqueeze(1)

        return Energy()
    else:
        raise ValueError


def inf_train_gen(data, rng=None, batch_size=200):
    """
    Create infinite generator of toy data.
    """
    if rng is None:
        rng = np.random.RandomState()

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "rings":
        obs = batch_size
        batch_size *= 20
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X, random_state=rng)

        # Add noise
        X += rng.normal(scale=0.08, size=X.shape)
        inds = np.random.choice(list(range(batch_size)), obs)
        X = X[inds]
        return X.astype("float32")

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data

    elif data == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    else:
        assert False



