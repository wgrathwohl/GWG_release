from __future__ import print_function

import torch
import torch.utils.data as data_utils
import torchvision
from torchvision import datasets, transforms
import numpy as np

from scipy.io import loadmat
import os

import pickle

def load_dynamic_mnist_cat(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'cat'
    args.dynamic_binarization = False
    
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the training and test data
    mnist_trainset = datasets.MNIST('~/.pytorch/MNIST_data/', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST('~/.pytorch/MNIST_data/', train=False, download=True, transform=transform)

    # Filter the data for only digits 0 and 1
    indices_train = ((mnist_trainset.targets == 0) | (mnist_trainset.targets == 1))
    indices_test = ((mnist_testset.targets == 0) | (mnist_testset.targets == 1))

    # Use these indices to create new subsets with only 0 and 1 digits
    train_data = torch.utils.data.Subset(mnist_trainset, torch.where(indices_train)[0])
    test_data = torch.utils.data.Subset(mnist_testset, torch.where(indices_test)[0])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # preparing data
    x_train = train_data.dataset.data[indices_train].float().numpy() / 255.
    x_train = np.reshape( x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2] ) )
    y_train = train_data.dataset.targets[indices_train].numpy()

    x_test = test_data.dataset.data[indices_test].float().numpy() / 255.
    x_test = np.reshape( x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2] ) )
    y_test = test_data.dataset.targets[indices_test].numpy()

    # validation set
    x_val = x_train[10555:12665]
    y_val = np.array(y_train[10555:12665], dtype=int)
    x_train = x_train[0:10555]
    y_train = np.array(y_train[0:10555], dtype=int)
    # binarize
    if args.dynamic_binarization:
        args.input_type = 'binary'
        np.random.seed(777)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)
    else:
        args.input_type = 'cat'

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, args


def load_dataset(args, **kwargs):
    if args.dataset_name == 'cat':
        train_loader, val_loader, test_loader, args = load_dynamic_mnist_cat(args, **kwargs)
    else:
        raise Exception('Wrong name of the dataset!')

    return train_loader, val_loader, test_loader, args

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import QuantileTransformer
    class A():
        def __init__(self):
            pass
    args = A()
    args.dataset_name = "cat"
    args.batch_size = 64
    args.test_batch_size = 64
    tr, val, te, args = load_dataset(args)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, args.input_size[1], args.input_size[2]),
                                                     p, normalize=True, nrow=int(x.size(0) ** .5))

    for x in tr:
        x, y = x

        x = (x * 256 - .5).int()

        quintiles = np.percentile(x.numpy(), [0, 50])
        q = quintiles.searchsorted(x.numpy())

        qt = QuantileTransformer()
        xt = torch.tensor(qt.fit_transform(x.view(x.size(0), -1).numpy())).float().view(x.size())
        print(xt.min(), xt.max(), xt.size(), x.size())
        print(quintiles)
        for buckets in [2, 4, 8, 16]:
            xd = (xt * buckets).int().float() / buckets
            xr = torch.tensor(qt.inverse_transform(xd.numpy() * 0 + 1)).float()

            quintiles = np.percentile(x.numpy(), 100 * np.linspace(0, 1, buckets + 1)[:-1])
            print(quintiles)

            out = torch.zeros_like(xr)

            print(list(set(xt[0].numpy())))
            print(list(set(xd[0].numpy())))
            print(sorted(list(set(xr[0].numpy()))))

        for deq in [1, 2, 2, 4, 8, 16, 32, 64, 128]:
            xd = ((x // deq) * deq).float()
            print(xd)
            plot("output_img/hist{}.png".format(deq), xd)

        print(x.min(), x.max())