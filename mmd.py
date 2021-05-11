import torch



def assert_shape(x, s):
    assert x.size() == s


def avg_hamming(x, y):
    diffs = (x[None, :] != y[:, None, :]).float().mean(-1)
    return diffs


def exp_avg_hamming(x, y):
    diffs = avg_hamming(x, y)
    return (-diffs).exp()


def scaled_exp_avg_hamming(x, y, s):
    diffs = avg_hamming(x, y) * s
    return (-diffs).exp()


class MMD(object):
    """
    Quadratic-time maximum mean discrepancy (MMD) test.
    Use the unbiased U-statistic.
    """
    def __init__(self, kernel_fun, use_ustat=False, quantile=.95, n_boot=5000):
        """
        Args:
            kernel: function, kernel function.
            use_ustat: boolean, whether to compute U-statistic or V-statistic.
            quantile: float, 1. - significance level
            n_boot: int, number of bootstraps for computing threshold
        """
        assert callable(kernel_fun)

        self.kernel = kernel_fun
        self.quantile = quantile  # 1 - alpha
        self.n_boot = n_boot  # Number of bootstraps for computing threshold
        self.use_ustat = use_ustat

    def compute_gram(self, x, y):
        """
        Compute Gram matrices:
            K: array((m+n, m+n))
            kxx: array((m, m))
            kyy: array((n, n))
            kxy: array((m, n))
        """
        (m, d1) = x.shape
        (n, d2) = y.shape
        assert d1 == d2

        xy = torch.cat([x, y], 0) #np.vstack([x, y])
        K = self.kernel(xy, xy)  # kxyxy
        assert_shape(K, (m+n, m+n))
        #assert is_psd(K)  # TODO: Remove check

        kxx = K[:m, :m]
        assert_shape(kxx, (m, m))
        # assert is_psd(kxx)
        #assert is_symmetric(kxx)

        kyy = K[m:, m:]
        assert_shape(kyy, (n, n))
        # assert is_psd(kyy)
        #assert is_symmetric(kyy)

        kxy = K[:m, m:]
        assert_shape(kxy, (m, n))

        return K, kxx, kyy, kxy

    def compute_statistic(self, kxx, kyy, kxy):
        """
        Compute MMD test statistic.
        """
        m = kxx.size(0)
        n = kyy.size(0)
        assert_shape(kxx, (m, m))
        assert_shape(kyy, (n, n))
        assert_shape(kxy, (m, n))

        if self.use_ustat:  # Compute U-statistics estimate
            term_xx = (kxx.sum() - torch.diag(kxx).sum()) / (m*(m-1))
            term_yy = (kyy.sum() - torch.diag(kyy).sum()) / (n*(n-1))
            term_xy = kxy.sum() / (m*n)

        else:  # Compute V-statistics estimate
            term_xx = kxx.sum() / (m**2)
            term_yy = kyy.sum() / (n**2)
            term_xy = kxy.sum() / (m*n)

        res = term_xx + term_yy - 2*term_xy

        return res

    def compute_threshold(self, m, n, K):
        1/0

    def compute_pval(self, stat, boot_stats):
        1/0

    def compute_mmd(self, x, y):
        K, kxx, kyy, kxy = self.compute_gram(x, y)
        stat = self.compute_statistic(kxx, kyy, kxy)
        return stat

    def perform_test(self, x, y):
        1/0