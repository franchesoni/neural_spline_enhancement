import torch
import numpy as np
import matplotlib.pyplot as plt


class BiancoSpline:
    def __init__(self, n):
        self.n, self.h, self.x0 = n, 1.0 / (n-1.0), 0
        mat = 4 * np.eye(n - 2)
        np.fill_diagonal(mat[1:, :-1], 1)
        np.fill_diagonal(mat[:-1, 1:], 1)
        A = 6 * np.linalg.inv(mat) / (self.h ** 2)
        z = np.zeros(n - 2)
        A = np.vstack([z, A, z])

        B = np.zeros([n - 2, n])
        np.fill_diagonal(B, 1)
        np.fill_diagonal(B[:, 1:], -2)
        np.fill_diagonal(B[:, 2:], 1)
        self.matrix = np.dot(A, B)
        self.matrix = torch.from_numpy(self.matrix).double()
        # self.matrix = self.matrix.cuda()

    def predict(self, raw, params):
        """
        raw: HxWx3
        params: dict(ys: 3N)
        """
        ys = params['ys'].reshape(3, -1)  # 3xN
        out = torch.empty_like(raw)
        for ch in range(3):
            cur_ch = raw[:, :, ch].clone()
            cur_ys = ys[ch, :].clone()
            identity = torch.arange(0,cur_ys.size(0)).double()/(cur_ys.size(0)-1)
            cur_coeffs = self.fit_coeffs(cur_ys+identity, 1 / (ys.shape[1]-1))
            out[:,:,ch] = self.apply(cur_coeffs, cur_ch.view(-1)).view(cur_ch.size())
            if ch == 2:
                with torch.no_grad():
                    xs =torch.arange(0,1,1.0/255.).double()
                    b = self.apply(cur_coeffs, xs).numpy()
                plt.plot(xs, b)
                plt.savefig('tests/oracle_Bianco_blue_spline.png')
        return out

    def fit_coeffs(self, ys, h):
        M = torch.mm(self.matrix, ys.view(-1,1)).squeeze()
        a = (M[1:] - M[:-1]) / (6 * h)
        b = M[:-1] / 2
        c = (ys[1:] - ys[:-1]) / h - (M[1:] + 2 * M[:-1]) * (h / 6)
        d = ys[:-1]
        coeffs = torch.stack([a,b,c,d], dim=0)
        return coeffs

    def apply(self, coeffs, x):
        """ interpolate new data using coefficients
        """
        xi = torch.clamp((x - self.x0) / self.h, 0, self.n-2)
        xi = torch.floor(xi)
        xf = x - self.x0 - xi*self.h
        ones = torch.ones(xf.size())
        ex = torch.stack([xf ** 3, xf ** 2, xf, ones], dim=0)
        #y = np.dot(coeffs.transpose(0,1), ex)
        y = torch.mm(coeffs.transpose(0,1), ex)
        # create constant mat
        sel_mat = torch.zeros(y.size(0),xi.size(0))
        rng = torch.arange(0,xi.size(0))
        sel_mat[xi.data.long(),rng.long()]=1
        # multiply to get the right coeffs
        res = y*sel_mat
        res = res.sum(0)
        # return
        return res



class TPS_RGB_ORDER_2:
    '''
    Thin plate spline of order 2
    params:
		xs_control: number of control points x 3 [for each output color channel]
		alphas: column vector, size of the number of control points [for each output color channel]
    '''
    @staticmethod
    def build_k(xs_control, xs_eval):
        # "classic" TPS energy (m=2), null space is just the affine functions span{1, r, g, b} 
        # xs_control : Nx3
        # xs_eval : Mx3
        # returns Mx(N+4) matrix
        M = xs_eval.shape[0]
        d = torch.linalg.norm(
            xs_eval[:, None] - xs_control[None], axis=2  # M x 1 x 3  # 1 x N x 3
        )  # M x N x 3
        return torch.hstack((d, torch.ones((M,1)), xs_control))

    @staticmethod
    def predict(raw, params):

        '''xs: N x 3 (input color dim) x 3 (output color dim) [control points]
        alphas: (N+4) x 3 (output color dim) [spline weights]
        raw: HxWx3
        '''
        fimg = raw.reshape(-1, 3)  # Mx3, flattened image
        
        K0 = TPS_RGB_ORDER_2.build_k(fimg[:, 0:1], params['xs'][:,0,:])
        out0 = K0 @ params["alphas"][:, 0:1]
        
        K1 = TPS_RGB_ORDER_2.build_k(fimg[:, 1:2], params['xs'][:,1,:]) if 1 < params["xs"].shape[1] else K0
        out1 = K1 @ params["alphas"][:, 1:2]
        
        K2 = TPS_RGB_ORDER_2.build_k(fimg[:, 2:3], params['xs'][:,2,:]) if 1 < params["xs"].shape[1] else K0
        out2 = K2 @ params["alphas"][:, 2:3]

        out = torch.hstack([out0, out1, out2])  # Mx3
        return out.reshape(raw.shape)  # HxWx3


class GaussianSpline:
    # params are xs alphas
    @staticmethod
    def predict(raw, params):
        """
        raw: HxWx3, M=HxW
        params: dict(alphas: Nx3,
                  xs: Nx3x3 or Nx1x3 (control points, channels-in, channels-out), i.e. provide the same xs for all channels (x1) or not (x3)
                  sigmas: 1x3)
        """
        fimg = raw.reshape(-1, 3)  # Mx3, flattened image

        xs = params["xs"][:, 0, :]  # Nx3
        K = GaussianSpline.build_k(
            fimg[:, 0:1], xs, sigma=params["sigmas"][0, 0:1]
        )
        out0 = K @ params["alphas"][:, 0:1]  # MxN x Nx1 = Mx1

        xs = params["xs"][:, 1, :] if 1 < params["xs"].shape[1] else xs  # Nx3
        K = GaussianSpline.build_k(
            fimg[:, 1:2], xs, sigma=params["sigmas"][0, 1:2]
        )
        out1 = K @ params["alphas"][:, 1:2]  # MxN x Nx1 = Mx1

        xs = params["xs"][:, 2, :] if 1 < params["xs"].shape[1] else xs  # Nx3
        K = GaussianSpline.build_k(
            fimg[:, 2:3], xs, sigma=params["sigmas"][0, 2:3]
        )
        out2 = K @ params["alphas"][:, 2:3]  # MxN x Nx1 = Mx1

        out = torch.hstack([out0, out1, out2])  # Mx3
        return out.reshape(raw.shape)  # HxWx3

    @staticmethod
    def build_k(xs, xs_control, sigma=50):
        """
        xs: Mx3
        xs_control: Nx3

        out: MxN
        """
        d = torch.linalg.norm(
            xs[:, None] - xs_control[None], axis=2  # M x 1 x 3  # 1 x N x 3
        )  # M x N x 3
        return torch.exp(-d**2 / (2 * sigma**2))

    @staticmethod
    def fit_alphas(xs, ys, sigma=1):
        """
        xs: Nx3
        ys: Nx1
        """
        N = len(xs)
        K = GaussianSpline.build_k(xs, ys, sigma)
        return torch.linalg.inv(K) @ ys


class GaussianSplineSlow:
    # params are xs ys
    pass

if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    S = 101
    raw_path = "tests/raw_000014.jpg"
    enh_path = "tests/enh_000014.jpg"
    raw, enh = torch.Tensor(
        np.array(Image.open(raw_path))[:S, :S]
    ), torch.Tensor(np.array(Image.open(enh_path))[:S, :S])

    spline = GaussianSpline()
    N = 10
    params1 = dict(
        alphas=torch.randn(N, 3),
        xs=torch.randn(N, 3, 3),
        sigmas=torch.randn(1, 3),
    )
    out1 = spline.predict(raw, params1)
    params2 = dict(
        alphas=torch.randn(N, 3),
        xs=torch.randn(N, 1, 3),
        sigmas=torch.randn(1, 3),
    )
    out2 = spline.predict(raw, params2)

    alphas = spline.fit_alphas(torch.randn(101, 3), torch.randn(101, 1))
    assert out1.shape == out2.shape == (S, S, 3)
    assert alphas.shape == (S, 1)
