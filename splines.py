import torch


class GaussianSpline:
    # params are xs alphas
    @staticmethod
    def predict(raw, params):
        """
        raw: HxWx3, M=HxW
        params: dict(alphas: Nx3,
                  xs: Nx3x3 or Nx1x3 (control points, channels-in, channels-out),
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
        return torch.exp(-d / (2 * sigma**2))

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
