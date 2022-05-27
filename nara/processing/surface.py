import numpy as np
import numba as nb
import numpy.linalg as la
from numpy.random import default_rng
from nara.rasterizing import edge_function, barycentric_interpolation
import math as m


class TextureProjector3d:
    def __init__(
        self,
        T: np.ndarray,
        F: np.ndarray,
        density_estimate_range=0.02,
        min_uv_val=0.0,
        max_uv_val=1.0,
    ):
        """
        :param T: {n_vertices x 2} UV Texture
        :param F: {n_faces x 3} Faces
        :param density_estimate_range: {float} defines the range on
            the UV map which defines the region of a point
        """
        super().__init__()
        self.T = T
        self.F = F
        self.n_vertices = len(T)
        self.rng = default_rng(12345)
        n_vertices = len(T)

        self.faces_indices = list(range(len(F)))

    def random_sample(self, n_samples: int, V: np.ndarray):
        """
        :param V: {n_vertices x 3}
        """
        F = self.F
        T = self.T
        A = V[F[:, 0]]
        B = V[F[:, 1]]
        C = V[F[:, 2]]
        AB = A - B
        AC = A - C
        S = la.norm(np.cross(AB, AC), axis=1) / 2
        S = S / np.sum(S)

        faces_indices = self.rng.choice(self.faces_indices, size=n_samples, p=S)

        r1 = self.rng.uniform(size=(n_samples, 1))
        r2 = self.rng.uniform(size=(n_samples, 1))
        r1_sqrt = np.sqrt(r1)

        # random sampling on triangle as (https://www.cs.princeton.edu/~funk/tog02.pdf)
        # Random point P:
        # P = (1-sqrt(r1))A + (sqrt(r1)(1-r2))B + (r2 sqrt(r1))C
        # where r1,r2 ~ U[0,1]
        A = V[F[faces_indices, 0]]
        B = V[F[faces_indices, 1]]
        C = V[F[faces_indices, 2]]
        P = (1 - r1_sqrt) * A + (r1_sqrt * (1 - r2)) * B + (r2 * r1_sqrt) * C

        A_uv = T[F[faces_indices, 0]]
        B_uv = T[F[faces_indices, 1]]
        C_uv = T[F[faces_indices, 2]]
        P_uv = (
            (1 - r1_sqrt) * A_uv + (r1_sqrt * (1 - r2)) * B_uv + (r2 * r1_sqrt) * C_uv
        )

        return P, P_uv
