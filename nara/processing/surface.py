import numpy as np
import numba as nb
from numpy.random import default_rng
from scipy.spatial import KDTree
from nara.rasterizing import edge_function


class TextureProjector3d:
    def __init__(
        self,
        T: np.ndarray,
        density_estimate_range=0.02,
        min_uv_val=0.0,
        max_uv_val=1.0,
    ):
        """
        :param T: {n_vertices x 2} UV Texture
        :param density_estimate_range: {float} defines the range on
            the UV map which defines the region of a point
        """
        super().__init__()
        self.T = T
        self.n_vertices = len(T)
        self.lookup = KDTree(data=T)
        self.rng = default_rng(12345)
        self.min_uv_val = min_uv_val
        self.max_uv_val = max_uv_val

        # calculate probability for sampling
        # some verticies are more densly packed than others
        # here we counter-act this by weighting sparse points
        # heigher than dense ones
        indices = self.lookup.query_ball_point(T, r=density_estimate_range)
        assert len(indices) == len(T)
        cnt = np.array([len(ii) for ii in indices], dtype=np.float64)
        self.inverse_cnt = 1 / cnt
        self.mask = np.zeros((self.n_vertices,), dtype=np.float64)
        self.sample_indices = np.array(range(len(cnt)), dtype=np.int64)

    def random_sample(self, n_samples: int, F: np.ndarray, max_dist=0.0001):
        """
        :param F: {n_faces x 3} Faces
        randomly sample points on the UV map.
        The samples are weighted so that they are uniform over
        the map.
        """
        # some vertices might not be present: mask them!
        sampling_mask = self.mask.copy()
        indices = np.unique(F.reshape(F.shape[0] * F.shape[1]))
        sampling_mask[indices] = 1.0
        sample_p = self.inverse_cnt * sampling_mask
        sample_p = sample_p / np.sum(sample_p)

        random_indices = self.rng.choice(
            self.sample_indices, size=n_samples, p=sample_p
        )
        mu = np.zeros((2,))
        cov = np.eye(2) * max_dist
        random_pts_local = self.rng.multivariate_normal(
            mean=mu, cov=cov, size=n_samples
        )
        selected_base_points = self.T[random_indices].copy() + random_pts_local
        # clamp
        selected_base_points[:, 0] = np.maximum(
            selected_base_points[:, 0], self.min_uv_val
        )
        selected_base_points[:, 0] = np.minimum(
            selected_base_points[:, 0], self.max_uv_val
        )
        selected_base_points[:, 1] = np.maximum(
            selected_base_points[:, 1], self.min_uv_val
        )
        selected_base_points[:, 1] = np.minimum(
            selected_base_points[:, 1], self.max_uv_val
        )
        return selected_base_points

    def query3d(self, uvs):
        """
        :param uvs: {n_points x 2}
        """
        raise NotImplementedError("Not yet")
        distances, indices = self.lookup.query(uvs, k=1)

        # faces lookup to find all triangles
        triangles_lookup = [None] * len(T)
        for face in F:
            for v in face:
                if triangles_lookup[v] is None:
                    triangles_lookup[v] = []
                triangles_lookup[v].append(face)

        # for i, faces in enumerate(triangles_lookup):
        #     if faces is None:
        #         raise ValueError(f"No faces with vertex {i}!")
        # print(triangles_lookup)

        for i, idx in enumerate(indices):

            orig = self.T[idx]
            print("~~>", orig, uvs[i])
