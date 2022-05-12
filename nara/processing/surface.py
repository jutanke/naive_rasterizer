import numpy as np
import numba as nb
from numpy.random import default_rng
from scipy.spatial import KDTree
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
        self.lookup = KDTree(data=T)
        self.rng = default_rng(12345)
        self.min_uv_val = min_uv_val
        self.max_uv_val = max_uv_val
        n_vertices = len(T)

        # calculate probability for sampling
        # some verticies are more densly packed than others
        # here we counter-act this by weighting sparse points
        # heigher than dense ones
        indices = self.lookup.query_ball_point(T, r=density_estimate_range)
        assert len(indices) == n_vertices
        unique_vertices = np.unique(F.reshape(F.shape[0] * 3))
        assert len(indices) == len(unique_vertices)
        assert (
            np.min(unique_vertices) == 0 and np.max(unique_vertices) == len(indices) - 1
        )

        cnt = np.array([len(ii) for ii in indices], dtype=np.float64)
        inverse_cnt = 1 / cnt
        self.sample_p = inverse_cnt / np.sum(inverse_cnt)
        self.sample_indices = np.array(range(len(cnt)), dtype=np.int64)

        # faces lookup to find all triangles
        triangles_lookup = [None] * n_vertices
        for face in F:
            for v in face:
                if triangles_lookup[v] is None:
                    triangles_lookup[v] = []
                triangles_lookup[v].append(face.view(np.ndarray))
        for i, faces in enumerate(triangles_lookup):
            if faces is None:
                raise ValueError(f"No faces with vertex {i}!")

        self.faces_count = np.zeros((n_vertices,), dtype=np.int64)
        for v in range(n_vertices):
            self.faces_count[v] = len(triangles_lookup[v])
        assert np.min(self.faces_count) > 0
        self.faces_list_dense = (
            np.ones((n_vertices, np.max(self.faces_count), 3), dtype=np.int64) * -1
        )
        for i, faces in enumerate(triangles_lookup):
            for j, face in enumerate(faces):
                self.faces_list_dense[i, j] = face

    def random_sample(self, n_samples: int, max_dist=0.0001):
        """
        randomly sample points on the UV map.
        The samples are weighted so that they are uniform over
        the map.
        """
        random_indices = self.rng.choice(
            self.sample_indices, size=n_samples, p=self.sample_p
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

    def query3d(self, uvs, V):
        """
        :param uvs: {n_points x 2}
        :param V: {n_vertices x 3}
        """
        distances, indices = self.lookup.query(uvs, k=1)
        best_faces, interpolation = native_find_best_face(
            self.T, uvs, self.faces_list_dense, self.faces_count, indices
        )
        assert np.min(best_faces) > -1
        interpolation = np.expand_dims(interpolation, axis=2)
        selected_pts = V[best_faces] * interpolation
        selected_pts = np.sum(selected_pts, axis=1)
        return selected_pts


@nb.njit(
    nb.types.Tuple((nb.int64[:, :], nb.float64[:, :]))(
        nb.float64[:, :], nb.float64[:, :], nb.int64[:, :, :], nb.int64[:], nb.int64[:]
    ),
    nogil=True,
)
def native_find_best_face(T, uvs, faces_list_dense, faces_count, closest_vertex_index):
    """
    :param T: {n_vertices x 2} UV Texture
    :param uvs: {n_points x 2}
    :param faces_list_dense: {n_vertices x n_faces x 3}
    :param faces_count: {n_vertices}
    :param closest_vertex_index: {n_points}
    """
    n_points = len(uvs)
    n_vertices = len(T)

    selected_face = np.ones((n_points, 3), dtype=np.int64)
    selected_face = selected_face * -1
    interpolation = np.empty((n_points, 3), dtype=np.float64)

    for i in range(n_points):
        v = closest_vertex_index[i]
        px = uvs[i, 0]
        py = uvs[i, 1]
        triangle_found = False
        for j in range(faces_count[v]):
            face = faces_list_dense[v, j]
            a = face[0]
            b = face[1]
            c = face[2]
            ax, ay = T[a]
            bx, by = T[b]
            cx, cy = T[c]
            if (
                edge_function(ax, ay, bx, by, px, py)
                and edge_function(bx, by, cx, cy, px, py)
                and edge_function(cx, cy, ax, ay, px, py)
            ):
                w1, w2, w3 = barycentric_interpolation(px, py, ax, ay, bx, by, cx, cy)
                selected_face[i, 0] = a
                selected_face[i, 1] = b
                selected_face[i, 2] = c
                interpolation[i, 0] = w1
                interpolation[i, 1] = w2
                interpolation[i, 2] = w3
                triangle_found = True
                continue

        if not triangle_found:
            # the point is outside any valid triangle:
            #
            #      (p)
            #
            #     (a)-----(b)
            #       \     /
            #        \   /
            #         (c)
            #
            # take the two clostest points in the closest
            # triangle instead ... in this case (a) and (b)
            best_distance = 9999999.99
            best_other = -1
            best_other2 = -1  # the 3d point on the triangle
            for j in range(faces_count[v]):
                face = faces_list_dense[v, j]
                a = face[0]
                b = face[1]
                c = face[2]

                if v == a:
                    o1 = b
                    o2 = c
                elif v == b:
                    o1 = a
                    o2 = c
                else:
                    o1 = a
                    o2 = b

                for other, other2 in [[o1, o2], [o2, o1]]:
                    ox, oy = T[other]
                    dist = (ox - px) ** 2 + (oy - py) ** 2
                    if best_distance > dist:
                        best_distance = dist
                        best_other = other
                        best_other2 = other2

            d1 = m.sqrt((px - T[v, 0]) ** 2 + (py - T[v, 1]) ** 2)
            d2 = m.sqrt(best_distance)

            selected_face[i, 0] = v
            selected_face[i, 1] = best_other
            selected_face[i, 2] = best_other2
            interpolation[i, 0] = d1 / (d1 + d2)
            interpolation[i, 1] = d2 / (d1 + d2)
            interpolation[i, 2] = 0

    return selected_face, interpolation
