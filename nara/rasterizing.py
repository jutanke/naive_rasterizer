import numpy as np
import numba as nb
import numpy.linalg as la
import math as m
import cv2


INFINITY = 999999.0


@nb.njit(
    nb.types.UniTuple(nb.float64, 3)(
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
    ),
    nogil=True,
)
def barycentric_interpolation(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
):
    """"""
    Px = px
    Py = py
    Xv1 = ax
    Xv2 = bx
    Xv3 = cx
    Yv1 = ay
    Yv2 = by
    Yv3 = cy

    Wv1 = ((Yv2 - Yv3) * (Px - Xv3) + (Xv3 - Xv2) * (Py - Yv3)) / (
        (Yv2 - Yv3) * (Xv1 - Xv3) + (Xv3 - Xv2) * (Yv1 - Yv3)
    )
    Wv2 = ((Yv3 - Yv1) * (Px - Xv3) + (Xv1 - Xv3) * (Py - Yv3)) / (
        (Yv2 - Yv3) * (Xv1 - Xv3) + (Xv3 - Xv2) * (Yv1 - Yv3)
    )
    Wv3 = 1 - Wv1 - Wv2
    return Wv1, Wv2, Wv3


@nb.njit(
    nb.boolean(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
    nogil=True,
)
def edge_function(ax, ay, bx, by, cx, cy):
    return ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) >= 0


@nb.njit(nb.float64[:, :](nb.float64[:, :], nb.int64[:, :]), nogil=True)
def native_calculate_normals(V, F):
    """
    :param V: {n_vertices x 3} Vertices
    :param F: {n_faces x 3} Faces
    """
    n_vertices = len(V)
    n_faces = len(F)
    normals = np.zeros((n_vertices, 3), dtype=np.float64)

    # step1: count how many faces touch the vertices
    faces_hits = np.zeros((n_vertices), dtype=np.float64)

    for i in range(n_faces):
        a = F[i, 0]
        b = F[i, 1]
        c = F[i, 2]
        faces_hits[a] = faces_hits[a] + 1.0
        faces_hits[b] = faces_hits[b] + 1.0
        faces_hits[c] = faces_hits[c] + 1.0

    normals_scale = 1.0 / faces_hits

    # step2: calculate normals per face and scale-add to vertices
    for i in range(n_faces):
        a = F[i, 0]
        b = F[i, 1]
        c = F[i, 2]

        ab = V[b] - V[a]
        ab = ab / la.norm(ab)
        ac = V[c] - V[a]
        ac = ac / la.norm(ac)
        n = np.cross(ab, ac)

        na = n * normals_scale[a]
        nb = n * normals_scale[b]
        nc = n * normals_scale[c]

        normals[a] = normals[a] + na
        normals[b] = normals[b] + nb
        normals[c] = normals[c] + nc

    return normals


@nb.njit(
    nb.types.Tuple((nb.float64[:, :], nb.float64[:, :, :], nb.float64[:, :, :]))(
        nb.float64[:, :],
        nb.int64[:, :],
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:, :],
        nb.int64,
        nb.int64,
        nb.float64,
        nb.boolean,
    ),
    nogil=True,
)
def native_rasterizing(
    V, F, T, D, N, w: int, h: int, INFINITY: float, calculate_normals: bool
):
    """
    :param V: {n_vertices x 2} Vertices
    :param F: {n_faces x 3} Faces
    :param T: {n_faces x 2} UV Texture
    :param D: {n_vertices} distance Camera-Vertices
    :param N: {n_vertices x 3} Normals
    :param w: image width
    :param h: image height
    """

    uv_image = np.zeros((h, w, 2), dtype=np.float64)
    normal_image = np.zeros((h, w, 3), dtype=np.float64)
    zbuffer = np.ones((h, w), dtype=np.float64) * INFINITY

    n_triangles = len(F)

    for i in range(n_triangles):
        ax, ay = V[F[i, 0]]
        bx, by = V[F[i, 1]]
        cx, cy = V[F[i, 2]]

        az = D[F[i, 0]]
        bz = D[F[i, 1]]
        cz = D[F[i, 2]]

        auv = T[F[i, 0]]
        buv = T[F[i, 1]]
        cuv = T[F[i, 2]]

        na = N[F[i, 0]]
        nb = N[F[i, 1]]
        nc = N[F[i, 2]]

        bb_left = int(m.floor(max(0.0, min(w - 1, min(ax, min(bx, cx))))))
        bb_top = int(m.floor(max(0.0, min(h - 1, min(ay, min(by, cy))))))

        bb_right = int(m.ceil(max(0.0, min(w - 1, max(ax, max(bx, cx))))))
        bb_bottom = int(m.ceil(max(0.0, min(h - 1, max(ay, max(by, cy))))))

        for x in range(bb_left, bb_right):
            for y in range(bb_top, bb_bottom):
                px = float(x)
                py = float(y)
                if (
                    edge_function(ax, ay, bx, by, px, py)
                    and edge_function(bx, by, cx, cy, px, py)
                    and edge_function(cx, cy, ax, ay, px, py)
                ):
                    w1, w2, w3 = barycentric_interpolation(
                        px, py, ax, ay, bx, by, cx, cy
                    )
                    d = w1 * az + w2 * bz + w3 * cz
                    if d < zbuffer[y, x]:
                        zbuffer[y, x] = d
                        uv_image[y, x] = w1 * auv + w2 * buv + w3 * cuv

                        if calculate_normals:
                            n = w1 * na + w2 * nb + w3 * nc
                            normal_image[y, x] = n

    return zbuffer, uv_image, normal_image


def rasterizing(
    V: np.ndarray, F: np.ndarray, T: np.ndarray, cam, calculate_normals=True
):
    """
    :params V: {n_vertices x 3}
    :param F: {n_faces x 3}
    :param T: {n_vertices x 2} texture indices
    :param cam: {Camera}
    """
    global INFINITY

    normals_per_vertex = native_calculate_normals(V, F)
    dist = np.expand_dims(la.norm(normals_per_vertex, axis=1), axis=1)
    normals_per_vertex /= dist

    R = cv2.Rodrigues(cam.rvec)[0]
    normals_per_vertex = normals_per_vertex @ R

    V2d = cam.project_points(V)
    cam_pos = np.expand_dims(cam.pos, axis=0)
    d = la.norm(cam_pos - V, axis=1)

    zbuffer, uv_image, normal_image = native_rasterizing(
        V2d, F, T, d, normals_per_vertex, cam.w, cam.h, INFINITY, calculate_normals
    )
    zbuffer[zbuffer > INFINITY - 0.00001] = 0

    if calculate_normals:
        return zbuffer, uv_image, normal_image
    else:
        return zbuffer, uv_image
