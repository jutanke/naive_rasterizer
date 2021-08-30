import numpy as np
import numba as nb
import numpy.linalg as la
import math as m


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


@nb.njit(
    nb.float64[:, :](
        nb.float64[:, :], nb.int64[:, :], nb.float64[:], nb.int64, nb.int64, nb.float64
    ),
    nogil=True,
)
def native_zbuffer(V, F, D, w: int, h: int, INFINITY: float):
    """
    :param V: {n_vertices x 2} Vertices
    :param F: {n_faces x 3} Faces
    :param D: {n_vertices} distance Camera-Vertices
    :param w: image width
    :param h: image height
    """
    zbuffer = np.ones((h, w), dtype=np.float64) * INFINITY

    n_triangles = len(F)

    for i in range(n_triangles):
        ax, ay = V[F[i, 0]]
        bx, by = V[F[i, 1]]
        cx, cy = V[F[i, 2]]

        az = D[F[i, 0]]
        bz = D[F[i, 1]]
        cz = D[F[i, 2]]

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

    return zbuffer


def zbuffer(V: np.ndarray, F: np.ndarray, cam):
    """
    :params V: {n_vertices x 3}
    :param F: {n_faces x 3}
    """

    V2d = cam.project_points(V)
    cam_pos = np.expand_dims(cam.pos, axis=0)

    d = la.norm(cam_pos - V, axis=1)

    INFINITY = 999999.0

    zbuf = native_zbuffer(V2d, F, d, cam.w, cam.h, INFINITY)
    zbuf[zbuf > INFINITY - 0.00001] = 0

    return zbuf