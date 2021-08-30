import numpy as np
import cv2


class Camera:
    def __init__(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        K: np.ndarray,
        dist_coeff: np.ndarray,
        w: int,
        h: int,
    ):
        """"""
        self.rvec = rvec
        self.tvec = tvec
        self.K = K
        self.dist_coeff = dist_coeff
        self.w = w
        self.h = h

        R = cv2.Rodrigues(self.rvec)[0]
        tvec = self.tvec
        self.pos = np.squeeze(-np.transpose(R) @ tvec)

    def project_points(self, pts3d):
        """
        :param pts3d: {n_points x 3}
        """
        pts2d, _ = cv2.projectPoints(
            pts3d,
            rvec=self.rvec,
            tvec=self.tvec,
            cameraMatrix=self.K,
            distCoeffs=self.dist_coeff,
        )
        if len(pts2d.shape) == 3:
            pts2d = pts2d[:, 0, :]

        return pts2d
