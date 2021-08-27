import matplotlib.pylab as plt
import numpy as np


class Plot:
    def __init__(self, w, h):
        super().__init__()
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim([0, w])
        self.ax.set_ylim([h, 0])
        self.ax.axis("off")

    def imshow(self, im):
        self.ax.imshow(im)
        # h, w, _ = im.shape
        # self.ax.set_xlim([0, w])
        # self.ax.set_ylim([h, 0])

    def scatter2d(self, pts2d, color=None, weights=None, alpha=1.0, s=10):
        if weights is not None:
            assert len(weights) == len(pts2d)
            weights = weights ** 4
            s = np.array([s] * len(weights)) * weights

        if color == None:
            self.ax.scatter(pts2d[:, 0], pts2d[:, 1], s=s, alpha=alpha)
        else:
            self.ax.scatter(pts2d[:, 0], pts2d[:, 1], color=color, s=s, alpha=alpha)

    def save(self, fname):
        plt.tight_layout()
        self.fig.savefig(fname)
        plt.close()