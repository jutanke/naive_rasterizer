import sys

sys.path.insert(0, ".")

# from trimesh.exchange.obj import load_obj
import trimesh
import cv2
import numpy as np
from nara.camera import Camera
from nara.vis import Plot

from nara.rasterizing import rasterizing

camera = 10
frame = 250
person = "377"

# extri_fpath = f"data/easymocap/extri_{camera}.yml"
# intri_fpath = f"data/easymocap/intri_{camera}.yml"
# mesh_fpath = "data/easymocap/000074.obj"
# extri_fpath = "data/extri.yml"
# intri_fpath = "data/intri.yml"
# mesh_fpath = "data/000000_2.obj"
uv_fpath = "data/uv_table.npy"

extri_fpath = f"/home/group-cvg/datasets/easymocap/cameras/{person}/extri.yml"
intri_fpath = f"/home/group-cvg/datasets/easymocap/cameras/{person}/intri.yml"

mesh_fpath = f"/home/group-cvg/datasets/easymocap/meshes/{person}/%06d.obj" % frame


# image_fpath = f"data/AllViewImage_000000/{camera}_000000.jpg"
image_fpath = (
    f"/home/group-cvg/datasets/easymocap/images/{person}/{camera}/%06d.jpg" % frame
)


intri_param = cv2.FileStorage(intri_fpath, flags=0)
extri_param = cv2.FileStorage(extri_fpath, flags=0)

im = cv2.cvtColor(cv2.imread(image_fpath), cv2.COLOR_BGR2RGB)

camera = "%02d" % camera

w = 1024
h = 1024
K = intri_param.getNode(f"K_{camera}").mat()
dist = intri_param.getNode(f"dist_{camera}").mat()
rvec = extri_param.getNode(f"R_{camera}").mat()  # 3x1 np.array
tvec = extri_param.getNode(f"T_{camera}").mat()  # 3x1 np.array

plot = Plot(w, h)
plot.imshow(im)


cam = Camera(rvec, tvec, K, dist, w, h)

mesh = trimesh.load(mesh_fpath)
V = mesh.vertices
F = mesh.faces
T = np.load(uv_fpath)


zbuffer, uv_image, normal_image = rasterizing(V, F, T, cam, calculate_normals=True)


print("normal_image", normal_image.shape, np.min(normal_image), np.max(normal_image))

normal_image = (normal_image + 1) / 2


plot.ax.imshow(im)
plot.ax.imshow(normal_image, alpha=0.5)

plot.save(f"output/eazmopca{person}_{camera}.png")
