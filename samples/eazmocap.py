import sys

sys.path.insert(0, ".")

# from trimesh.exchange.obj import load_obj
import trimesh
import cv2
from nara.camera import Camera
from nara.vis import Plot


extri_fpath = "data/extri.yml"
intri_fpath = "data/intri.yml"
mesh_fpath = "data/000000_2.obj"

camera = "1"

image_fpath = f"data/AllViewImage_000000/{camera}_000000.jpg"

intri_param = cv2.FileStorage(intri_fpath, flags=0)
extri_param = cv2.FileStorage(extri_fpath, flags=0)

im = cv2.cvtColor(cv2.imread(image_fpath), cv2.COLOR_BGR2RGB)

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

# V[:, [0, 1, 2]] = V[:, [0, 2, 1]]


v2d = cam.project_points(V)

plot.scatter2d(v2d)

plot.save("output/eazmopca.png")
