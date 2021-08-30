# (stupidly) naive Rasterisation
Super naive rasterizer for OpenCV cameras:
* zbuffer
* uv image
* normal image 

## install
Install the following libraries:
* numba
* numpy
* OpenCV
```
pip install git+https://github.com/jutanke/naive_rasterizer.git
```

## minimal sample
```python
from nara.rasterizing import rasterizing
from nara.camera import Camera

# load OpenCV camera
rvec = .. # 3x1
tvec = .. # 3x1
K = .. # 3x3
dist = .. # 5x1
w = .. # int
h = .. # int
cam = Camera(rvec, tvec, K, dist, w, h)

# load mesh (not part of library)
import trimesh
mesh = trimesh.load("/path/to/mesh.obj")
V = mesh.vertices
F = mesh.faces
T = np.load("/path/to/textures.npy")

zbuffer, uv_image, normal_image = rasterizing(V, F, T, cam, calculate_normals=True)

# uv_image --> wxhx2
# normal_image --> wxhx3
```
