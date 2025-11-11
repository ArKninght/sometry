# 尝试使用tigre中的fista算法进行一定的重建
import tigre
import numpy as np
from tigre.utilities import sample_loader
from tigre.utilities import CTnoise
import tigre.algorithms as algs
from matplotlib import pyplot as plt
import os

#%% Geometry
geo = tigre.geometry()
# VARIABLE                                   DESCRIPTION                    UNITS
# -------------------------------------------------------------------------------------
# Distances
geo.DSD = 1500  # Distance Source Detector      (mm)
geo.DSO = 1000  # Distance Source Origin        (mm)
# Detector parameters
geo.nDetector = np.array([256, 256])  # number of pixels              (px)
geo.dDetector = np.array([1.0, 1.0])  # size of each pixel            (mm)
geo.sDetector = geo.nDetector * geo.dDetector  # total size of the detector    (mm)
# Image parameters
geo.nVoxel = np.array([64, 256, 256])  # number of voxels              (vx)
geo.dVoxel = np.array([1.0, 0.625, 0.625])  # size of each voxel            (mm)
geo.sVoxel = geo.dVoxel * geo.nVoxel  # total size of the image       (mm)
# Offsets
geo.offOrigin = np.array([0, 0, 0])  # Offset of image from origin   (mm)
geo.offDetector = np.array([0, 0])  # Offset of Detector            (mm)
# Auxiliary
geo.accuracy = 0.5  # Variable to define accuracy of 'interpolated' projection
# Optional Parameters
geo.COR = 0  # y direction displacement for
geo.rotDetector = np.array([0, 0, 0])  # Rotation of the detector, by
geo.mode = "cone"  # Or 'parallel'. Geometry type.

#%% Load data and generate projections
# define angles
angles = np.linspace(0, 2 * np.pi, 360)
# Load raw data
head = np.fromfile(r'/mnt/d/CT数据/仿真数据/lung/train_raw/000057_00_256x256x64.raw', dtype='float32').reshape(geo.nVoxel[0], geo.nVoxel[1], geo.nVoxel[2])
# head = sample_loader.load_head_phantom(geo.nVoxel)
# generate projections
projections = tigre.Ax(head, geo, angles)

# create directory for saving individual projection frames
output_dir = 'projection_frames'
os.makedirs(output_dir, exist_ok=True)

# save each projection frame as a separate .raw file
for i in range(projections.shape[0]):
    frame_filename = os.path.join(output_dir, f'projection_frame_{i:04d}.raw')
    projections[i].tofile(frame_filename)

print(f"已保存 {projections.shape[0]} 个投影帧到文件夹: {output_dir}")
print(f"每帧投影形状: {projections[0].shape}")
print(f"角度范围: 0° 到 {angles[-1]*180/np.pi:.1f}°")

# add noise
# noise_projections = CTnoise.add(projections, Poisson=1e5, Gaussian=np.array([0, 10]))


## FISTA is a quadratically converging algorithm.

# 'hyper': This parameter should approximate the largest
#          eigenvalue in the A matrix in the equation Ax-b and Atb.
#          Empirical tests show that for, the headphantom object:
#
#               geo.nVoxel = [64,64,64]'    ,      hyper (approx=) 2.e8
#               geo.nVoxel = [512,512,512]' ,      hyper (approx=) 2.e4
#          Default: 2.e8
# 'tviter':  Number of iterations of Im3ddenoise to use. Default: 20
# 'lambda':  Multiplier for the tvlambda used, which is proportional to
#            L (hyper). Default: 0.1
# 'verbose': get feedback or not. Default: 1
# --------------------------------------------------------------------------

# imgFISTA_default = algs.fista(projections, geo, angles, 100, hyper=2.0e5)

# # ## Adding a different hyper parameter
# # imgFISTA_hyper = algs.fista(noise_projections, geo, angles, 100, hyper=2.0e6)

# # ## Recon more significant tv parameters
# # imgFISTA_hightv = algs.fista(
# #     noise_projections, geo, angles, 100, hyper=2.0e6, tviter=100, tvlambda=20
# # )
# imgFISTA_default.tofile("fistaRecon.raw")

# ## Plot results
# tigre.plotimg(
#     np.concatenate([imgFISTA_default], axis=1),
#     dim="z",
#     clims=[0, 1],
# )
