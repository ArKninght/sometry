import numpy as np

def getProjectionVector(project_vect, DSD, DSO, pixle_volume, pixel_detector, volume_size, detector_size, total_angle, angles_num, offset_ratio):
	'''
	offset_ratio 代表平板的偏移率,取值在[-0.5,0.5],正代表向右偏移,负代表向左偏移,0表示不偏移
	'''
	# 获取当前投影的几何数据
	Ds = DSO        # 源半径
	Dd = DSD - DSO  # 探测器半径
	angle_interval = total_angle / angles_num / 180 * np.pi # 角度间隔
	# 计算缩放比例
	dSampleInterval = pixle_volume[0] / pixel_detector[0]
	dSliceInterval = pixle_volume[2] / pixel_detector[1]

	for i in range(angles_num):
		project_vect[i * 12] = Ds * np.cos(angle_interval * i) * (1 / pixel_detector[0]) / dSampleInterval
		project_vect[i * 12 + 1] = Ds * np.sin(angle_interval * i) * (1 / pixel_detector[0]) / dSampleInterval
		project_vect[i * 12 + 2] = 0
		project_vect[i * 12 + 6] = 0
		project_vect[i * 12 + 7] = 0
		project_vect[i * 12 + 8] = -1 / dSliceInterval
		project_vect[i * 12 + 9] = np.cos(angle_interval * i + np.pi / 2) / dSampleInterval
		project_vect[i * 12 + 10] = np.sin(angle_interval * i + np.pi / 2) / dSampleInterval
		project_vect[i * 12 + 11] = 0
		project_vect[i * 12 + 3] = -Dd * np.cos(angle_interval * i) * (1 / pixel_detector[0]) / dSampleInterval + offset_ratio * detector_size[0] * project_vect[i * 12 + 9]
		project_vect[i * 12 + 4] = -Dd * np.sin(angle_interval * i) * (1 / pixel_detector[0]) / dSampleInterval + offset_ratio * detector_size[0] * project_vect[i * 12 + 10]
		project_vect[i * 12 + 5] = 0

def astra_getProjectionVector(project_vect, DSD, DSO, pixle_volume, pixel_detector, volume_size, detector_size, total_angle, angles_num, offset_ratio):
	'''
	offset_ratio 代表平板的偏移率,取值在[-0.5,0.5],正代表向右偏移,负代表向左偏移,0表示不偏移
	'''
	# 获取当前投影的几何数据
	Ds = DSO        # 源半径
	Dd = DSD - DSO  # 探测器半径
	angle_interval = total_angle / angles_num / 180 * np.pi # 角度间隔
	# 计算缩放比例
	dSampleInterval = pixle_volume[0] / pixel_detector[0]
	dSliceInterval = pixle_volume[2] / pixel_detector[1]

	for i in range(angles_num):
		project_vect[i * 12] = Ds * np.cos(angle_interval * i) * (1 / pixel_detector[0]) / dSampleInterval
		project_vect[i * 12 + 1] = Ds * np.sin(angle_interval * i) * (1 / pixel_detector[0]) / dSampleInterval
		project_vect[i * 12 + 2] = 0
		project_vect[i * 12 + 6] = np.cos(angle_interval * i + np.pi / 2) / dSampleInterval
		project_vect[i * 12 + 7] = np.sin(angle_interval * i + np.pi / 2) / dSampleInterval
		project_vect[i * 12 + 8] = 0
		project_vect[i * 12 + 9] = 0
		project_vect[i * 12 + 10] = 0
		project_vect[i * 12 + 11] = -1 / dSliceInterval
		project_vect[i * 12 + 3] = -Dd * np.cos(angle_interval * i) * (1 / pixel_detector[0]) / dSampleInterval + offset_ratio * detector_size[0] * project_vect[i * 12 + 6]
		project_vect[i * 12 + 4] = -Dd * np.sin(angle_interval * i) * (1 / pixel_detector[0]) / dSampleInterval + offset_ratio * detector_size[0] * project_vect[i * 12 + 7]
		project_vect[i * 12 + 5] = 0