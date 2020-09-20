import numpy as np
from joblib import Parallel, delayed

def remove_unwanted_keypoints(array_4D_data, unwanted_keypoints=[10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94]):

    if unwanted_keypoints is None:
        output_array_4D_data= array_4D_data
        return output_array_4D_data
    else:
        output_array_4D_data = np.delete(array_4D_data, unwanted_keypoints, axis=2)
        return output_array_4D_data




def centering_onesample(one_sample, x_resolution=1.0, y_resolution=1.0, x_coord=0, y_coord=1):

    centering_keypoint = 1 # neck index

    one_sample[:, :, x_coord] = one_sample[:, :, x_coord]/x_resolution
    one_sample[:, :, y_coord] = one_sample[:, :, y_coord]/y_resolution

    center_x_coord = one_sample[0, centering_keypoint, x_coord]
    center_y_coord = one_sample[0, centering_keypoint, y_coord]

    one_sample[:, :, x_coord] = one_sample[:, :, x_coord] - center_x_coord
    one_sample[:, :, y_coord] = one_sample[:, :, y_coord] - center_y_coord

    return one_sample



def centeringframe_allsamples(array_4D_data, x_resolution=1.0, y_resolution=1.0):

    centered_array_4D_data = Parallel(n_jobs=-1)(delayed(centering_onesample)(array_4D_data[one_sample_idx,:,:,:], x_resolution=x_resolution, y_resolution=y_resolution) for one_sample_idx in range(array_4D_data.shape[0]))

    return centered_array_4D_data



def interpolate_onesample_allframes(one_sample, interpolated_total_frames=8, x_resolution=1.0, y_resolution=1.0, x_coord=0, y_coord=1, c_coord=2):

    one_sample = centering_onesample(one_sample, x_resolution=x_resolution, y_resolution=y_resolution)

    original_frames, total_keypoints, xyc_coords = one_sample.shape

    interpolated_onesample = np.empty((interpolated_total_frames, total_keypoints, xyc_coords))
    interpolated_xaxis = np.linspace(0, 1, interpolated_total_frames)
    original_xaxis = np.linspace(0, 1, original_frames)

    for kpidx in range(total_keypoints):
        interpolated_onesample[:, kpidx, x_coord] = np.interp(interpolated_xaxis, original_xaxis, one_sample[:, kpidx, x_coord])
        interpolated_onesample[:, kpidx, y_coord] = np.interp(interpolated_xaxis, original_xaxis, one_sample[:, kpidx, y_coord])
        interpolated_onesample[:, kpidx, c_coord] = np.interp(interpolated_xaxis, original_xaxis, one_sample[:, kpidx, c_coord])

    assert len(interpolated_onesample.shape) == 3
    assert interpolated_onesample.shape[0] == interpolated_total_frames
    assert interpolated_onesample.shape[1] == one_sample.shape[1]
    assert interpolated_onesample.shape[2] == one_sample.shape[2]

    return interpolated_onesample



def interpolate_allsamples(input_samples_list, interpolated_total_frames=8, x_resolution=1.0, y_resolution=1.0):

    interpolated_data = np.asarray(Parallel(n_jobs=-1)(delayed(interpolate_onesample_allframes)(one_sample, interpolated_total_frames=interpolated_total_frames, x_resolution=x_resolution, y_resolution=y_resolution)  for one_sample in input_samples_list))

    interpolated_data = np.asarray(centeringframe_allsamples(interpolated_data, x_resolution=x_resolution, y_resolution=y_resolution))

    return interpolated_data




###############################################################################

# keeplist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136]

# deletelist = [10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94]

