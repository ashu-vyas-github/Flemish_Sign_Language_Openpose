import numpy as np
# import feature_engineering.feature_calculation_helper as feat_help
import feature_engineering.feature_preprocessing as feature_preprocessing
import feature_engineering.features_4D_array as feat_help

def facial_features_extractor(array_4D_data):

    facial_features = []

    for one_sample_idx in range(array_4D_data.shape[0]):
        calc = feat_help.facial_features_onesample(array_4D_data[one_sample_idx, :, :, :])
        facial_features.append(calc)

    return np.asarray(facial_features)



def body_features_extractor(array_4D_data):

    body_features = []

    for one_sample_idx in range(array_4D_data.shape[0]):
        calc = feat_help.body_features_onesample(array_4D_data[one_sample_idx, :, :, :])
        body_features.append(calc)

    return np.asarray(body_features)



def hands_features_extractor(array_4D_data):

    hands_features = []

    for one_sample_idx in range(array_4D_data.shape[0]):
        calc = feat_help.both_hands_features_onesample(array_4D_data[one_sample_idx, :, :, :])
        hands_features.append(calc)

    return np.asarray(hands_features)



def physics_extractor(array_4D_data, linear_flag=True, angular_flag=True, std_flag=True, velocity_flag=True, acceleration_flag=True):

    physics_features = []

    for one_sample_idx in range(array_4D_data.shape[0]):
        calc = feat_help.physics_features(array_4D_data[one_sample_idx, :, :, :], linear_flag=linear_flag, angular_flag=angular_flag, std_flag=std_flag, velocity_flag=velocity_flag, acceleration_flag=acceleration_flag)
        physics_features.append(calc)

    return np.asarray(physics_features)



def trajectory_extractor(array_4D_data):

    trajectory_vector = []
    if array_4D_data.shape[2] < 130:
        keypoints_list = [2, 3, 5, 6] # if removing lower body keypoints
    else:
        keypoints_list = [3, 4, 6, 7] # original 137 keypoints
    # keypoints_list = [3, 4, 6, 7, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136]

    for one_sample_idx in range(array_4D_data.shape[0]):
        trajectory_vector_onesample = []
        for one_keypoint in keypoints_list:
            traj_peri_val = feat_help.trajectory_perimeter(array_4D_data[one_sample_idx, :, :, :], keypoint_idx=one_keypoint)
            trajectory_vector_onesample.append(traj_peri_val)
            traj_area = feat_help.trajectory_area(array_4D_data[one_sample_idx, :, :, :], keypoint_idx=one_keypoint)
            trajectory_vector_onesample.append(traj_area)
        trajectory_vector.append(trajectory_vector_onesample)

    return np.asarray(trajectory_vector)



def main_feature_extractor(array_4D_data, face=True, body=True, hands=True, physics=True, trajectory=True, linear_flag=True, angular_flag=True, std_flag=True, velocity_flag=True, acceleration_flag=True, remove_keypoints=True, unwanted_keypoints=[10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94]):

    features_object_list = []

    if face in [True]:
        print("Extracting facial features.....")
        facial_features = facial_features_extractor(array_4D_data)
        features_object_list.append(facial_features)
    else:
        print("Facial features not extracted")

    if body in [True]:
        print("Extracting body features.....")
        body_features = body_features_extractor(array_4D_data)
        features_object_list.append(body_features)
    else:
        print("Body features not extracted")

    if hands in [True]:
        print("Extracting hands features.....")
        hands_features = hands_features_extractor(array_4D_data)
        features_object_list.append(hands_features)
    else:
        print("Hands features not extracted")

    if physics in [True]:
        print("Extracting physics features.....")
        if remove_keypoints in [True]:
            array_4D_subset = feature_preprocessing.remove_unwanted_keypoints(array_4D_data, unwanted_keypoints=unwanted_keypoints)
        else:
            array_4D_subset = array_4D_data
        physics_features = physics_extractor(array_4D_subset, linear_flag=linear_flag, angular_flag=angular_flag, std_flag=std_flag, velocity_flag=velocity_flag, acceleration_flag=acceleration_flag)
        features_object_list.append(physics_features)
    else:
        print("Physics features not extracted")

    if trajectory in [True]:
        print("Extracting trajectory features.....")
        trajectory_features = trajectory_extractor(array_4D_data)
        features_object_list.append(trajectory_features)
    else:
        print("Trajectory features not extracted")

    print("Features extracted")
    if len(features_object_list) >= 2:
        extracted_features = np.hstack(features_object_list)
        return extracted_features
    elif len(features_object_list) == 1:
        extracted_features = features_object_list[0]
        return extracted_features
    else:
        print("No features extracted")
        return 0




###############################################################################
###############################################################################


# full_list = [3, 4, 6, 7, 31, 32, 33, 34, 35, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136]

# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136]
