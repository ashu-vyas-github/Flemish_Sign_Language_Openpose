import numpy as np
# from joblib import Parallel, delayed
# import feature_engineering.feature_calculation_helper as feat_help
import feature_engineering.feature_calculation_helper_onevector_allframes as feat_help

def facial_features_extractor(array_4D_data, eyes_eyebrows_plus4=False):

    #facial_features = np.asarray(Parallel(n_jobs=-1)(delayed(feat_help.facial_features_onesample)(array_4D_data[one_sample_idx, :, :, :], eyes_eyebrows_plus4=eyes_eyebrows_plus4)  for one_sample_idx in range(array_4D_data.shape[0])))

    facial_features = []

    for one_sample_idx in range(array_4D_data.shape[0]):
        calc = feat_help.facial_features_onesample(array_4D_data[one_sample_idx, :, :, :], eyes_eyebrows_plus4=eyes_eyebrows_plus4)
        facial_features.append(calc)

    return np.asarray(facial_features)


def body_features_extractor(array_4D_data):

    #body_features = np.asarray(Parallel(n_jobs=-1)(delayed(feat_help.body_features_onesample)(array_4D_data[one_sample_idx, :, :, :])  for one_sample_idx in range(array_4D_data.shape[0])))

    body_features = []

    for one_sample_idx in range(array_4D_data.shape[0]):
        calc = feat_help.body_features_onesample(array_4D_data[one_sample_idx, :, :, :])
        body_features.append(calc)

    return np.asarray(body_features)

def hands_features_extractor(array_4D_data):

    # hands_features = np.asarray(Parallel(n_jobs=-1)(delayed(feat_help.both_hands_features_onesample)(array_4D_data[one_sample_idx, :, :, :])  for one_sample_idx in range(array_4D_data.shape[0])))

    hands_features = []

    for one_sample_idx in range(array_4D_data.shape[0]):
        calc = feat_help.both_hands_features_onesample(array_4D_data[one_sample_idx, :, :, :])
        hands_features.append(calc)

    return np.asarray(hands_features)



def physics_extractor(samples_list, angular_flag=True):

    physics_features = []

    for one_sample_idx in range(len(samples_list)):
        calc = feat_help.physics_features(samples_list[one_sample_idx], angular_flag=angular_flag)
        physics_features.append(calc)

    return np.asarray(physics_features)



def trajectory_extractor(samples_list):

    trajectory_vector = []
    keypoints_list = [2, 3, 5, 6] # if removing lower body keypoints
    # keypoints_list = [3, 4, 6, 7] # original 137 keypoints

    for one_sample_idx in range(len(samples_list)):
        trajectory_vector_onesample = []
        for one_keypoint in keypoints_list:
            traj_peri_val = feat_help.trajectory_perimeter(samples_list[one_sample_idx], keypoint_idx=one_keypoint)
            trajectory_vector_onesample.append(traj_peri_val)
            traj_area = feat_help.trajectory_area(samples_list[one_sample_idx], keypoint_idx=one_keypoint)
            trajectory_vector_onesample.append(traj_area)
        trajectory_vector.append(trajectory_vector_onesample)

    return np.asarray(trajectory_vector)



def main_feature_extractor(array_4D_data, samples_list_data, face=True, body=True, hands=True, physics=True, angular_flag=True, trajectory=True, eyes_eyebrows_plus4=False):

    features_object_list = []

    if face in [True]:
        print("Extracting facial features.....")
        facial_features = facial_features_extractor(array_4D_data, eyes_eyebrows_plus4=eyes_eyebrows_plus4)
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
        physics_features = physics_extractor(samples_list_data, angular_flag=angular_flag)
        features_object_list.append(physics_features)
    else:
        print("Physics features not extracted")

    if trajectory in [True]:
        print("Extracting trajectory features.....")
        trajectory_features = trajectory_extractor(samples_list_data)
        features_object_list.append(trajectory_features)
    else:
        print("Trajectory features not extracted")

    print("\nFeatures extracted")
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
