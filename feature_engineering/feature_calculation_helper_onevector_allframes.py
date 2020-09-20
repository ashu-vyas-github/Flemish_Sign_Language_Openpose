import numpy as np
from joblib import Parallel, delayed

np.seterr(all='raise', divide='raise', over='raise', under='raise', invalid='raise')

def euclidean_distance(point1_xy, point2_xy, x_coord=0, y_coord=1, c_coord=2):

    diff_x = point2_xy[x_coord] - point1_xy[x_coord]
    diff_y = point2_xy[y_coord] - point1_xy[y_coord]
    square_add = np.power(diff_x, 2) + np.power(diff_y, 2)
    distance = np.sqrt(square_add)

    return distance

def area_polygon(xy_coordinates, x_coord=0, y_coord=1, c_coord=2):
    """
    Enclosed polygon area calculation using Shoelace formula

    https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

    Parameters
    ----------
    xy_coordinates : array of shape (num_points_closed_figure, 2)

    Returns
    -------
    area : Scalar value of area of the closed shape

    """

    x_coordinates = xy_coordinates[:, x_coord]
    y_coordinates = xy_coordinates[:, y_coord]
    dot_product = np.dot(x_coordinates, np.roll(y_coordinates, 1))
    area = 0.5*np.abs(dot_product)

    return area

def trajectory_area(onesample, keypoint_idx, x_coord=0, y_coord=1, c_coord=2):

    x_coordinates = onesample[:, keypoint_idx, x_coord]
    y_coordinates = onesample[:, keypoint_idx, y_coord]
    dot_product = np.dot(x_coordinates, np.roll(y_coordinates, 1))
    area = 0.5*np.abs(dot_product)

    return area

def shape_orientation_angel(point1_xy, point2_xy):

    numer = point2_xy[1] - point1_xy[1]
    denom = point2_xy[0] - point1_xy[0]
    alpha = np.arctan2(numer, denom)

    return alpha


def shape_perimeter(shape_xy_points):

    perimeter = 0.0
    num_points = shape_xy_points.shape[0]

    for onepoint in range(num_points):
        if onepoint < (num_points - 2):
            point1_xy = shape_xy_points[onepoint, :]
            point2_xy = shape_xy_points[onepoint+1, :]
            dist = euclidean_distance(point1_xy, point2_xy)
        elif onepoint == (num_points - 1):
            point1_xy = shape_xy_points[0, :]
            point2_xy = shape_xy_points[onepoint, :]
            dist = euclidean_distance(point1_xy, point2_xy)

        perimeter = dist + perimeter

    return perimeter


def trajectory_perimeter(onesample, keypoint_idx, x_coord=0, y_coord=1, c_coord=2):

    perimeter = 0.0
    dist = 0.0
    num_frames = onesample.shape[0]

    for oneframe_idx in range(num_frames):
        if oneframe_idx < (num_frames - 2):
            point1_xy = onesample[oneframe_idx, keypoint_idx, :]
            point2_xy = onesample[oneframe_idx+1, keypoint_idx, :]
            dist = euclidean_distance(point1_xy, point2_xy)
        elif oneframe_idx == (num_frames - 1):
            point1_xy = onesample[0, keypoint_idx, :]
            point2_xy = onesample[oneframe_idx, keypoint_idx, :]
            dist = euclidean_distance(point1_xy, point2_xy)

        perimeter = dist + perimeter

    return perimeter


def shape_compactness(shape_area, shape_perimeter):

    numer = 4*(np.pi)*shape_area
    denom = np.power(shape_perimeter, 2)

    try:
        compactness = numer/denom
    except FloatingPointError:
        print("Exception shape_compactness")
        compactness = 0.0

    return compactness


def law_of_cosines(vertexA, vertexB, vertexC):

    sideA = euclidean_distance(vertexB, vertexC)
    sideB = euclidean_distance(vertexA, vertexC)
    sideC = euclidean_distance(vertexA, vertexB)

    length_scaling_factor = 1e6

    numer = length_scaling_factor*(np.power(sideA, 2) + np.power(sideB, 2) - np.power(sideC, 2))
    denom = length_scaling_factor*(2*sideA*sideB)

    try:
        angle_C = np.arccos(numer/denom)
    except FloatingPointError:
        print("Exception law_of_cosines")
        angle_C = 0.0

    return angle_C

###############################################################################
###############################################################################

def facial_features_oneframe(oneframe, eyes_eyebrows_plus4=False):
    """
    Features calculated using the paper https://ieeexplore.ieee.org/abstract/document/4813472
    """

    face_height = euclidean_distance(oneframe[27, :], oneframe[8, :])
    face_width = euclidean_distance(oneframe[0, :], oneframe[16, :])

    eye_eye_dist = euclidean_distance(oneframe[68, :], oneframe[69, :])
    if eye_eye_dist > 0.0:
        eye_eye_dist = eye_eye_dist
    else:
        eye_eye_dist = 1.0
    left_eye_eyebrow_center_dist = euclidean_distance(oneframe[19, :], oneframe[68, :])#/eye_eye_dist
    right_eye_eyebrow_center_dist = euclidean_distance(oneframe[24, ], oneframe[69, :])#/eye_eye_dist

    mouth_area = area_polygon(oneframe[48:60, :])#/(np.power(eye_eye_dist, 2))
    mouth_height = euclidean_distance(oneframe[51, :], oneframe[57, :])#/eye_eye_dist
    mouth_width = euclidean_distance(oneframe[48, :], oneframe[54, :])#/eye_eye_dist

    alpha = shape_orientation_angel(oneframe[27, :], oneframe[8, :])
    o1_orientation = np.sin(2*alpha)
    o2_orientation = np.cos(alpha)

    perimeter = shape_perimeter(oneframe[48:60, :])
    compactness = shape_compactness(mouth_area, perimeter)

    try:
        eccentricity = face_width/face_height
    except FloatingPointError:
        print("Exception facial_features_oneframe")
        eccentricity = 0.0

    if eyes_eyebrows_plus4 in [False]:

        # return face_height, face_width, left_eye_eyebrow_center_dist, right_eye_eyebrow_center_dist, mouth_area, mouth_height, mouth_width, o1_orientation, o2_orientation, compactness, eccentricity

        return face_height, mouth_area, mouth_height, mouth_width, o1_orientation, o2_orientation, compactness, eccentricity

    else:
        left_eye_eyebrow_ear_dist = euclidean_distance(oneframe[36, :], oneframe[17, :])#/eye_eye_dist
        right_eye_eyebrow_ear_dist = euclidean_distance(oneframe[45, :], oneframe[26, :])#/eye_eye_dist
        left_eye_eyebrow_nose_dist = euclidean_distance(oneframe[39, :], oneframe[21, :])#/eye_eye_dist
        right_eye_eyebrow_nose_dist = euclidean_distance(oneframe[22, :], oneframe[42, :])#/eye_eye_dist

        return face_height, face_width, left_eye_eyebrow_center_dist, right_eye_eyebrow_center_dist, mouth_area, mouth_height, mouth_width, o1_orientation, o2_orientation, compactness, eccentricity, left_eye_eyebrow_ear_dist, right_eye_eyebrow_ear_dist, left_eye_eyebrow_nose_dist, right_eye_eyebrow_nose_dist



### Main facial features from all frames for one sample
def facial_features_onesample(onesample, eyes_eyebrows_plus4=False):

    face_data = onesample[:, 25:95, :]
    num_frames = face_data.shape[0]

    features = np.asarray(Parallel(n_jobs=-1)(delayed(facial_features_oneframe)(face_data[oneframe_idx, :, :], eyes_eyebrows_plus4=eyes_eyebrows_plus4) for oneframe_idx in range(num_frames)))

    rows, cols = features.shape[0], features.shape[1]
    facial_features_onesample = np.zeros((rows, cols+6))
    facial_features_onesample[:, :cols] = features

    mouth_area_column_idx = 4
    mouth_height_column_idx = 5
    mouth_width_column_idx = 6
    facial_features_onesample[:, cols] = np.gradient(features[:, mouth_area_column_idx]) #velocity
    facial_features_onesample[:, cols+1] = np.gradient(features[:, mouth_height_column_idx]) #velocity
    facial_features_onesample[:, cols+2] = np.gradient(features[:, mouth_width_column_idx]) #velocity
    facial_features_onesample[:, cols+3] = np.gradient(facial_features_onesample[:, cols]) #acceleration
    facial_features_onesample[:, cols+4] = np.gradient(facial_features_onesample[:, cols+1]) #acceleration
    facial_features_onesample[:, cols+5] = np.gradient(facial_features_onesample[:, cols+2]) #acceleration

    facial_features_onesample_mean = np.mean(facial_features_onesample, axis=0)
    facial_features_onesample_std = np.std(facial_features_onesample, axis=0)
    facial_features_onesample = np.hstack([facial_features_onesample_mean, facial_features_onesample_std]).ravel()

    return facial_features_onesample

###############################################################################

def body_features_oneframe(oneframe):

    """
    Features calculated using the paper https://link.springer.com/chapter/10.1007/978-3-319-20801-5_59
    """

# body angles
    l_angle_shoulder_torso = law_of_cosines(oneframe[3, :], oneframe[1, :], oneframe[2, :])
    l_angle_elbow = law_of_cosines(oneframe[4, :], oneframe[2, :], oneframe[3, :])
    l_angle_shoulder_hip = law_of_cosines(oneframe[3, :], oneframe[9, :], oneframe[2, :])
    l_angle_neck = law_of_cosines(oneframe[1, :], oneframe[4, :], oneframe[0, :])

    r_angle_shoulder_torso = law_of_cosines(oneframe[6, :], oneframe[1, :], oneframe[5, :])
    r_angle_elbow = law_of_cosines(oneframe[7, :], oneframe[5, :], oneframe[6, :])
    r_angle_shoulder_hip = law_of_cosines(oneframe[6, :], oneframe[12, :], oneframe[5, :])
    r_angle_neck = law_of_cosines(oneframe[1, :], oneframe[7, :], oneframe[0, :])

# body euclidean distances
    torso_length = euclidean_distance(oneframe[0, :], oneframe[8, :])
    l_arm = euclidean_distance(oneframe[2, :], oneframe[4, :])#/torso_length
    r_arm = euclidean_distance(oneframe[5, :], oneframe[7, :])#/torso_length
    l_wrist_torso = euclidean_distance(oneframe[4, :], oneframe[1, :])#/torso_length
    r_wrist_torso = euclidean_distance(oneframe[7, :], oneframe[1, :])#/torso_length
    l_elbow_hip = euclidean_distance(oneframe[3, :], oneframe[8, :])#/torso_length
    r_elbow_hip = euclidean_distance(oneframe[6, :], oneframe[8, :])#/torso_length
    l_elbow_r_shoulder = euclidean_distance(oneframe[3, :], oneframe[5, :])#/torso_length
    r_elbow_l_shoulder = euclidean_distance(oneframe[2, :], oneframe[6, :])#/torso_length
    wrist_to_wrist = euclidean_distance(oneframe[4, :], oneframe[7, :])#/torso_length
    elbow_to_elbow = euclidean_distance(oneframe[3, :], oneframe[6, :])#/torso_length

# radial and angular features
# radial distances of elbows and wrists from neck
    l_elbow_neck = euclidean_distance(oneframe[1, :], oneframe[3, :])
    l_wrist_neck = euclidean_distance(oneframe[1, :], oneframe[4, :])
    r_elbow_neck = euclidean_distance(oneframe[1, :], oneframe[6, :])
    r_wrist_neck = euclidean_distance(oneframe[1, :], oneframe[7, :])

#    return l_angle_shoulder_torso, l_angle_elbow, l_angle_shoulder_hip, l_angle_neck, r_angle_shoulder_torso, r_angle_elbow, r_angle_shoulder_hip, r_angle_neck, l_arm, r_arm, l_wrist_torso, r_wrist_torso, l_elbow_hip, r_elbow_hip, l_elbow_r_shoulder, r_elbow_l_shoulder, wrist_to_wrist, elbow_to_elbow

    return l_angle_shoulder_torso, l_angle_elbow, l_angle_neck, r_angle_shoulder_torso, r_angle_elbow, r_angle_neck, wrist_to_wrist, elbow_to_elbow, l_elbow_neck, l_wrist_neck, r_elbow_neck, r_wrist_neck



### Main BODY features from all frames for one sample
def body_features_onesample(onesample):

    body_data = onesample[:, 0:25, :]
    num_frames = body_data.shape[0]

    features = np.asarray(Parallel(n_jobs=-1)(delayed(body_features_oneframe)(body_data[oneframe_idx, :, :]) for oneframe_idx in range(num_frames)))

    rows, cols = features.shape[0], features.shape[1]
    body_features_onesample = np.zeros((rows, cols+16))
    body_features_onesample[:, :cols] = features

# radial velocity and radial acceleration
    body_features_onesample[:, cols] = np.gradient(features[:, -4])
    body_features_onesample[:, cols+1] = np.gradient(features[:, -3])
    body_features_onesample[:, cols+2] = np.gradient(features[:, -2])
    body_features_onesample[:, cols+3] = np.gradient(features[:, -1])
    body_features_onesample[:, cols+4] = np.gradient(body_features_onesample[:, cols])
    body_features_onesample[:, cols+5] = np.gradient(body_features_onesample[:, cols+1])
    body_features_onesample[:, cols+6] = np.gradient(body_features_onesample[:, cols+2])
    body_features_onesample[:, cols+7] = np.gradient(body_features_onesample[:, cols+3])


    body_features_onesample_mean = np.mean(body_features_onesample, axis=0)
    body_features_onesample_std = np.std(body_features_onesample, axis=0)
    body_features_onesample = np.hstack([body_features_onesample_mean, body_features_onesample_std]).ravel()

    return body_features_onesample


###############################################################################

def one_hand_calculations(hand_data):

    angle_thumb_index = law_of_cosines(hand_data[8, :], hand_data[4, :], hand_data[2, :])
    angle_index_middle = law_of_cosines(hand_data[12, :], hand_data[8, :], hand_data[5, :])
    angle_middle_ring = law_of_cosines(hand_data[16, :], hand_data[12, :], hand_data[9, :])
    angle_ring_pinky = law_of_cosines(hand_data[20, :], hand_data[16, :], hand_data[13, :])

    thumb_length = euclidean_distance(hand_data[2, :], hand_data[4, :])
    index_length = euclidean_distance(hand_data[5, :], hand_data[8, :])
    middle_length = euclidean_distance(hand_data[9, :], hand_data[12, :])
    ring_length = euclidean_distance(hand_data[13, :], hand_data[16, :])
    pinky_length = euclidean_distance(hand_data[17, :], hand_data[20, :])

    finger_tips_points_xy = np.array([hand_data[0, :], hand_data[4, :], hand_data[8, :], hand_data[12, :], hand_data[16, :], hand_data[20, :]])
    area = area_polygon(finger_tips_points_xy)

    return angle_thumb_index, angle_index_middle, angle_middle_ring, angle_ring_pinky, thumb_length, index_length, middle_length, ring_length, pinky_length, area


def both_hands_features_oneframe(left_hand_data, right_hand_data):

    l_angle_thumb_index, l_angle_index_middle, l_angle_middle_ring, l_angle_ring_pinky, l_thumb_length, l_index_length, l_middle_length, l_ring_length, l_pinky_length, l_area = one_hand_calculations(left_hand_data)

    r_angle_thumb_index, r_angle_index_middle, r_angle_middle_ring, r_angle_ring_pinky, r_thumb_length, r_index_length, r_middle_length, r_ring_length, r_pinky_length, r_area = one_hand_calculations(right_hand_data)

    return l_angle_thumb_index, l_angle_index_middle, l_angle_middle_ring, l_angle_ring_pinky, l_thumb_length, l_index_length, l_middle_length, l_ring_length, l_pinky_length, l_area, r_angle_thumb_index, r_angle_index_middle, r_angle_middle_ring, r_angle_ring_pinky, r_thumb_length, r_index_length, r_middle_length, r_ring_length, r_pinky_length, r_area




### Main hands features from all frames for one sample
def both_hands_features_onesample(onesample):

    left_hand_data = onesample[:, 95:116, :]
    right_hand_data = onesample[:, 116:, :]
    num_frames = left_hand_data.shape[0]

    features = np.asarray(Parallel(n_jobs=-1)(delayed(both_hands_features_oneframe)(left_hand_data[oneframe_idx, :, :], right_hand_data[oneframe_idx, :, :]) for oneframe_idx in range(num_frames)))

    rows, cols = features.shape[0], features.shape[1]
    both_hands_features_onesample = np.zeros((rows, cols+4))
    both_hands_features_onesample[:, :cols] = features

    l_area_column_idx = 9
    r_area_column_idx = 19
    both_hands_features_onesample[:, cols] = np.gradient(features[:, l_area_column_idx]) #velocity
    both_hands_features_onesample[:, cols+1] = np.gradient(features[:, r_area_column_idx]) #velocity
    both_hands_features_onesample[:, cols+2] = np.gradient(both_hands_features_onesample[:, cols]) #acceleration
    both_hands_features_onesample[:, cols+3] = np.gradient(both_hands_features_onesample[:, cols+1]) #acceleration

    both_hands_features_onesample_mean = np.mean(both_hands_features_onesample, axis=0)
    both_hands_features_onesample_std = np.std(both_hands_features_onesample, axis=0)
    both_hands_features_onesample = np.hstack([both_hands_features_onesample_mean, both_hands_features_onesample_std]).ravel()

    return both_hands_features_onesample


def physics_features(onesample, angular_flag=True, x_coord=0, y_coord=1, c_coord=2):

# Linear Cartesian features
    x_position_mean = np.mean(onesample[:, :, x_coord], axis=0)
    y_position_mean = np.mean(onesample[:, :, y_coord], axis=0)
    # x_position_std = np.std(onesample[:, :, x_coord], axis=0)
    # y_position_std = np.std(onesample[:, :, y_coord], axis=0)

    xy_position_mean_std = np.hstack([x_position_mean, y_position_mean])#, x_position_std, y_position_std])

    try:
        x_velocity_mean = np.mean(np.gradient(onesample[:, :, x_coord], axis=0), axis=0)
        y_velocity_mean = np.mean(np.gradient(onesample[:, :, y_coord], axis=0), axis=0)

        x_acceleration_mean = np.mean(np.gradient(np.gradient(onesample[:, :, x_coord], axis=0), axis=0), axis=0)
        y_acceleration_mean = np.mean(np.gradient(np.gradient(onesample[:, :, y_coord], axis=0), axis=0), axis=0)

    except ValueError:
        x_velocity_mean = np.zeros(x_position_mean.shape)
        y_velocity_mean = np.zeros(x_position_mean.shape)
        x_acceleration_mean = np.zeros(x_position_mean.shape)
        y_acceleration_mean = np.zeros(x_position_mean.shape)


    xy_velocity_mean = np.hstack([x_velocity_mean, y_velocity_mean])
    xy_acceleration_mean = np.hstack([x_acceleration_mean, y_acceleration_mean])
    if angular_flag in [True]:
# angular displacement
        l_elbow_neck_angle = np.arctan2(onesample[:, 3, y_coord], onesample[:, 3, x_coord])
        l_wrist_neck_angle = np.arctan2(onesample[:, 4, y_coord], onesample[:, 4, x_coord])
        r_elbow_neck_angle = np.arctan2(onesample[:, 6, y_coord], onesample[:, 6, x_coord])
        r_wrist_neck_angle = np.arctan2(onesample[:, 7, y_coord], onesample[:, 7, x_coord])
# angular velocity and acceleration
        l_elbow_neck_disp = np.mean(l_elbow_neck_angle, axis=0)
        l_wrist_neck_disp = np.mean(l_wrist_neck_angle, axis=0)
        r_elbow_neck_disp = np.mean(r_elbow_neck_angle, axis=0)
        r_wrist_neck_disp = np.mean(r_wrist_neck_angle, axis=0)

        try:
            l_elbow_neck_velo = np.mean(np.gradient(l_elbow_neck_angle, axis=0), axis=0)
            l_wrist_neck_velo = np.mean(np.gradient(l_wrist_neck_angle, axis=0), axis=0)
            r_elbow_neck_velo = np.mean(np.gradient(r_elbow_neck_angle, axis=0), axis=0)
            r_wrist_neck_velo = np.mean(np.gradient(r_wrist_neck_angle, axis=0), axis=0)

            l_elbow_neck_accl = np.mean(np.gradient(np.gradient(l_elbow_neck_angle, axis=0), axis=0), axis=0)
            l_wrist_neck_accl = np.mean(np.gradient(np.gradient(l_wrist_neck_angle, axis=0), axis=0), axis=0)
            r_elbow_neck_accl = np.mean(np.gradient(np.gradient(r_elbow_neck_angle, axis=0), axis=0), axis=0)
            r_wrist_neck_accl = np.mean(np.gradient(np.gradient(r_wrist_neck_angle, axis=0), axis=0), axis=0)
        except ValueError:
            l_elbow_neck_velo = np.zeros(l_elbow_neck_disp.shape)
            l_wrist_neck_velo = np.zeros(l_elbow_neck_disp.shape)
            r_elbow_neck_velo = np.zeros(l_elbow_neck_disp.shape)
            r_wrist_neck_velo = np.zeros(l_elbow_neck_disp.shape)

            l_elbow_neck_accl = np.zeros(l_elbow_neck_disp.shape)
            l_wrist_neck_accl = np.zeros(l_elbow_neck_disp.shape)
            r_elbow_neck_accl = np.zeros(l_elbow_neck_disp.shape)
            r_wrist_neck_accl = np.zeros(l_elbow_neck_disp.shape)

        # feature_vector = np.hstack([xy_position_mean_std, xy_velocity_mean, xy_acceleration_mean, l_elbow_neck_disp, l_wrist_neck_disp, r_elbow_neck_disp, r_wrist_neck_disp, l_elbow_neck_velo, l_wrist_neck_velo,r_elbow_neck_velo, r_wrist_neck_velo, l_elbow_neck_accl, l_wrist_neck_accl, r_elbow_neck_accl, r_wrist_neck_accl])

        feature_vector = np.hstack([xy_position_mean_std, l_elbow_neck_velo, l_wrist_neck_velo,r_elbow_neck_velo, r_wrist_neck_velo])

    else:
        # feature_vector = np.hstack([xy_position_mean_std, xy_velocity_mean, xy_acceleration_mean])
        feature_vector = xy_position_mean_std

    return feature_vector





