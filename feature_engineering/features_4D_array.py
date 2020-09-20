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
    """
    angle will be inscribed at vertexC
    """

    sideA = euclidean_distance(vertexB, vertexC)
    sideB = euclidean_distance(vertexA, vertexC)
    sideC = euclidean_distance(vertexA, vertexB)

    # length_scaling_factor = 1e6

    numer = (np.power(sideA, 2) + np.power(sideB, 2) - np.power(sideC, 2))
    denom = (2*sideA*sideB)

    try:
        angle_C = np.arccos(numer/denom)
    except FloatingPointError:
        print("Exception law_of_cosines")
        angle_C = 0.0

    return angle_C

###############################################################################
###############################################################################

def facial_features_oneframe(oneframe):
    """
    Features calculated using the paper https://ieeexplore.ieee.org/abstract/document/4813472
    """

    mouth_area = area_polygon(oneframe[48:60, :])
    mouth_height = euclidean_distance(oneframe[51, :], oneframe[57, :])
    mouth_width = euclidean_distance(oneframe[48, :], oneframe[54, :])

    # alpha = shape_orientation_angel(oneframe[57, :], oneframe[51, :])
    # mouth_o1_orientation = np.sin(2*alpha)
    # mouth_o2_orientation = np.cos(alpha)

    # perimeter = shape_perimeter(oneframe[48:60, :])
    # compactness = shape_compactness(mouth_area, perimeter)

    # try:
    #     eccentricity = mouth_height/mouth_width
    # except FloatingPointError:
    #     print("Exception facial_features_oneframe")
    #     eccentricity = 0.0

    # return mouth_area, mouth_height, mouth_width, mouth_o1_orientation, mouth_o2_orientation, compactness, eccentricity

    return mouth_area, mouth_height, mouth_width



### Main facial features from all frames for one sample
def facial_features_onesample(onesample):

    face_data = onesample[:, 25:95, :]
    num_frames = face_data.shape[0]

    features = np.asarray(Parallel(n_jobs=-1)(delayed(facial_features_oneframe)(face_data[oneframe_idx, :, :]) for oneframe_idx in range(num_frames)))

    rows, cols = features.shape[0], features.shape[1]
    facial_features_onesample = np.zeros((rows, cols+6)) # change here to +6 for velocity as well
    facial_features_onesample[:, :cols] = features

    mouth_area_column_idx = 0
    mouth_height_column_idx = 1
    mouth_width_column_idx = 2
    # facial_features_onesample[:, cols] = np.gradient(features[:, mouth_area_column_idx]) #velocity
    # facial_features_onesample[:, cols+1] = np.gradient(features[:, mouth_height_column_idx]) #velocity
    # facial_features_onesample[:, cols+2] = np.gradient(features[:, mouth_width_column_idx]) #velocity
    # facial_features_onesample[:, cols+3] = np.gradient(facial_features_onesample[:, cols]) #acceleration
    # facial_features_onesample[:, cols+4] = np.gradient(facial_features_onesample[:, cols+1]) #acceleration
    # facial_features_onesample[:, cols+5] = np.gradient(facial_features_onesample[:, cols+2]) #acceleration

    facial_features_onesample_mean = np.mean(facial_features_onesample, axis=0)
    facial_features_onesample = facial_features_onesample_mean

    return facial_features_onesample

###############################################################################

def body_features_oneframe(oneframe):

    """
    Features calculated using the paper https://link.springer.com/chapter/10.1007/978-3-319-20801-5_59
    """

# body angles
    # l_angle_shoulder_torso = law_of_cosines(oneframe[3, :], oneframe[1, :], oneframe[2, :])
    # l_angle_elbow = law_of_cosines(oneframe[4, :], oneframe[2, :], oneframe[3, :])
    # l_angle_wrist = law_of_cosines(oneframe[1, :], oneframe[2, :], oneframe[4, :])

    # r_angle_shoulder_torso = law_of_cosines(oneframe[6, :], oneframe[1, :], oneframe[5, :])
    # r_angle_elbow = law_of_cosines(oneframe[7, :], oneframe[5, :], oneframe[6, :])
    # r_angle_wrist = law_of_cosines(oneframe[1, :], oneframe[5, :], oneframe[7, :])

# body euclidean distances
    l_arm = euclidean_distance(oneframe[2, :], oneframe[4, :])
    r_arm = euclidean_distance(oneframe[5, :], oneframe[7, :])
    l_wrist_torso = euclidean_distance(oneframe[4, :], oneframe[1, :])
    r_wrist_torso = euclidean_distance(oneframe[7, :], oneframe[1, :])
    l_elbow_hip = euclidean_distance(oneframe[3, :], oneframe[8, :])
    r_elbow_hip = euclidean_distance(oneframe[6, :], oneframe[8, :])
    l_wrist_r_shoulder = euclidean_distance(oneframe[4, :], oneframe[5, :])
    r_wrist_l_shoulder = euclidean_distance(oneframe[2, :], oneframe[7, :])
    wrist_to_wrist = euclidean_distance(oneframe[4, :], oneframe[7, :])
    elbow_to_elbow = euclidean_distance(oneframe[3, :], oneframe[6, :])

    # return l_angle_shoulder_torso, l_angle_elbow, l_angle_wrist, r_angle_shoulder_torso, r_angle_elbow, r_angle_wrist, l_arm, r_arm, l_wrist_torso, r_wrist_torso, l_elbow_hip, r_elbow_hip, l_wrist_r_shoulder, r_wrist_l_shoulder, wrist_to_wrist, elbow_to_elbow
# indices as 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

    return l_arm, r_arm, l_wrist_torso, r_wrist_torso, l_elbow_hip, r_elbow_hip, l_wrist_r_shoulder, r_wrist_l_shoulder, wrist_to_wrist, elbow_to_elbow



### Main BODY features from all frames for one sample
def body_features_onesample(onesample):

    body_data = onesample[:, 0:25, :]
    num_frames = body_data.shape[0]

    features = np.asarray(Parallel(n_jobs=-1)(delayed(body_features_oneframe)(body_data[oneframe_idx, :, :]) for oneframe_idx in range(num_frames)))

    rows, cols = features.shape[0], features.shape[1]
    body_features_onesample = np.zeros((rows, cols+8))
    body_features_onesample[:, :cols] = features

# first and second order derivative of l_angle_wrist, r_angle_wrist, l_wrist_torso, r_wrist_torso
    body_features_onesample[:, cols] = np.gradient(features[:, 2]) #2 l_angle_wrist
    body_features_onesample[:, cols+1] = np.gradient(features[:, 3]) #5 r_angle_wrist
    body_features_onesample[:, cols+2] = np.gradient(features[:, 8])
    body_features_onesample[:, cols+3] = np.gradient(features[:, 9])
    # body_features_onesample[:, cols+4] = np.gradient(body_features_onesample[:, cols])
    # body_features_onesample[:, cols+5] = np.gradient(body_features_onesample[:, cols+1])
    # body_features_onesample[:, cols+6] = np.gradient(body_features_onesample[:, cols+2])
    # body_features_onesample[:, cols+7] = np.gradient(body_features_onesample[:, cols+3])

    body_features_onesample_mean = np.mean(body_features_onesample, axis=0)
    body_features_onesample = body_features_onesample_mean

    return body_features_onesample


###############################################################################

def one_hand_calculations(hand_data):

    # angle_thumb_index = law_of_cosines(hand_data[8, :], hand_data[4, :], hand_data[2, :])
    # angle_index_middle = law_of_cosines(hand_data[12, :], hand_data[8, :], hand_data[5, :])
    # angle_middle_ring = law_of_cosines(hand_data[16, :], hand_data[12, :], hand_data[9, :])
    # angle_ring_pinky = law_of_cosines(hand_data[20, :], hand_data[16, :], hand_data[13, :])

    thumb_length = euclidean_distance(hand_data[2, :], hand_data[4, :])
    index_length = euclidean_distance(hand_data[5, :], hand_data[8, :])
    middle_length = euclidean_distance(hand_data[9, :], hand_data[12, :])
    ring_length = euclidean_distance(hand_data[13, :], hand_data[16, :])
    pinky_length = euclidean_distance(hand_data[17, :], hand_data[20, :])

    finger_tips_points_xy = np.array([hand_data[0, :], hand_data[4, :], hand_data[8, :], hand_data[12, :], hand_data[16, :], hand_data[20, :]])
    area = area_polygon(finger_tips_points_xy)

    alpha = shape_orientation_angel(hand_data[5, :], hand_data[17, :])
    hand_o1_orientation = np.sin(2*alpha)
    hand_o2_orientation = np.cos(alpha)

    # return angle_thumb_index, angle_index_middle, angle_middle_ring, angle_ring_pinky, thumb_length, index_length, middle_length, ring_length, pinky_length, area, hand_o1_orientation, hand_o2_orientation

    return thumb_length, index_length, middle_length, ring_length, pinky_length, area, hand_o1_orientation, hand_o2_orientation


def both_hands_features_oneframe(left_hand_data, right_hand_data):

    hands_features = []
    hands_features.append(one_hand_calculations(left_hand_data))
    hands_features.append(one_hand_calculations(right_hand_data))

    hands_features = np.asarray(hands_features).ravel()

    return hands_features




### Main hands features from all frames for one sample
def both_hands_features_onesample(onesample):

    left_hand_data = onesample[:, 95:116, :]
    right_hand_data = onesample[:, 116:, :]
    num_frames = left_hand_data.shape[0]

    features = np.asarray(Parallel(n_jobs=-1)(delayed(both_hands_features_oneframe)(left_hand_data[oneframe_idx, :, :], right_hand_data[oneframe_idx, :, :]) for oneframe_idx in range(num_frames)))

    rows, cols = features.shape[0], features.shape[1]
    both_hands_features_onesample = np.zeros((rows, cols+4))
    both_hands_features_onesample[:, :cols] = features

    l_area_column_idx = 5 #9
    r_area_column_idx = 13 #21
    # both_hands_features_onesample[:, cols] = np.gradient(features[:, l_area_column_idx]) #velocity
    # both_hands_features_onesample[:, cols+1] = np.gradient(features[:, r_area_column_idx]) #velocity
    # both_hands_features_onesample[:, cols+2] = np.gradient(both_hands_features_onesample[:, cols]) #acceleration
    # both_hands_features_onesample[:, cols+3] = np.gradient(both_hands_features_onesample[:, cols+1]) #acceleration

    both_hands_features_onesample_mean = np.mean(both_hands_features_onesample, axis=0)
    both_hands_features_onesample = both_hands_features_onesample_mean

    return both_hands_features_onesample



#### THIS IS MAX FUNCTION
# def physics_features(onesample, linear_flag=True, angular_flag=True, std_flag=True, velocity_flag=True, acceleration_flag=True, x_coord=0, y_coord=1, c_coord=2):

#     # Assumption that irrelevant keypoints have been removed from onesample, see feature_extractors
#     abc = 0
#     feature_vector_list = []

# # Linear Cartesian features
#     if linear_flag in [True]:
#         # linear max xy positions
#         feature_vector_list.append(np.max(onesample[:, :, x_coord], axis=0))
#         feature_vector_list.append(np.max(onesample[:, :, y_coord], axis=0))
#         feature_vector_list.append(np.max(onesample[:, :, c_coord], axis=0))
#         if std_flag in [True]:
#             feature_vector_list.append(np.std(onesample[:, :, x_coord], axis=0))
#             feature_vector_list.append(np.std(onesample[:, :, y_coord], axis=0))
#             feature_vector_list.append(np.std(onesample[:, :, c_coord], axis=0))
#         else:
#             abc = 0

#         if velocity_flag in [True]:
#         # linear max xy velocity
#             feature_vector_list.append(np.max(np.gradient(onesample[:, :, x_coord], axis=0), axis=0))
#             feature_vector_list.append(np.max(np.gradient(onesample[:, :, y_coord], axis=0), axis=0))
#             feature_vector_list.append(np.max(np.gradient(onesample[:, :, c_coord], axis=0), axis=0))
#         else:
#             abc = 0

#         if acceleration_flag in [True]:
#             # linear max xy acceleration
#             feature_vector_list.append(np.max(np.gradient(np.gradient(onesample[:, :, x_coord], axis=0), axis=0), axis=0))
#             feature_vector_list.append(np.max(np.gradient(np.gradient(onesample[:, :, y_coord], axis=0), axis=0), axis=0))
#             feature_vector_list.append(np.max(np.gradient(np.gradient(onesample[:, :, c_coord], axis=0), axis=0), axis=0))
#     else:
#         abc = 0

# # Angular radial and polar features
#     if angular_flag in [True]:
#         complex_onesample = onesample.astype(complex)
#         complex_onesample = complex_onesample[:,:,x_coord] + complex_onesample[:,:,y_coord]*1j
#         # angular displacement
#         radius_calc = np.sqrt(np.power(onesample[:,:,x_coord], 2) + np.power(onesample[:,:,x_coord], 2))
#         theta_calc = np.angle(complex_onesample)
#         feature_vector_list.append(np.max(radius_calc, axis=0))
#         feature_vector_list.append(np.max(theta_calc, axis=0))
#         feature_vector_list.append(np.max(onesample[:, :, c_coord], axis=0))

#         if velocity_flag in [True]:
#             feature_vector_list.append(np.max(np.gradient(radius_calc, axis=0), axis=0))
#             feature_vector_list.append(np.max(np.gradient(theta_calc, axis=0), axis=0))
#             feature_vector_list.append(np.max(np.gradient(onesample[:, :, y_coord], axis=0), axis=0))
#         else:
#             abc = 0

#         if acceleration_flag in [True]:
#             feature_vector_list.append(np.max(np.gradient(np.gradient(radius_calc, axis=0), axis=0), axis=0))
#             feature_vector_list.append(np.max(np.gradient(np.gradient(theta_calc, axis=0), axis=0), axis=0))
#             feature_vector_list.append(np.max(np.gradient(np.gradient(onesample[:, :, c_coord], axis=0), axis=0), axis=0))
#         else:
#             abc = 0

#     feature_vector = np.asarray(feature_vector_list)
#     feature_vector = feature_vector.astype(float)
#     feature_vector = feature_vector.ravel()

#     return feature_vector


##### THIS IS MEAN FUNCTION
def physics_features(onesample, linear_flag=True, angular_flag=True, std_flag=True, velocity_flag=True, acceleration_flag=True, x_coord=0, y_coord=1, c_coord=2):

    # Assumption that irrelevant keypoints have been removed from onesample, see feature_extractors
    abc = 0
    feature_vector_list = []

# Linear Cartesian features
    if linear_flag in [True]:
        # linear mean xy positions
        feature_vector_list.append(np.mean(onesample[:, :, x_coord], axis=0))
        feature_vector_list.append(np.mean(onesample[:, :, y_coord], axis=0))
        feature_vector_list.append(np.mean(onesample[:, :, c_coord], axis=0))
        if std_flag in [True]:
            feature_vector_list.append(np.std(onesample[:, :, x_coord], axis=0))
            feature_vector_list.append(np.std(onesample[:, :, y_coord], axis=0))
            feature_vector_list.append(np.std(onesample[:, :, c_coord], axis=0))
        else:
            abc = 0

        if velocity_flag in [True]:
        # linear mean xy velocity
            feature_vector_list.append(np.mean(np.gradient(onesample[:, :, x_coord], axis=0), axis=0))
            feature_vector_list.append(np.mean(np.gradient(onesample[:, :, y_coord], axis=0), axis=0))
            feature_vector_list.append(np.mean(np.gradient(onesample[:, :, c_coord], axis=0), axis=0))
        else:
            abc = 0

        if acceleration_flag in [True]:
            # linear mean xy acceleration
            feature_vector_list.append(np.mean(np.gradient(np.gradient(onesample[:, :, x_coord], axis=0), axis=0), axis=0))
            feature_vector_list.append(np.mean(np.gradient(np.gradient(onesample[:, :, y_coord], axis=0), axis=0), axis=0))
            feature_vector_list.append(np.mean(np.gradient(np.gradient(onesample[:, :, c_coord], axis=0), axis=0), axis=0))
    else:
        abc = 0

# Angular radial and polar features
    if angular_flag in [True]:
        complex_onesample = onesample.astype(complex)
        complex_onesample = complex_onesample[:,:,x_coord] + complex_onesample[:,:,y_coord]*1j
        # angular displacement
        radius_calc = np.sqrt(np.power(onesample[:,:,x_coord], 2) + np.power(onesample[:,:,x_coord], 2))
        theta_calc = np.angle(complex_onesample)
        feature_vector_list.append(np.mean(radius_calc, axis=0))
        feature_vector_list.append(np.mean(theta_calc, axis=0))
        feature_vector_list.append(np.mean(onesample[:, :, c_coord], axis=0))

        if velocity_flag in [True]:
            feature_vector_list.append(np.mean(np.gradient(radius_calc, axis=0), axis=0))
            feature_vector_list.append(np.mean(np.gradient(theta_calc, axis=0), axis=0))
            feature_vector_list.append(np.mean(np.gradient(onesample[:, :, c_coord], axis=0), axis=0))
        else:
            abc = 0

        if acceleration_flag in [True]:
            feature_vector_list.append(np.mean(np.gradient(np.gradient(radius_calc, axis=0), axis=0), axis=0))
            feature_vector_list.append(np.mean(np.gradient(np.gradient(theta_calc, axis=0), axis=0), axis=0))
            feature_vector_list.append(np.mean(np.gradient(np.gradient(onesample[:, :, y_coord], axis=0), axis=0), axis=0))
        else:
            abc = 0

    feature_vector = np.asarray(feature_vector_list)
    feature_vector = feature_vector.astype(float)
    feature_vector = feature_vector.ravel()

    return feature_vector


# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136]
