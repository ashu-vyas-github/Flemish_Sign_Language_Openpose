import random
import numpy as np
import pandas as pd
from  sklearn.utils import shuffle
from sklearn.base import BaseEstimator, TransformerMixin

def scale(scale_x_factor, scale_y_factor):

    scaling_array = np.array([[scale_x_factor, 0], [0, scale_y_factor]])
    return scaling_array



def flip_leftright():

    leftright = scale(-1, 1)
    return leftright



def flip_updown():

    updown = scale(1, -1)
    return updown



def rotate(theta):

    sin_val, cos_val = np.sin(theta), np.cos(theta)
    rotation_array = np.array([[cos_val, -sin_val], [sin_val, cos_val]])
    return rotation_array



def shear_horizontal(h):

    shear_x = np.array([[1, h], [0, 1]])
    return shear_x



def shear_vertical(h):

    shear_y = np.array([[1, 0], [h, 1]])
    return shear_y



def barrel():
    """
    https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion#6227310
    """
    # implemented in else-if condition below for vectorized approach
    return 0



def data_augmentation(original_samples, original_labels, original_groups, augmentation_factor=2, x_coord=0, y_coord=1, c_coord=2):

    """
     Pass original_samples after interpolation as a constant shape of (n_samples, NUM_FRAMES, n_keypoints, xyc)
    """
    org_num_samples, org_num_frames, org_num_keypoints, org_num_coords = np.asarray(original_samples).shape
    aug_num_samples, aug_num_frames, aug_num_keypoints, aug_num_coords = int(org_num_samples*(augmentation_factor-1)), org_num_frames, org_num_keypoints, org_num_coords

    augmented_samples = np.zeros((aug_num_samples, aug_num_frames, aug_num_keypoints, aug_num_coords))

    for one_sample_idx in range(org_num_samples):

        random_scale_magnitude = random.uniform(-2.0, 2.0)
        random_angle = random.uniform(-np.pi, np.pi)

        augmentation_functions = [scale(random_scale_magnitude, random_scale_magnitude), flip_leftright(), flip_updown(), rotate(random_angle), shear_horizontal(random_scale_magnitude), shear_vertical(random_scale_magnitude), barrel()]
        len_aug_func = len(augmentation_functions)
        random_idx = random.randrange(0, len_aug_func, 1)

        if random_idx in range(len(augmentation_functions) - 1):
            aug_func_mat = augmentation_functions[random_idx]

            augmented_samples[one_sample_idx,:,:,0:2] = np.matmul(original_samples[one_sample_idx,:,:,0:2], aug_func_mat)

        elif random_idx in [len(augmentation_functions)]:

            theta_calc = np.arctan2(original_samples[one_sample_idx,:,:,x_coord], original_samples[one_sample_idx,:,:,y_coord])
            radius_undistort_squared = np.power(original_samples[one_sample_idx,:,:,x_coord], 2) + np.power(original_samples[one_sample_idx,:,:,y_coord], 2)

            augmented_samples[one_sample_idx,:,:,x_coord] = np.sqrt(radius_undistort_squared)(1 - random_scale_magnitude*radius_undistort_squared)*np.cos(theta_calc)
            augmented_samples[one_sample_idx,:,:,y_coord] = np.sqrt(radius_undistort_squared)(1 - random_scale_magnitude*radius_undistort_squared)*np.sin(theta_calc)

    total_samples, total_labels, total_groups =  shuffle(np.vstack([original_samples, augmented_samples]), np.tile(original_labels, augmentation_factor), np.tile(original_groups, augmentation_factor), random_state=42)
    print("Total data shape after augmentation",total_samples.shape)

    return total_samples, total_labels, total_groups




class SLRImbAugmentation(TransformerMixin, BaseEstimator):
    """
    Data augmentation based on above methods
    """

    def __init__(self, *, augmentation_factor=2):
        # self.original_samples = original_samples
        # self.original_labels = original_labels
        self.augmentation_factor = augmentation_factor

    def fit(self, X, y, groups, augmentation_factor=2):
        X_dataframe = resample_data(X, y, groups)
        X = X_dataframe['Data'].tolist()
        X = np.asarray(X)
        y = X_dataframe['Label'].tolist()
        y = np.asarray(y)
        groups = X_dataframe['Gloss'].tolist()
        groups = np.asarray(groups)
        self.total_samples, self.total_labels, self.total_groups = data_augmentation(X, y, groups, augmentation_factor=augmentation_factor)
        return self.total_samples, self.total_labels, self.total_groups

    def transform(self, X):
        # No augmentation of X_valid or X_test
        return X




def resample_data(X, y, groups, min_max_class_deviation=0.5):

    X_dataframe = pd.DataFrame(data=y, columns=['Label'])
    X_dataframe['Gloss'] = groups
    X = X.tolist()
    X_dataframe['Data'] = X_dataframe['Label'].apply(lambda sample: X[sample])

    resampled_groups = []
    group_by_class = X_dataframe.groupby('Gloss')
    class_counts = group_by_class.count()['Data']
    mean_ct, std_ct = class_counts.mean(), class_counts.std()


    for class_id, group in group_by_class:
        count = group.count()['Data']
        if count < mean_ct - min_max_class_deviation * std_ct:
            resampled_groups.append(group.sample(n=int(mean_ct - min_max_class_deviation * std_ct), replace=True))
        elif count > mean_ct + min_max_class_deviation * std_ct:
            resampled_groups.append(group.sample(n=int(mean_ct + min_max_class_deviation * std_ct), replace=False))
        else:
            resampled_groups.append(group)

    # return pd.concat(resampled_groups).sort_values(by=['Gloss']).reset_index(drop=True)
    X_dataframe = pd.concat(resampled_groups).sort_values(by=['Gloss']).reset_index(drop=True)
    X = X_dataframe['Data'].tolist()
    X = np.asarray(X)
    y = X_dataframe['Label'].tolist()
    y = np.asarray(y)
    groups = X_dataframe['Gloss'].tolist()
    groups = np.asarray(groups)

    return X, y, groups

