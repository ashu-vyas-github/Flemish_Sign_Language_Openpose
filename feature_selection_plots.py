# import numpy as np
# import pandas as pd
# from datetime import datetime
# import matplotlib.pyplot as plt
# from os.path import join as pjoin
# import util.vis as V
# import util.helpers as H
# import data_analysis

# import csv
# import random
# import gc
# from glob import glob
# import sklearn as sk
# from sklearn import preprocessing

# import feature_engineering.feature_preprocessing as feat_prepro
# import feature_engineering.feature_extractors_4D_array as feat_extract
# from feature_engineering.data_augmentation import SLRImbAugmentation
# import feature_engineering.data_augmentation as data_augm
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GroupKFold
# from util.stratified_group_cv import StratifiedGroupKFold
# from sklearn.feature_selection import SelectKBest

# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV

# from util.results_plots_evaluation import map3_scorer
# import util.results_plots_evaluation as results
# from sklearn.metrics import accuracy_score
# import util.helpers as kaggle_submission


# from sklearn.linear_model import LogisticRegression

# np.seterr(all='raise', divide='raise', over='raise', under='raise', invalid='raise')
# rng = np.random.RandomState(42)
# startTime= datetime.now()

# n_splits = 5
# remove_keypoints = True
# save_plot = False

# unwanted_keypoints=[10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94]

# face_body_hand = [True, True, True, False, False, False, False, False, False, False]
# physics1 = [False, False, False, True, False, True, False, False, False, False]
# physics2 = [False, False, False, True, False, True, True, False, False, False]
# physics3 = [False, False, False, True, False, True, False, False, True, False]
# physics4 = [False, False, False, True, False, True, False, False, False, True]
# physics5 = [False, False, False, True, False, False, False, True, False, False]
# physics6 = [False, False, False, True, False, False, True, True, False, False]
# physics7 = [False, False, False, True, False, False, False, True, True, False]
# physics8 = [False, False, False, True, False, False, False, True, False, True]
# trajectory = [False, False, False, False, True, False, False, False, False, False]
# all_feat = [True, True, True, True, True, True, True, True, True, True]

# # PATHS
# DATA_DIR = '../data'
# POSE_DIR = '../data/pose'
# TRAIN_DIR = POSE_DIR + "/train"
# TEST_DIR = POSE_DIR + "/test"

# # Read CSV file of labels
# full_dataframe = pd.read_csv(pjoin(DATA_DIR, "labels.csv"))
# full_dataframe['Data'] = full_dataframe['File'].apply(lambda title: np.load(pjoin(TRAIN_DIR, title + ".npy")))

# print("\n~~~~~#####     Start     #####~~~~~\n")

# num_frames_list = [2, 4, 8, 15, 24, 48, 60]

# gscv_best_score_list = []
# validtt_map3_trn = []
# validtt_map3_vld = []

# for interpolated_total_frames in num_frames_list:

#     print("\n\n\nRunning interpolated_total_frames:", interpolated_total_frames)

#     # 4D data as (n_samples, n_frames, n_keypoints, n_coords)
#     samples_centered_4D_array = feat_prepro.interpolate_allsamples(full_dataframe.Data, interpolated_total_frames=interpolated_total_frames, x_resolution=1.0, y_resolution=1.0)

#     print("Interplated data shape",samples_centered_4D_array.shape)
# # Train and test split
#     X_traintt, X_validtt, y_traintt, y_validtt, group_traintt, group_validtt = train_test_split(samples_centered_4D_array, np.asarray(full_dataframe.Label), np.asarray(full_dataframe.Person), test_size=0.25, random_state=42, shuffle=True, stratify=None)#=np.asarray(full_dataframe.Label))


#     face_flag=True
#     body_flag=True
#     hand_flag=True
#     physics_flag=True
#     trajectory_flag=True
#     linear_flag=True
#     std_flag=True
#     angular_flag=False
#     velocity_flag=False
#     acceleration_flag=False


#     X_train = feat_extract.main_feature_extractor(array_4D_data=X_traintt, face=face_flag, body=body_flag, hands=hand_flag, physics=physics_flag, trajectory=trajectory_flag, linear_flag=linear_flag, angular_flag=angular_flag, std_flag=std_flag, velocity_flag=velocity_flag, acceleration_flag=acceleration_flag, remove_keypoints=remove_keypoints, unwanted_keypoints=unwanted_keypoints)


#     X_valid = feat_extract.main_feature_extractor(array_4D_data=X_validtt, face=face_flag, body=body_flag, hands=hand_flag, physics=physics_flag, trajectory=trajectory_flag, linear_flag=linear_flag, angular_flag=angular_flag, std_flag=std_flag, velocity_flag=velocity_flag, acceleration_flag=acceleration_flag, remove_keypoints=remove_keypoints, unwanted_keypoints=unwanted_keypoints)


#     ### Standard Scaler
#     stdscl = StandardScaler()

#     ### Cross validator
#     # cvld = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, train_size=None, random_state=42)
#     # cvld = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     cvld = GroupKFold(n_splits=n_splits)

#     ### Estimator
#     estimator = LogisticRegression(C=1.0, tol=1e-4, class_weight=None, solver='lbfgs', max_iter=5000, multi_class='ovr', penalty='l2', dual=False, fit_intercept=True, intercept_scaling=1, random_state=42, verbose=0, warm_start=False, n_jobs=-1, l1_ratio=None)

#     print("\nTraining the model", str(estimator))

#     pipe = Pipeline([('scale', stdscl), ('clf', estimator)])

#     ### Grid Search CV
#     param_grid = dict(clf__C=np.logspace(-3, 1, 5))

#     print("Running GSCV.....")
#     grid = GridSearchCV(pipe, param_grid=param_grid, cv=cvld, n_jobs=-1, verbose=0, scoring=map3_scorer)
#     grid.fit(X_train, y_traintt, group_traintt)
#     print(grid.best_params_)
#     print(grid.best_score_)
#     gscv_best_score_list.append(grid.best_score_)
#     map3_trn, map3_vld = results.predict_print_results(grid, X_train, X_valid, y_traintt, y_validtt)
#     validtt_map3_trn.append(map3_trn)
#     validtt_map3_vld.append(map3_vld)


# plt.rcParams.update({'font.size':6})
# bar_width = 0.25
# dpi_setting = 1200
# labels = num_frames_list
# fname = 'num_frames_logreg_mean_raw_groupK'


# plt.figure(num=None, figsize=None, dpi=dpi_setting, facecolor='w', edgecolor='w')
# plt.title("Optimum #Frames LogReg GroupK")
# plt.xlabel('Interpolated #Frames')
# plt.ylabel('map@3 score')
# plt.ylim(0.0,1.1)

# plt.bar(x=np.arange(len(gscv_best_score_list))-bar_width, height=gscv_best_score_list, width=bar_width, label='Best GSCV Score', align='center')
# plt.bar(x=np.arange(len(gscv_best_score_list)), height=validtt_map3_trn, width=bar_width, label='Training score', align='center')
# plt.bar(x=np.arange(len(gscv_best_score_list))+bar_width, height=validtt_map3_vld, width=bar_width, label='Test Score', align='center')

# plt.xticks(ticks=np.arange(len(gscv_best_score_list)), labels=labels, rotation=0)
# plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)
# plt.legend()
# plt.savefig("{txt1}.png".format(txt1=fname), dpi=dpi_setting, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format='png', transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
# plt.show()

# print("\n~~~~~#####      Done     #####~~~~~\n")

# timeElapsed = datetime.now() - startTime
# print('Time elpased (hh:mm:ss.ms) {}'.format(timeElapsed))






###############################################################################
###############################################################################
###########          BELOW IS THE CODE FOR FEATURE SELECTION
###############################################################################
###############################################################################

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from os.path import join as pjoin
import util.vis as V
import util.helpers as H
import data_analysis

import csv
import random
import gc
from glob import glob
import sklearn as sk
from sklearn import preprocessing

import feature_engineering.feature_preprocessing as feat_prepro
import feature_engineering.feature_extractors_4D_array as feat_extract
from feature_engineering.data_augmentation import SLRImbAugmentation
import feature_engineering.data_augmentation as data_augm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GroupKFold
from util.stratified_group_cv import StratifiedGroupKFold
from sklearn.feature_selection import SelectKBest

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from util.results_plots_evaluation import map3_scorer
import util.results_plots_evaluation as results
from sklearn.metrics import accuracy_score
import util.helpers as kaggle_submission


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

np.seterr(all='raise', divide='raise', over='raise', under='raise', invalid='raise')
rng = np.random.RandomState(42)
startTime= datetime.now()

interpolated_total_frames = 15
n_splits = 7
face_flag = None
body_flag = None
hand_flag = None

physics_flag = None
trajectory_flag = None

linear_flag = None
std_flag = None

angular_flag = None

velocity_flag = None
acceleration_flag = None

remove_keypoints = False
save_plot = False

unwanted_keypoints=[10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94]

face_body_hand = [True, True, True, False, False, False, False, False, False, False]
physics1 = [False, False, False, True, False, True, False, False, False, False]
physics2 = [False, False, False, True, False, True, True, False, False, False]
physics3 = [False, False, False, True, False, True, False, False, True, False]
physics4 = [False, False, False, True, False, True, False, False, False, True]
physics5 = [False, False, False, True, False, False, False, True, False, False]
physics6 = [False, False, False, True, False, False, True, True, False, False]
physics7 = [False, False, False, True, False, False, False, True, True, False]
physics8 = [False, False, False, True, False, False, False, True, False, True]
trajectory = [False, False, False, False, True, False, False, False, False, False]
all_feat = [True, True, True, True, True, True, True, True, True, True]

features_selection_array = np.array([face_body_hand, physics1, physics2, physics3, physics4, physics5, physics6, physics7, physics8, trajectory, all_feat])


print(features_selection_array.shape)

# PATHS
DATA_DIR = '../data'
POSE_DIR = '../data/pose'
TRAIN_DIR = POSE_DIR + "/train"
TEST_DIR = POSE_DIR + "/test"

# Read CSV file of labels
full_dataframe = pd.read_csv(pjoin(DATA_DIR, "labels.csv"))
full_dataframe['Data'] = full_dataframe['File'].apply(lambda title: np.load(pjoin(TRAIN_DIR, title + ".npy")))

# Resampling and augmentation setp

print("\n~~~~~#####     Start     #####~~~~~\n")

# 4D data as (n_samples, n_frames, n_keypoints, n_coords)
samples_centered_4D_array = feat_prepro.interpolate_allsamples(full_dataframe.Data, interpolated_total_frames=interpolated_total_frames, x_resolution=1.0, y_resolution=1.0)

print("Interpolated data shape",samples_centered_4D_array.shape)
# Train and test split
# X_traintt, X_validtt, y_traintt, y_validtt, group_traintt, group_validtt = train_test_split(samples_centered_4D_array, np.asarray(full_dataframe.Label), np.asarray(full_dataframe.Person), test_size=0.25, random_state=42, shuffle=True, stratify=None)#=np.asarray(full_dataframe.Label))
# print("Training shape 4D split",X_traintt.shape)
# print("Validation shape 4D split",X_validtt.shape)

y_train = np.asarray(full_dataframe.Label)
group_train = np.asarray(full_dataframe.Person)

#### Augmentation
# slr_obj = SLRImbAugmentation()
# X_traintt, y_traintt, group_traintt = slr_obj.fit(X=X_traintt, y=y_traintt, groups=group_traintt, augmentation_factor=2)
# X_traintt, y_traintt, group_traintt = data_augm.resample_data(X=X_traintt, y=y_traintt, groups=group_traintt)

gscv_best_score_list = []
validtt_map3_trn = []
# validtt_map3_vld = []

for one_feature_set_idx in range(features_selection_array.shape[0]):

    print("\n\n\nRunning feature set:", one_feature_set_idx)
    face_flag, body_flag, hand_flag, physics_flag, trajectory_flag, linear_flag, std_flag, angular_flag, velocity_flag, acceleration_flag = features_selection_array[one_feature_set_idx, :].ravel()


    X_train = feat_extract.main_feature_extractor(array_4D_data=samples_centered_4D_array, face=face_flag, body=body_flag, hands=hand_flag, physics=physics_flag, trajectory=trajectory_flag, linear_flag=linear_flag, angular_flag=angular_flag, std_flag=std_flag, velocity_flag=velocity_flag, acceleration_flag=acceleration_flag, remove_keypoints=remove_keypoints, unwanted_keypoints=unwanted_keypoints)


    # X_valid = feat_extract.main_feature_extractor(array_4D_data=X_validtt, face=face_flag, body=body_flag, hands=hand_flag, physics=physics_flag, trajectory=trajectory_flag, linear_flag=linear_flag, angular_flag=angular_flag, std_flag=std_flag, velocity_flag=velocity_flag, acceleration_flag=acceleration_flag, remove_keypoints=remove_keypoints, unwanted_keypoints=unwanted_keypoints)


    ### Standard Scaler
    stdscl = StandardScaler()

    ### Feature selection
    selection = VarianceThreshold(threshold=0.0)

    ### Cross validator
    # cvld = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, train_size=None, random_state=42)
    # cvld = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cvld = GroupKFold(n_splits=n_splits)

    ### Estimator
    # estimator = LogisticRegression(C=1.0, tol=1e-4, class_weight=None, solver='lbfgs', max_iter=5000, multi_class='ovr', penalty='l2', dual=False, fit_intercept=True, intercept_scaling=1, random_state=42, verbose=0, warm_start=False, n_jobs=-1, l1_ratio=None)
    estimator = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovo', break_ties=False, random_state=42)

    print("\nTraining the model", str(estimator))

    pipe = Pipeline([('feat_select', selection), ('scale', stdscl), ('clf', estimator)])

    ### Grid Search CV
    param_grid = dict( clf__C=[0.01, 0.1, 0.5, 1.0, 4.0, 8.0, 10.0])

    print("Running GSCV.....")
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=cvld, n_jobs=-1, verbose=0, scoring=map3_scorer)
    # grid.fit(X_train, y_traintt, group_traintt)
    grid.fit(X_train, y_train, groups=group_train)
    print(grid.best_params_)
    print(grid.best_score_)
    gscv_best_score_list.append(grid.best_score_)
    map3_trn, map3_vld = results.predict_print_results(grid, X_train, X_train, y_train, y_train)
    validtt_map3_trn.append(map3_trn)
    # validtt_map3_vld.append(map3_vld)





plt.rcParams.update({'font.size':6})
bar_width = 0.25
dpi_setting = 1200
labels = [int(x+1) for x in range(features_selection_array.shape[0])]
fname = 'feat_sets_confidence_svcRBFovo_mean_raw_groupK'


plt.figure(num=None, figsize=None, dpi=dpi_setting, facecolor='w', edgecolor='w')
plt.title("Feature set comparison SVC-RBF GroupK")
plt.xlabel('Features set')
plt.ylabel('map@3 score')
plt.ylim(0.0,1.1)

plt.bar(x=np.arange(len(gscv_best_score_list))-bar_width/2.0, height=gscv_best_score_list, width=bar_width, label='Best GSCV Score', align='center')
plt.bar(x=np.arange(len(gscv_best_score_list))+bar_width/2.0, height=validtt_map3_trn, width=bar_width, label='Training score', align='center')
# plt.bar(x=np.arange(len(gscv_best_score_list))+bar_width, height=validtt_map3_vld, width=bar_width, label='Test Score', align='center')

plt.xticks(ticks=np.arange(len(gscv_best_score_list)), labels=labels, rotation=0)
plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)
plt.legend()
plt.savefig("{txt1}.png".format(txt1=fname), dpi=dpi_setting, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format='png', transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
plt.show()

print("\n~~~~~#####      Done     #####~~~~~\n")

timeElapsed = datetime.now() - startTime
print('Time elpased (hh:mm:ss.ms) {}'.format(timeElapsed))








# clf__C=[0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 5.0, 10.0] for Logreg OVR
# clf__C=[0.01, 0.1, 0.5, 1.0, 4.0, 8.0, 10.0] for SVC RBF OVO
#