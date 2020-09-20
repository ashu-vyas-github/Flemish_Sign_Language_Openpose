import numpy as np
import matplotlib.pyplot as plt
import util.helpers as H
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import accuracy_score, plot_confusion_matrix
import time
import itertools


SAVE_DIR = r'C:\Users\AVyasWin10EE\Desktop\Desktop_moved\Machine_Learning_Aug2020\sign_language_recognition\code\plots_results'
plt.rcParams.update({'font.size':6})
dpi_setting=1200

def map3_scorer(estimator, X, y_true):
    y_predicted_probabilities = estimator.predict_proba(X)
    map3_score_value = H.mapk(y_predicted_probabilities, y_true)
    return map3_score_value



def plot_learning_curve(estimator, X, y, groups=None, score_met=map3_scorer, title=None, shuffle=False, ylim=None, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 7), save_plot=True):

    plt.figure(num=None, figsize=None, dpi=dpi_setting, facecolor='w', edgecolor='w')
    plt.title("Learning Curve for "+title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training samples")
    plt.ylabel('map@3 score')

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, groups=groups, scoring=score_met, cv=cv,  shuffle=shuffle, n_jobs=n_jobs, train_sizes=train_sizes, return_times=False)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")

    if save_plot in [True]:
        fname = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(SAVE_DIR+"\learn_{txt1}.png".format(txt1=fname), dpi=dpi_setting, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format='png', transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
    else:
        print("Plot not saved")

    plt.show()
    return 0




def plot_validation_curve(estimator, X, y, param_name, param_range, ylim=None, groups=None, cv=None, xlog=False, save_plot=True):

    plt.figure(num=None, figsize=None, dpi=dpi_setting, facecolor='w', edgecolor='w')
    plt.title("Validation Curve for "+str(estimator))
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(param_name)
    plt.ylabel("map@3 score")
    train_scores, valid_scores = validation_curve(estimator, X, y, param_name, param_range, groups=groups, cv=cv, n_jobs=-1, scoring=map3_scorer)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(param_range, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color="g")
    plt.plot(param_range, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(param_range, valid_scores_mean, 'o-', color="g", label="Cross-validation score")

    if xlog in [True]:
        plt.xscale('log')

    plt.legend(loc="best")
    if save_plot in [True]:
        fname = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(SAVE_DIR+"\valid_{txt1}.png".format(txt1=fname), dpi=dpi_setting, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format='png', transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
    else:
        print("Plot not saved")

    plt.show()
    return 0




def plot_con_mat(estimator, X_test, y_test, labels=None, display_labels=None, xticks_rotation='horizontal', save_plot=True):

    plot_confusion_matrix(estimator, X_test, y_test, labels=labels, sample_weight=None, normalize=None, display_labels=display_labels, include_values=True, xticks_rotation=xticks_rotation, values_format=None, cmap='viridis', ax=None)

    plt.title("Absolute Confusion Matrix")

    if save_plot in [True]:
        cmtype = 'CM_abs'
        fname = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(SAVE_DIR+"\{txt1}_{txt2}.png".format(txt1=cmtype, txt2=fname), dpi=dpi_setting, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format='png', transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
    else:
        print("Plot not saved")
    plt.show()


    plot_confusion_matrix(estimator, X_test, y_test, labels=labels, sample_weight=None, normalize='true', display_labels=display_labels, include_values=True, xticks_rotation=xticks_rotation, values_format=None, cmap='viridis', ax=None)
    plt.title("Normalized Confusion Matrix")

    if save_plot in [True]:
        cmtype = 'CM_nrm'
        fname = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(SAVE_DIR+"\{txt1}_{txt2}.png".format(txt1=cmtype, txt2=fname), dpi=dpi_setting, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format='png', transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
    else:
        print("Plot not saved")
    plt.show()

    return 0




def show_save_plots(estimator, X, y, vld_param_name, vld_param_range, cm, classes, crs_vld_obj=5, shuffle=False, ylim=None, groups=None, cv=None, xlog=False, normalize=False, learn=True, valid=True, conmat=True, save_plot=True):

    if learn in [True]:
        lrn = plot_learning_curve(estimator, X, y, groups=groups, cv=crs_vld_obj, shuffle=shuffle, ylim=ylim, save_plot=save_plot)
    else:
        print("No learning curve")

    if valid in [True]:
        vld = plot_validation_curve(estimator, X, y, vld_param_name, vld_param_range, ylim=ylim, groups=groups, cv=crs_vld_obj, xlog=xlog, save_plot=save_plot)
    else:
        print("No validation curve")

    if conmat in [True]:
        cmt = plot_confusion_matrix(cm, classes, normalize=normalize, cmap=plt.cm.Blues, save_plot=save_plot)
    else:
        print("No confusion matrix")

    return 0



def predict_print_results(estimator, X_train, X_valid, y_train, y_valid):

    ytrn_pred = estimator.predict(X_train)
    yvld_pred = estimator.predict(X_valid)
    ytrn_pred_proba = estimator.predict_proba(X_train)
    yvld_pred_proba = estimator.predict_proba(X_valid)

    accur_trn = accuracy_score(y_train, ytrn_pred)
    accur_vld = accuracy_score(y_valid, yvld_pred)

    map3_trn = H.mapk(ytrn_pred_proba, y_train)
    map3_vld =  H.mapk(yvld_pred_proba, y_valid)

    top3_trn = H.top3_accuracy(ytrn_pred_proba, y_train)
    top3_vld = H.top3_accuracy(yvld_pred_proba, y_valid)

    # Show the accuracy obtained on the training set
    print("")
    print('Training set accuracy:   ', accur_trn)
    print('Validation set accuracy: ', accur_vld)
    print("")
    print('Training set score (map@3):', map3_trn)
    print('Validation set score (map@3):', map3_vld)
    print("")
    print('Training set top-3 accuracy:', top3_trn)
    print('Validation set top-3 accuracy:', top3_vld)

    return map3_trn, map3_vld



# def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues, save_plot=True):

#     plt.figure(num=None, figsize=None, dpi=dpi_setting, facecolor='w', edgecolor='w')
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         plt.title('Confusion matrix normalized')
#         fmt = '.2f'
#         cmtype = 'CM_nrm'
#     else:
#         plt.title('Confusion matrix')
#         fmt = 'd'
#         cmtype = 'CM_abs'

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=90)
#     plt.yticks(tick_marks, classes)

#     thresh = cm.max() / 2.0
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()

#     if save_plot in [True]:
#         fname = time.strftime("%Y%m%d-%H%M%S")
#         plt.savefig(SAVE_DIR+"\{txt1}_{txt2}.png".format(txt1=cmtype, txt2=fname), dpi=dpi_setting, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format='png', transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
#     else:
#         print("Plot not saved")

#     plt.show()
#     return 0