import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':6})

def plot_distribution(x_axis_data, y_axis_data, title_text, x_text, y_text, x_ticks_text, fname, xlim=None, ylim=None, bar_width=0.3, dpi_setting=120, rotation=90, save_plot=True):

    plt.figure(num=None, figsize=None, dpi=dpi_setting, facecolor='w', edgecolor='w')
    plt.title(title_text)
    plt.xlabel(x_text)
    plt.ylabel(y_text)
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.bar(x=np.arange(len(x_axis_data)), height=y_axis_data, width=bar_width)
    plt.xticks(ticks=np.arange(len(x_axis_data)), labels=x_ticks_text, rotation=rotation)

    plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)

    if save_plot in [True]:
        plt.savefig("{txt1}.png".format(txt1=fname), dpi=dpi_setting, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format='png', transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)

    plt.show()



def plot_2D_heatmap(array_to_plot, title_text, x_text, y_text, x_ticks_text, y_ticks_text, fname, dpi_setting=120, save_plot=True):

    plt.figure(num=None, figsize=None, dpi=dpi_setting, facecolor='w', edgecolor='w')
    plt.title(title_text)
    plt.xlabel(x_text)
    plt.ylabel(y_text)
    plt.xticks(ticks=np.arange(len(x_ticks_text)), labels=x_ticks_text, rotation=90)
    plt.yticks(ticks=np.arange(len(y_ticks_text)), labels=y_ticks_text)
    plt.imshow(array_to_plot, cmap='viridis', norm=None, aspect='auto', interpolation='nearest', alpha=None, vmin=None, vmax=None, origin=None, extent=None, filternorm=True, filterrad=4.0, resample=None, url=None, data=None)
    plt.colorbar()

    if save_plot in [True]:
        plt.savefig("{txt1}.png".format(txt1=fname), dpi=dpi_setting, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format='png', transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)

    plt.show()

def main_plot_function(full_dataframe, label_dist=True, person_dist=True, frame_dist=True, frames_class=True, frames_person=True, person_class=True, save_plot=True):

    bar_width = 0.3
    dpi_setting = 1200

    samples_list = full_dataframe.Data
    num_samples = len(samples_list)
    label_list = np.asarray(full_dataframe.Label)
    words_list = full_dataframe.Gloss
    person_list = np.asarray(full_dataframe.Person)

    if label_dist in [True]:
        uqlabel, ctlabel = np.unique(words_list, return_counts=True)
        plot_distribution(uqlabel, ctlabel, title_text='Class label distribtution', x_text='Class label', y_text='Frequency count', x_ticks_text=uqlabel, fname='class_label_dist', xlim=None, ylim=None, bar_width=bar_width, dpi_setting=dpi_setting, rotation=90, save_plot=save_plot)
    else:
        print("No plot")

    if person_dist in [True]:
        uqperson, ctperson= np.unique(person_list, return_counts=True)
        plot_distribution(uqperson, ctperson, title_text='Person distribtution', x_text='Person', y_text='Frequency count', x_ticks_text=uqperson, fname='person_dist', xlim=None, ylim=None, bar_width=0.5, dpi_setting=dpi_setting, rotation=0, save_plot=save_plot)
    else:
        print("No plot")

    if frame_dist in [True]:
        frames_list = []
        for onesample, onesample_idx in zip(samples_list, range(num_samples)):
            num_frames, num_keypoints, xyc = onesample.shape
            frames_list.append(num_frames)

        print(sum(frames_list))
        uqframe, ctframe = np.unique(frames_list, return_counts=True)
        print(sum(ctframe))
        plot_distribution(uqframe, ctframe, title_text='Frames distribtution', x_text='#Frames', y_text='Frequency count', x_ticks_text=uqframe, fname='frame_dist', xlim=None, ylim=None, bar_width=0.7, dpi_setting=dpi_setting, rotation=90, save_plot=save_plot)
    else:
        print("No plot")


    if frames_class in [True]:
        frames_list = []
        for onesample, onesample_idx in zip(samples_list, range(num_samples)):
            num_frames, num_keypoints, xyc = onesample.shape
            frames_list.append(num_frames)
        uqframe = np.unique(frames_list)
        uqlabel = np.unique(words_list)
        frame_class_array = np.zeros((uqlabel.shape[0], uqframe.shape[0]))
        for onesample, onelabel in zip(samples_list, label_list):
            num_frames, _, _ = onesample.shape
            index_tuple = np.where(uqframe == num_frames)
            frame_class_array[onelabel, index_tuple[0]] = frame_class_array[onelabel, index_tuple[0]] + 1.0

        plot_2D_heatmap(frame_class_array, title_text='Frame length per class', x_text='Frame length', y_text='Class label', x_ticks_text=uqframe, y_ticks_text=uqlabel, fname='frames_class_2D', dpi_setting=dpi_setting, save_plot=save_plot)
    else:
        print("No plot")


    if frames_person in [True]:
        frames_list = []
        for onesample, onesample_idx in zip(samples_list, range(num_samples)):
            num_frames, num_keypoints, xyc = onesample.shape
            frames_list.append(num_frames)
        uqframe = np.unique(frames_list)
        uqperson = np.unique(person_list)
        frame_person_array = np.zeros((uqperson.shape[0], uqframe.shape[0]))
        for onesample, oneperson in zip(samples_list, person_list):
            num_frames, _, _ = onesample.shape
            index_tuple = np.where(uqframe == num_frames)
            frame_person_array[oneperson-1, index_tuple[0]] = frame_person_array[oneperson-1, index_tuple[0]] + 1.0

        plot_2D_heatmap(frame_person_array, title_text='Frame length per person', x_text='Frame length', y_text='Person label', x_ticks_text=uqframe, y_ticks_text=uqperson, fname='frames_person_2D', dpi_setting=dpi_setting, save_plot=save_plot)
    else:
        print("No plot")


    if person_class in [True]:
        uqlabel = np.unique(words_list)
        uqperson = np.unique(person_list)
        person_class_array = np.zeros((uqlabel.shape[0], uqperson.shape[0]))
        for onelabel, oneperson in zip(label_list, person_list):
            person_class_array[onelabel, oneperson-1] = person_class_array[onelabel, oneperson-1] + 1.0

        plot_2D_heatmap(person_class_array, title_text='Person per class', x_text='Person label', y_text='Class label', x_ticks_text=uqperson, y_ticks_text=uqlabel, fname='person_class_2D', dpi_setting=dpi_setting, save_plot=save_plot)
    else:
        print("No plot")



def check_undetected_keypoints_allframes(samples_list, threshold=1e-1, dpi_setting=120, save_plot=True):

    bar_width = 0.5
    undetected_keypoints = []
    x_coord = 0
    y_coord = 1
    count = 0.0

    for onesample in samples_list:
        for oneframe_idx in range(onesample.shape[0]):
            for one_kp_idx in range(onesample.shape[1]):
                count = count + 1.0
                if (onesample[oneframe_idx, one_kp_idx, x_coord] < threshold) and (onesample[oneframe_idx, one_kp_idx, y_coord] < threshold):
                    undetected_keypoints.append(one_kp_idx)
                else:
                    continue

    print("Total undetected keypoints", len(undetected_keypoints),"out of",int(count))
    print("About {txt}%".format(txt=round(100.0*len(undetected_keypoints)/count, 4)))

    uqkps, ctkps = np.unique(undetected_keypoints, return_counts=True)
    plot_distribution(uqkps, ctkps, title_text="Undetected keypoints distribution", x_text="Keypoints", y_text="Zero freq. counts", x_ticks_text=uqkps, fname="undet_full137_frames", xlim=None, ylim=None, bar_width=0.3, dpi_setting=dpi_setting, rotation=90, save_plot=save_plot)

    body_list = []
    face_list = []
    hands_list = []

    for one_kp in undetected_keypoints:
        if one_kp <25:
            body_list.append(one_kp)
        elif (one_kp >= 25) and (one_kp < 95):
            face_list.append(one_kp)
        else:
            hands_list.append(one_kp)

    uqkps, ctkps = np.unique(body_list, return_counts=True)
    plot_distribution(uqkps, ctkps, title_text="Undetected keypoints for body-25", x_text="Keypoints", y_text="Zero freq. counts", x_ticks_text=uqkps, fname="undet_body_frames", xlim=None, ylim=None, bar_width=bar_width, dpi_setting=dpi_setting, rotation=90, save_plot=save_plot)

    uqkps, ctkps = np.unique(face_list, return_counts=True)
    plot_distribution(uqkps, ctkps, title_text="Undetected keypoints for face", x_text="Keypoints", y_text="Zero freq. counts", x_ticks_text=uqkps, fname="undet_face_frames", xlim=None, ylim=None, bar_width=bar_width, dpi_setting=dpi_setting, rotation=90, save_plot=save_plot)

    uqkps, ctkps = np.unique(hands_list, return_counts=True)
    plot_distribution(uqkps, ctkps, title_text="Undetected keypoints for both hands", x_text="Keypoints", y_text="Zero freq. counts", x_ticks_text=uqkps, fname="undet_hands_frames", xlim=None, ylim=None, bar_width=bar_width, dpi_setting=dpi_setting, rotation=90, save_plot=save_plot)



def check_undetected_keypoints_allsamples(samples_list, threshold=1e-1, dpi_setting=120, save_plot=True):

    bar_width = 0.5
    undetected_keypoints = []
    x_coord = 0
    y_coord = 1
    count = 0.0

    for onesample in samples_list:
        for one_kp_idx in range(onesample.shape[1]):
            count = count + 1.0
            if (np.mean(onesample[:, one_kp_idx, x_coord], axis=0) < threshold) and (np.mean(onesample[:, one_kp_idx, y_coord], axis=0) < threshold):
                undetected_keypoints.append(one_kp_idx)
            else:
                continue

    print("Total undetected keypoints", len(undetected_keypoints))

    uqkps, ctkps = np.unique(undetected_keypoints, return_counts=True)
    plot_distribution(uqkps, ctkps, title_text="Undetected keypoints distribution per sample", x_text="Keypoints", y_text="Zero freq. counts", x_ticks_text=uqkps, fname="undet_full137_samples", xlim=None, ylim=None, bar_width=0.3, dpi_setting=dpi_setting, rotation=90, save_plot=save_plot)

    body_list = []
    face_list = []
    hands_list = []

    for one_kp in undetected_keypoints:
        if one_kp <25:
            body_list.append(one_kp)
        elif (one_kp >= 25) and (one_kp < 95):
            face_list.append(one_kp)
        else:
            hands_list.append(one_kp)

    uqkps, ctkps = np.unique(body_list, return_counts=True)
    plot_distribution(uqkps, ctkps, title_text="Undetected keypoints for body-25  per sample", x_text="Keypoints", y_text="Zero freq. counts", x_ticks_text=uqkps, fname="undet_body_samples", xlim=None, ylim=None, bar_width=bar_width, dpi_setting=dpi_setting, rotation=90, save_plot=save_plot)

    uqkps, ctkps = np.unique(face_list, return_counts=True)
    plot_distribution(uqkps, ctkps, title_text="Undetected keypoints for face per sample", x_text="Keypoints", y_text="Zero freq. counts", x_ticks_text=uqkps, fname="undet_face_samples", xlim=None, ylim=None, bar_width=bar_width, dpi_setting=dpi_setting, rotation=90, save_plot=save_plot)

    uqkps, ctkps = np.unique(hands_list, return_counts=True)
    plot_distribution(uqkps, ctkps, title_text="Undetected keypoints for both hands per sample", x_text="Keypoints", y_text="Zero freq. counts", x_ticks_text=uqkps, fname="undet_hands_samples", xlim=None, ylim=None, bar_width=bar_width, dpi_setting=dpi_setting, rotation=90, save_plot=save_plot)











