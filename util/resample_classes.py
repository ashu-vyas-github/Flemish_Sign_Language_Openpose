import pandas as pd
def resample_classes(data, variation_rate=1/2):
    """Resample classes to be more balanced
    Overrepresented classes will be undersampled, while underrepresented classes will be oversampled
    Parameters:
    -----------
    data : DataFrame
        The data to be resampled
    variation_rate : float
        The number of standard deviations that the number of samples should deviate from the mean number of samples
        per class.
    Returns:
    --------
    A new dataframe containing the resampled classes
    """

    group_by_class = data.groupby('Gloss')

    class_count = group_by_class.count()['File']
    mean, std = class_count.mean(), class_count.std()

    new_groups = []
    for class_id, group in group_by_class:
        count = group.count()['File']
        if count < mean - variation_rate * std:
            new_groups.append(group.sample(
                n=int(mean - variation_rate * std), replace=True))
        elif count > mean + variation_rate * std:
            new_groups.append(group.sample(
                n=int(mean + variation_rate * std), replace=False))
        else:
            new_groups.append(group)

    return pd.concat(new_groups) \
        .sort_values(by=['Gloss']) \
        .reset_index(drop=True)