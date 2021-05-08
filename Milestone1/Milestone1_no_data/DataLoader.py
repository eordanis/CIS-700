import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class DataLoader:

    def __init__(self):
        pass

    def get_ds_infos(self):
        ## 0:Code, 1:Weight, 2:Height, 3:Age, 4:Gender
        dss = np.genfromtxt("MotionSense/data_subjects_info.csv", delimiter=',')
        dss = dss[1:]
        print("----> Data subjects information is imported.")
        return dss

    def extract_from_csv(self, features, activities, verbose):

        num_features = len(features)
        num_act_labels = len(activities)
        dataset_columns = num_features + num_act_labels

        label_codes = {"dws": num_features, "ups": num_features + 1, "wlk": num_features + 2, "jog": num_features + 3,
                       "sit": num_features + 4, "std": num_features + 5}
        trial_codes = {"dws": [1, 2, 11], "ups": [3, 4, 12], "wlk": [7, 8, 15], "jog": [9, 16], "sit": [5, 13],
                       "std": [6, 14]}

        new = {}

        for requested in trial_codes:
            if requested in activities:
                new[requested] = trial_codes[requested]

        trial_codes = new
        label_codes = {}
        count = 0
        for key in trial_codes:
            label_codes[key] = num_features + count
            count += 1

        if verbose == True:
            print(label_codes)
            print(trial_codes)

        ds_list = self.get_ds_infos()

        train_data = np.zeros((0, dataset_columns))
        test_data = np.zeros((0, dataset_columns))

        for i, sub_id in enumerate(ds_list[:, 0]):
            for j, act in enumerate(trial_codes):
                for trial in trial_codes[act]:
                    fname = 'MotionSense/A_DeviceMotion_data/' + act + '_' + str(trial) + '/sub_' + str(
                        int(sub_id)) + '.csv'
                    if verbose == True:
                        print("Loading: ", fname)
                    raw_data = pd.read_csv(fname)
                    raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                    raw_data = raw_data[features]
                    unlabel_data = raw_data.values

                    label_data = np.zeros((len(unlabel_data), dataset_columns))
                    label_data[:, :-(num_act_labels)] = unlabel_data
                    label_data[:, label_codes[act]] = 1
                    if trial > 10:
                        test_data = np.append(test_data, label_data, axis=0)
                    else:
                        train_data = np.append(train_data, label_data, axis=0)

        return train_data, test_data, num_features, num_act_labels

    def time_series_to_section(self, dataset, num_act_labels, sliding_window_size, step_size_of_sliding_window):
        data = dataset[:, 0:-(num_act_labels)]
        act_labels = dataset[:, -(num_act_labels):]

        data = data.T

        size_features = data.shape[0]
        size_data = data.shape[1]

        number_of_secs = round(((size_data - sliding_window_size) / step_size_of_sliding_window))

        secs_data = np.zeros((number_of_secs, size_features, sliding_window_size))
        act_secs_labels = np.zeros((number_of_secs, num_act_labels))

        k = 0
        for i in range(0, (size_data) - sliding_window_size, step_size_of_sliding_window):
            j = i // step_size_of_sliding_window
            if (j >= number_of_secs):
                break
            if (not (act_labels[i] == act_labels[i + sliding_window_size - 1]).all()):
                continue
            secs_data[k] = data[0:size_features, i:i + sliding_window_size]
            act_secs_labels[k] = act_labels[i].astype(int)
            k = k + 1
        secs_data = secs_data[0:k]
        secs_data = np.expand_dims(secs_data, axis=3)
        act_secs_labels = act_secs_labels[0:k]

        return secs_data, act_secs_labels




