import numpy as np


class DataHandler:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

        self.trn_standardise_params, self.trn_normalise_params = self.obtain_std_nrm_paras(self.train_data)
        self.tst_standardise_params, self.tst_normalise_params = self.obtain_std_nrm_paras(self.test_data)

    def obtain_std_nrm_paras(self, data):

        """
        Abbreviation for obtain standardisation and normalisation parameters.
        Calculates the mean, std, min and max of data array along the axis
        containing the number of features. It stores these values as
        attributes of the class so that they can be used to reverse the effects
        of normalisation and standardisation
        """

        means, stds = [], []
        mins, maxs = [], []

        for i in range(data.shape[1]):
            min_, max_ = np.min(data[:, i]), np.max(data[:, i])
            mean, std = np.mean(data[:, i], dtype=np.float64), np.std(data[:, i], dtype=np.float64)

            mins.append(min_)
            maxs.append(max_)

            means.append(mean)
            stds.append(std)

        standardise_params = [means, stds]
        normalise_params = [mins, maxs]

        return standardise_params, normalise_params

    def normalise(self, datatype):
        """
        Performs data normalisation (scaling such that the scaled data has max=1
        and min=1) along each feature axis.
        """

        mins, maxs = self.trn_normalise_params
        if datatype == "train":
            data = self.train_data
            print("training_data")

        if datatype == "test":
            data = self.test_data

        normalised = np.zeros((data.shape))
        for i in range(data.shape[1]):
            normalised[:, i] = (2 * (data[:, i] - mins[i]) / (maxs[i] - mins[i])) - 1

        return normalised

    def standardise(self, datatype):
        """
        Performs data standardisation ( scaling such that the scaled data has
        mean=0 and std=1) along each feature axis.
        """

        means, stds = self.trn_standardise_params

        if datatype == "train":
            data = self.training_data

        if datatype == "test":
            data = self.test_data

        standardised = np.zeros((data.shape))

        for i in range(data.shape[1]):
            standardised[:, i] = (data[:, i] - means[i]) / stds[i]

        return standardised

    def un_normalise(self, normalised_data):
        """
        Reverses the effect of normalise() using only the statistics of the training
        set as calculated using obtain_std_nrm_pars()
        """
        mins, maxs = self.trn_normalise_params
        unnormalised = np.zeros((normalised_data.shape))

        for i in range(normalised_data.shape[1]):
            unnormalised[:, i] = (normalised_data[:, i] + 1) / 2 * (maxs[i] - mins[i]) + mins[i]

        return unnormalised

    def un_standardise(self, standardised_data):
        """
        Reverses the effect of standardise() using only the statistics of the training
        set as calculated using obtain_std_nrm_pars()
        """
        means, stds = self.trn_standardise_params

        unstandardised = np.zeros((standardised_data.shape))
        for i in range(standardised_data.shape[1]):
            unstandardised[:, i] = (standardised_data[:, i] * stds[i]) + means[i]

        return unstandardised
