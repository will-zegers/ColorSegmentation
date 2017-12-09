import numpy as np

class Keys():
    RED = 'barrel'
    OTHER = 'other'

class EMModel():

    def __init__(self, model_1, model_2):
        self.model = {}
        self.model[Keys.RED] = model_1
        self.model[Keys.OTHER] = model_2
        self.prediction = np.array([])

    def predict(self, test_data, features, golden=None):
        prediction = {}
        prediction[Keys.RED] = self.model[Keys.RED].predict(test_data, features)
        prediction[Keys.OTHER] = self.model[Keys.OTHER].predict(test_data, features)

        self.prediction = np.array([prediction[Keys.RED] >= prediction[Keys.OTHER]], dtype='uint8')
        if golden is not None:
            self.compute_error(golden)

    def compute_error(self, golden):
        self.prediction = self.prediction.reshape((golden.shape[1], golden.shape[0])).T
        fg_class_err = sum(self.prediction[golden == 1] == 0) / sum(sum(golden == 1))
        bg_class_err = sum(self.prediction[golden == 0] == 1) / sum(sum(golden == 0))
        total_err = sum(sum(golden != self.prediction)) / self.prediction.size
