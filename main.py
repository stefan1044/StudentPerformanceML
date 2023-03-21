import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

from IPython import display
from dataReader import read_dataset
from model import *

mpl.use('TkAgg')
plt.ion()

if __name__ == "__main__":
    print("Ran from main.py")

    _model = LinearRegressionModel(0.001)
    # _model.load_state_dict(torch.load("models/test.pth"))

    _data = read_dataset("StudentsPerformance.csv")
    random.shuffle(_data)
    LEN_TRAINING_DATA = int(len(_data) * 0.7)
    LEN_VALIDATION_DATA = int(len(_data) * 0.2)

    _train_data = _data[:LEN_TRAINING_DATA]
    _validation_data = _data[LEN_TRAINING_DATA:LEN_TRAINING_DATA + LEN_VALIDATION_DATA]
    _test_data = _data[LEN_TRAINING_DATA + LEN_VALIDATION_DATA:]
    _train_loss, _train_epoch_data, _validation_loss, _validation_epoch_data, _test_loss = \
        start_training(30, _model, _train_data, _validation_data, _test_data)

    display.clear_output(wait=True)

    plt.plot(_train_epoch_data, _train_loss)
    plt.plot(_validation_epoch_data, _validation_loss)

    print(f"Average testing loss is {sum(_test_loss)/len(_test_data)}")
    save_model("test", _model)
    plt.pause(100)
