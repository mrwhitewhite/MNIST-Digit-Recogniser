# import all the necessary libraries and modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import pickle
from typing import Literal

import matplotlib.pyplot as plt

from script.object.CNN import CNN as CNN
from script.object.Data import Data as Data
from script.object.model import model as model
from script.lib.show import show as show

# set seed for reproducibility
seed = 123

# set Hyperparameters
_cf = 8

# load data
train_path = "../../data/train.csv"
test_path = "../../data/test.csv"
submission_path = "../../data/sample_submission.csv"

d = Data(
    train_path,
    test_path,
    submission_path,
    cv1_size=0.1,
    cv2_size=0.1,
)

# train model
m = model(d)
m.fit(
    mode="train",
    epochs=32,
    iteration=1000,
    lr=0.01,
    lr_patience=100,
    lr_gamma=0.1,
    print_interval=1000,
)

# save model
pickle.dump(m, open("../model/model.pkl", "wb"))

# print success message
print("model trained and saved successfully!")
input("Press enter to close the program")