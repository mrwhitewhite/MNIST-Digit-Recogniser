# import all the necessary libraries and modules
import os
import pickle
from object.model import Model as Model
from object.Data import Data as Data
from lib.show import show as show

# set seed for reproducibility
seed = 123

# set Hyperparameters
_cf = 16
epochs=32
iteration=100000
lr=0.001
lr_patience=100
lr_gamma=0.5
print_interval=1000

# load data
train_path = os.path.join(os.path.abspath(os.path.join(__file__, "..", "..", "..")), "data", "train.csv")
test_path = os.path.join(os.path.abspath(os.path.join(__file__, "..", "..", "..")), "data", "test.csv")
submission_path = os.path.join(
    os.path.abspath(os.path.join(__file__, "..", "..", "..")), "data", "sample_submission.csv"
)


d = Data(
    train_path,
    test_path,
    submission_path,
    seed=seed,
    cv1_size=0.1,
    cv2_size=0.1,
)

# train model
m = Model(d, _cf, seed)
m.fit(
    mode="train",
    epochs=epochs,
    iteration=iteration,
    lr=lr,
    lr_patience=lr_patience,
    lr_gamma=lr_gamma,
    print_interval=print_interval,
)

# save model
pickle.dump(m, open(os.path.join(os.path.abspath(os.path.join(__file__, "..", "..")), "model", "model.pkl"), "wb"))

# print success message
print("Model Trained And Saved Successfully!")
input("Press Enter To Close The Program")
