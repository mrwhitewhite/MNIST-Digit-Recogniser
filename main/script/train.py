# import all the necessary libraries and modules
import pickle
from object import model as model
from object.Data import Data as Data
from lib.show import show as show

# set seed for reproducibility
seed = 123

# set Hyperparameters
_cf = 8

# load data
train_path = "/data/train.csv"
test_path = "/data/test.csv"
submission_path = "/data/sample_submission.csv"

d = Data(
    train_path,
    test_path,
    submission_path,
    seed=seed,
    cv1_size=0.1,
    cv2_size=0.1,
)

# train model
m = model(d, _cf, seed)
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
