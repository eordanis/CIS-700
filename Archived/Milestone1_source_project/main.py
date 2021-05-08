from trainer import Trainer
from gan import GAN
from DataLoader import DataLoader
from DataHandler import DataHandler
from utils import create_directories
#import sys
#sys.path.append("\MotionSense")

features = ["userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]
act_labels = ["jog"]#,"ups","wlk", "jog", "sit", "std"]


train_loader = DataLoader()
train_ts, test_ts,num_features, num_act_labels = train_loader.extract_from_csv(features, act_labels, verbose=True)


train_data, act_train_labels = train_loader.time_series_to_section(train_ts.copy(),
                                                                   num_act_labels,
                                                                   sliding_window_size=200,
                                                                   step_size_of_sliding_window=10)

test_data, act_test_labels = train_loader.time_series_to_section(test_ts.copy(),
                                                                 num_act_labels,
                                                                 sliding_window_size=200,
                                                                 step_size_of_sliding_window=10)

print("---Data is successfully loaded")
handler = DataHandler(train_data, test_data)
norm_train = handler.normalise("train")
norm_test = handler.normalise("test")

print("--- Shape of Training Data:", train_data.shape)
print("--- Shape of Test Data:", test_data.shape)

expt_name = "thurs_Script_jog2"

create_directories(expt_name)
gan_ = GAN(norm_train.shape)
trainer_ = Trainer(gan_, expt_name)
trainer_.train_gan(epochs=200, batch_size=128, sample_interval=10, train_data=norm_train)