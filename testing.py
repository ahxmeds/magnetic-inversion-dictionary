import os

from ssmaginv.dataset.data_dict import load_npz_true_data
import ssmaginv.config as config
import ssmaginv.plot.plot as plot

train_dir = config.TRAIN_DIR

# Load a sample from dir
sample_path = os.path.join(train_dir, "0001.npz")
x_true, data_true, data_true_noise = load_npz_true_data(sample_path)

p = plot.plot_triplet
