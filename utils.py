from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, plot_model
import numpy as np
import csv, os, json

def load_config():
    file = os.path.join(os.getcwd(), "config.json")
    try:
        with open(file) as config_file:
            return json.loads(config_file.read())
    except:
        raise ValueError(f"Could not load config file: {file}")

def prepare_data(input, targets):
    x_train, x_test, y_train, y_test = train_test_split(input, targets, test_size=0.2)
    return x_train, y_train, x_test, y_test

def load_data():

    # Generate paths
    DATA_DIR = os.path.join(os.getcwd(), "data")

    img_file_1 = os.path.join(DATA_DIR, "SUBJ_x.npy") # averaged side scan
    img_file_2 = os.path.join(DATA_DIR, "T88_90_x.npy") # top
    img_file_3 = os.path.join(DATA_DIR, "T88_95_x.npy") # zoomed side scan
    img_file_4 = os.path.join(DATA_DIR, "T88_110_x.npy") # frontal scan

    metadata_file = os.path.join(DATA_DIR, "Oasis-metadata_final.csv")

    labels_file = os.path.join(DATA_DIR, "y-2.npy")

    # Load metadata file
    csv_reader = csv.reader(open(metadata_file, "r"), delimiter=",")
    x = list(csv_reader)
    x.pop(0)
    metadata = np.array(x)
    metadata = np.delete(metadata, 0, 1)
    metadata = np.delete(metadata, -1, 1)
    metadata = metadata.astype(np.float)

    # Load targets file
    targets = np.load(labels_file)
    targets[targets == 2] = 3
    targets[targets == 1] = 2
    targets[targets == 0.5] = 1
    targets = to_categorical(targets, num_classes=4)
    

    # Load image files
    img_arr_1 = np.load(img_file_1)
    img_arr_2 = np.load(img_file_2)
    img_arr_3 = np.load(img_file_3)
    img_arr_4 = np.load(img_file_4)

    return img_arr_1, img_arr_2, img_arr_3, img_arr_4, metadata, targets