from sklearn.model_selection import train_test_split
import os

from mixedmodel import MixedModel as mm
from utils import load_data, prepare_data, load_config

def run(mode="mlp"):
    output_folder = os.path.join(os.getcwd(), "output")

    img1, img2, img3, img4, metadata, labels = load_data()
    config = load_config()
    model_params = config['MODEL']

    if mode == "mlp": # train a default multi-layer perceptron using only the patient metadata
        metadata_train_x, metadata_test_x, train_labels, test_labels = train_test_split(metadata, labels, test_size=0.2)
        input_size, output_size = [metadata_train_x.shape[1:]], [train_labels.shape[1:]]
        mlp_model = mm(mode="mlp", 
                        model_params=model_params, 
                        input_shapes=input_size, 
                        output_shape=output_size, 
                        output_folder=output_folder,
                        auto_build=True)
        mlp_model.train(metadata_train_x, train_labels, None)
        res = mlp_model.test(metadata_test_x, test_labels)
        print(res)

    elif mode == "cnn": # train a default convolutional neural network using one set of images
        img1_train_x, img1_test_x, train_labels, test_labels = train_test_split(img1, labels, test_size=0.2)
        input_size, output_size = [img1_train_x.shape[1:]], [train_labels.shape[1:]]
        cnn_model = mm(mode="cnn", 
                        model_params=model_params, 
                        input_shapes=input_size, 
                        output_shape=output_size, 
                        output_folder=output_folder,
                        auto_build=True)
        cnn_model.train(img1_train_x, train_labels, None)
        res = cnn_model.test(img1_test_x, test_labels)
        print(res)

    else: # train a default mixed-input model using patient metadata and one set of images together
        metadata_train_x, metadata_test_x, img1_train_x, img1_test_x, train_labels, test_labels = train_test_split(metadata, img1, labels, test_size=0.2)
        input_size, output_size = [img1_train_x.shape[1:], metadata_train_x.shape[1:]], [train_labels.shape[1:]]
        mixed_model = mm(mode="mixed", 
                        model_params=model_params, 
                        input_shapes=input_size, 
                        output_shape=output_size,
                        output_folder=output_folder,
                        auto_build=True)
        mixed_model.train([img1_train_x, metadata_train_x], train_labels, None)
        res = mixed_model.test([img1_test_x, metadata_test_x], test_labels)
        print(res)

if __name__ == "__main__":
    mode = "cnn" # Options are "mlp", "cnn", or "mixed"
    run(mode)