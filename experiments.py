from sklearn.model_selection import train_test_split
import os, itertools, json
from mixedmodel import MixedModel as mm
from utils import *

def mlp_search(neurons, x, y, model_params, full_rewrite=False, save_output=True):
    output_folder = os.path.join(os.getcwd(), "output")
    search_output_file = os.path.join(output_folder, "mlp_search_neurons.json")
    
    # Prepare data
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    input_size, output_size = [train_x.shape[1:]], [train_y.shape[1:]]

    # Generate all combinations of layers and neurons
    expanded_neurons = list()
    for l in neurons:
        thislayer = list()
        for i in range(1, l + 1):
            thislayer.append(i)
        expanded_neurons.append(thislayer)

    # Generate all possible combinations
    combinations = [p for p in itertools.product(*expanded_neurons)]

    # Load previously saved results
    output = {"neurons": list(), "results": list()}
    if not full_rewrite:
        # Open existing results and don't repeat these
        with open(search_output_file, 'r') as s:
            output = json.loads(s.read())

    # Remove those configurations that we have already evaluated
    for n in output["neurons"]:
        if n in combinations:
            combinations.remove(n)

    # Evaluate all possible combinations of neuron arrangement
    for c in combinations:
        accs = list()
        for r in range(5): # Take average over 5 runs
            mlp_model = mm(mode="mlp-custom", 
                model_params=model_params, 
                input_shapes=input_size, 
                output_shape=output_size, 
                output_folder=output_folder,
                auto_build=False)
            mlp_model.build_model(neurons=c, auto_compile=True)
            history = mlp_model.train(train_x, train_y, None)
            results = mlp_model.test(test_x, test_y)
            accs.append(results['accuracy'])
        
        output["neurons"].append(c)
        output["results"].append(sum(accs)/len(accs))
    
        # Output results to file
        with open(search_output_file, 'w') as s:
            json.dump(output, s)

def run():

    output_folder = os.path.join(os.getcwd(), "output")

    img1, img2, img3, img4, metadata, labels = load_data()

    config = load_config()
    mlp_params = config['MLP_SEARCH']
    neurons = mlp_params["NEURONS"]
    params = mlp_params["PARAMS"]

    mlp_search(neurons=neurons, x=metadata, y=labels, model_params=params)

if __name__ == "__main__":
    run()