from sklearn.model_selection import train_test_split
import os, itertools, json
from mixedmodel import MixedModel as mm
from utils import load_data, prepare_data, load_config

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
            print(f"Trial: {r}, Neurons: {c}, Accuracy: {results['accuracy']}")
        
        output["neurons"].append(c)
        output["results"].append(sum(accs)/len(accs))
    
        # Output results to file
        if save_output:
            with open(search_output_file, 'w') as s:
                json.dump(output, s)

def cnn_search(x, y, filters, conv_size, pooling_size, num_layers, model_params, full_rewrite=False, save_output=False):
    output_folder = os.path.join(os.getcwd(), "output")
    search_output_file = os.path.join(output_folder, "cnn_search.json")
    x = x.reshape(len(x), x.shape[1], x.shape[2], 1)
    
    # Prepare data
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    input_size, output_size = [train_x.shape[1:]], [train_y.shape[1:]]

    # Generate combinations of convolution operations
    conv_layer_opts = {"conv": conv_size, "filters": filters}
    conv_keys = conv_layer_opts.keys()
    conv_values = (conv_layer_opts[key] for key in conv_keys)
    conv_combinations = [dict(zip(conv_keys, combination)) for combination in itertools.product(*conv_values)]
    conv_layer_combos = [c for c in itertools.combinations(conv_combinations, 2)]

    # Generate combinations of pooling operations
    pooling_opts = {"pooling": pooling_size}
    pooling_keys = pooling_opts.keys()
    pooling_values = (pooling_opts[key] for key in pooling_keys)
    pooling_combinations = [dict(zip(pooling_keys, combination)) for combination in itertools.product(*pooling_values)]

    output = {"layers": list(), "results": list()}

    for c in conv_layer_combos:
        for p in pooling_combinations:
            accs = list()
            for r in range(5): # Take average over 5 runs
                ops = [d for d in c]
                ops.append(p)

                cnn_model = mm(mode="cnn-custom", 
                    model_params=model_params, 
                    input_shapes=input_size, 
                    output_shape=output_size, 
                    output_folder=output_folder,
                    auto_build=False)

                cnn_model.build_model(cnn_layers=ops)
                history = cnn_model.train(train_x, train_y, None)
                results = cnn_model.test(test_x, test_y)
                accs.append(results['accuracy'])
                print(f"Trial: {r}, Layers: {ops}, Accuracy: {results['accuracy']}")
        
            output["layers"].append(ops)
            output["results"].append(sum(accs)/len(accs))

            print(output)
            print(f"overall average results: {sum(accs)/len(accs)}")

            # Output results to file
            #if save_output:
            with open(search_output_file, 'w') as s:
                print(f"writing to file...")
                json.dump(output, s)

def run(mode="mlp"):

    # Load data, config and set up output directory
    output_folder = os.path.join(os.getcwd(), "output")
    img1, img2, img3, img4, metadata, labels = load_data()
    config = load_config()
    
    if mode == "mlp":
        mlp_params = config['MLP_SEARCH']
        neurons, params = mlp_params["NEURONS"], mlp_params["PARAMS"]

        # Perform search and generate output
        mlp_search(neurons=neurons, x=metadata, y=labels, model_params=params)

    if mode == "cnn":
        cnn_params = config['CNN_SEARCH']
        filters, conv_size, pooling, layers = cnn_params["FILTERS"], cnn_params["CONV2D_SIZE"], cnn_params["MAX_POOL"], cnn_params["LAYERS"]
        params = cnn_params["PARAMS"]

        # Perform search and generate output
        cnn_search(x=img1, y=labels, filters=filters, conv_size=conv_size, pooling_size=pooling,num_layers=layers,model_params=params)

if __name__ == "__main__":
    run(mode="cnn")