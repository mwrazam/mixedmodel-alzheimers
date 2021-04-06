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

def cnn_search(x, y, filters, conv_size, pooling_size, num_layers, model_params, full_rewrite=False, save_output=True):
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

    # Generate combinations list
    combos = list()
    for c in conv_layer_combos:
        iteration = list()
        for p in pooling_combinations:
            cs = [d for d in c]
            cs.append(p)
        combos.append(cs)

    # Load previously saved results
    if not full_rewrite:
        if not full_rewrite:
            with open(search_output_file, 'r') as s:
                output = json.loads(s.read())

        # Remove ones we have already computed from our combination list
        for c in combos:
            if c in output['layers']:
                combos.remove(c)

    # Run model search
    for idx, c in enumerate(combos):
        accs = list()
        for r in range(5):
            cnn_model = mm(mode="cnn-custom", 
                model_params=model_params, 
                input_shapes=input_size, 
                output_shape=output_size, 
                output_folder=output_folder,
                auto_build=False)

            cnn_model.build_model(cnn_layers=c)
            history = cnn_model.train(train_x, train_y, None)
            results = cnn_model.test(test_x, test_y)
            accs.append(results['accuracy'])
            print(f"Model #{idx}/{len(combos)}, Trial: {r}, Layers: {c}, Accuracy: {results['accuracy']}")
        
        output["layers"].append(c)
        output["results"].append(sum(accs)/len(accs))

        print(output)
        print(f"overall average results: {sum(accs)/len(accs)}")

        if save_output:
            with open(search_output_file, 'w') as s:
                print(f"writing to file...")
                json.dump(output, s)

def cnn_block_search(x, y, layer_defs, num_blocks, model_params, save_output=True):

    output_folder = os.path.join(os.getcwd(), "output")
    block_search_output_file = os.path.join(output_folder, "block_search.json")
    x = x.reshape(len(x), x.shape[1], x.shape[2], 1)
    
    # Prepare data
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    input_size, output_size = [train_x.shape[1:]], [train_y.shape[1:]]

    output = list()
    for l in layer_defs:
        layers = list()
        for b in range(num_blocks):
            layers = layers + l
            accs = list()
            for e in range(5):
                cnn_model = mm(mode="cnn-custom", 
                    model_params=model_params, 
                    input_shapes=input_size, 
                    output_shape=output_size, 
                    output_folder=output_folder,
                    auto_build=False)

                cnn_model.build_model(cnn_layers=layers)
                history = cnn_model.train(train_x, train_y, None)
                results = cnn_model.test(test_x, test_y)
                accs.append(results['accuracy'])
                print(f"Trial: {e}, layers: {layers}")

            o = {'layers': layers, 'acc': sum(accs)/len(accs)}
            print(o)
            output.append(o)

            if save_output:
                with open(block_search_output_file, 'w') as s:
                    print(f"writing to file...")
                    json.dump(output, s)

def mixed_searched(img, metadata, y):

    accs = list()
    for n in range(5):
        model_params = {
            "MINIBATCH_SIZE": 64,
            "EPOCHS": 100,
            "LOSS": "categorical_crossentropy",
            "OPTIMIZER": "rmsprop",
            "PATIENCE": 20,
            "VERBOSE_OUTPUT": 1,
            "VALIDATION_PERCENTAGE": 0.2
        }
        
        metadata_train_x, metadata_test_x, img_train_x, img_test_x, train_labels, test_labels = train_test_split(metadata, img, y, test_size=0.2)
        img_train_x = img_train_x.reshape(len(img_train_x), img_train_x.shape[1], img_train_x.shape[2], 1)
        input_size, output_size = [img_train_x.shape[1:], metadata_train_x.shape[1:]], [train_labels.shape[1:]]
        mixed_model = mm(mode="mixed-searched", 
                            model_params=model_params, 
                            input_shapes=input_size, 
                            output_shape=output_size,
                            output_folder=None,
                            auto_build=False)
        mixed_model.build_model()
        mixed_model.train([img_train_x, metadata_train_x], train_labels, None)
        res = mixed_model.test([img_test_x, metadata_test_x], test_labels)
        accs.append(res['accuracy'])
    print(accs)
    print(sum(accs)/len(accs))


def full_mixed_model(i1, i2, i3, i4, m, y): # put hardcoded definitions into mixedmodel class
    
    accs = list()
    for n in range(10):
        model_params = {
            "MINIBATCH_SIZE": 10,
            "EPOCHS": 150,
            "LOSS": "categorical_crossentropy",
            "OPTIMIZER": "adam",
            "PATIENCE": 20,
            "VERBOSE_OUTPUT": 1,
            "VALIDATION_PERCENTAGE": 0.2
        }
        i1_train_x, i1_test_x, i2_train_x, i2_test_x, i3_train_x, i3_test_x, i4_train_x, i4_test_x, m_train_x, m_test_x, train_y, test_y = train_test_split(i1, i2, i3, i4, m, y, test_size=0.2)
        i1_train_x = i1_train_x.reshape(len(i1_train_x), i1_train_x.shape[1], i1_train_x.shape[2], 1)
        i2_train_x = i2_train_x.reshape(len(i2_train_x), i2_train_x.shape[1], i2_train_x.shape[2], 1)
        i3_train_x = i3_train_x.reshape(len(i3_train_x), i3_train_x.shape[1], i3_train_x.shape[2], 1)
        i4_train_x = i4_train_x.reshape(len(i4_train_x), i4_train_x.shape[1], i4_train_x.shape[2], 1)
        input_size, output_size = [i1_train_x.shape[1:], i2_train_x.shape[1:], i3_train_x.shape[1:], i4_train_x.shape[1:], m_train_x.shape[1:]], [train_y.shape[1:]]

        mixed_model = mm(mode="full-mixed", 
                            model_params=model_params, 
                            input_shapes=input_size, 
                            output_shape=output_size,
                            output_folder=None,
                            auto_build=False)
        mixed_model.build_model()
        mixed_model.train([i1_train_x, i2_train_x, i3_train_x, i4_train_x, m_train_x], train_y, None)
        res = mixed_model.test([i1_test_x, i2_test_x, i3_test_x, i4_test_x, m_test_x], test_y)
        print(res)
        accs.append(res['accuracy'])
    print(accs)
    print(sum(accs)/len(accs))

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

    if mode == "cnn-block":
        cnn_params = config['CNN_SEARCH']
        params = cnn_params["PARAMS"]
        num_blocks = cnn_params['LAYERS']

        # Find best models to search for
        with open((os.path.join(output_folder, "cnn_search.json"))) as f:
            res = json.loads(f.read())
            maxes = list()
            for n in range(6):
                max = {'block': None, 'acc': 0}
                removeIdx = None
                for i, r in enumerate(res['results']):
                    if r > max['acc']:
                        max['block'] = res['layers'][i]
                        max['acc'] = r
                        removeIdx = i
                res['results'].pop(removeIdx)
                res['layers'].pop(removeIdx)
                maxes.append(max)

            max_block_defs = [b['block'] for b in maxes]
            
            cnn_block_search(img1, labels, max_block_defs, num_blocks, params)

    if mode == "mixed-searched":
        mixed_searched(img1, metadata, labels)

    if mode == "full-mixed":
        # best found model #1 had config: acc = 0.642553174495697, layers ={'conv': 3, 'filters': 16}, {'conv': 5, 'filters': 16}, {'pooling': 2}
        # best found model #2 had config: acc = 0.642553174495697, layers ={'conv': 3, 'filters': 16}, {'conv': 5, 'filters': 32}, {'pooling': 2}
        full_mixed_model(img1, img2, img3, img4, metadata, labels)

if __name__ == "__main__":
    run(mode="full-mixed")