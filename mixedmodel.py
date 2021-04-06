from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Dropout, concatenate
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class MixedModel:
    def __init__(self, mode, model_params, input_shapes, output_shape, output_folder, neurons=None, auto_build=False):
        self.model = None
        self.mode = mode
        self.is_trained = False
        self.output_folder = output_folder
        self.input_shapes = input_shapes
        self.output_shape = output_shape

        self.batch_size = model_params['MINIBATCH_SIZE']
        self.epochs = model_params['EPOCHS']
        self.loss_func = model_params['LOSS']
        self.optimizer = model_params['OPTIMIZER']
        self.patience = model_params['PATIENCE']
        self.verbose = model_params['VERBOSE_OUTPUT']
        self.validation_percentage = model_params['VALIDATION_PERCENTAGE']

        self.neurons=neurons

        if auto_build:
            self.build_model(self.mode)

    def build_model(self, neurons=None, cnn_layers=None, auto_compile=True):
        #if self.mode != "mixed" and len(self.input_shapes) > 1:
        #    raise ValueError("Cannot accept more than one input shape if mode is not mixed")
        
        if self.mode == "mlp":
            inputs, x = self.build_default_mlp(self.input_shapes[0])
            x = Dense(self.output_shape[0][0], activation="softmax")(x)
            self.model = Model(inputs, x)
        elif self.mode == "cnn":
            inputs, x = self.build_default_cnn(self.input_shapes[0])
            x = Dense(self.output_shape[0][0], activation="softmax")(x)
            self.model = Model(inputs, x)
        elif self.mode == "mixed":
            inputs1, x = self.build_default_cnn(self.input_shapes[0])
            inputs2, y = self.build_default_mlp(self.input_shapes[1])
            z = concatenate([x, y])
            z = Dense(self.output_shape[0][0], activation="softmax")(z)
            self.model = Model([inputs1, inputs2], z)
        elif self.mode == "mlp-custom":
            inputs, x = self.build_custom_mlp(self.input_shapes[0], neurons)
            x = Dense(self.output_shape[0][0], activation="softmax")(x)
            self.model = Model(inputs, x)
        elif self.mode == "cnn-custom":
            #input_size = self.input_shapes[0] + (1,)
            self.model = tf.keras.Sequential()
            self.model.add(Input(shape=self.input_shapes[0]))
            self.build_custom_cnn(cnn_layers)
            self.model.add(Flatten())
            self.model.add(Dense(6, activation="relu"))
            self.model.add(Dense(self.output_shape[0][0], activation="softmax"))
        elif self.mode =="mixed-searched":
            
            # build model for image
            inputs1 = Input(shape=self.input_shapes[0])
            a = Conv2D(filters=16, kernel_size=(3,3), padding="same", activation="relu")(inputs1)
            a = BatchNormalization(axis=-1)(a)
            a = Dropout(0.25)(a)
            a = Conv2D(filters=16, kernel_size=(5,5), padding="same", activation="relu")(a)
            a = BatchNormalization(axis=-1)(a)
            a = MaxPooling2D(pool_size=(2,2))(a)
            a = Flatten()(a)
            a = Dense(6, activation="relu")(a)

            # build model for mlp
            inputs2 = Input(shape=self.input_shapes[1])
            b = Dense(10, activation="relu")(inputs2)
            b = Dropout(0.25)(b) # removing this yieled about 58% accuracy
            b = Dense(4, activation="relu")(b)
            b = Dense(10, activation="relu")(b)

            # concatenate
            z = concatenate([a, b])

            # softmax activation to find output
            z = Dense(self.output_shape[0][0], activation="softmax")(z)

            # create model
            self.model = Model([inputs1, inputs2], z)

        elif self.mode == "full-mixed":
            
            # build model for img1
            inputs1 = Input(shape=self.input_shapes[0])
            a = Conv2D(filters=16, kernel_size=(3,3), padding="same", activation="relu")(inputs1)
            a = BatchNormalization(axis=-1)(a)
            a = Dropout(0.25)(a)
            a = Conv2D(filters=16, kernel_size=(5,5), padding="same", activation="relu")(a)
            a = BatchNormalization(axis=-1)(a)
            a = MaxPooling2D(pool_size=(2,2))(a)
            a = Flatten()(a)
            a = Dense(6, activation="relu")(a)

            # build model for img2
            inputs2 = Input(shape=self.input_shapes[1])
            b = Conv2D(filters=16, kernel_size=(3,3), padding="same", activation="relu")(inputs2)
            b = BatchNormalization(axis=-1)(b)
            b = Dropout(0.25)(b)
            b = Conv2D(filters=16, kernel_size=(5,5), padding="same", activation="relu")(b)
            b = BatchNormalization(axis=-1)(b)
            b = MaxPooling2D(pool_size=(2,2))(b)
            b = Flatten()(b)
            b = Dense(6, activation="relu")(b)

            # build model for img3
            inputs3 = Input(shape=self.input_shapes[2])
            c = Conv2D(filters=16, kernel_size=(3,3), padding="same", activation="relu")(inputs3)
            c = BatchNormalization(axis=-1)(c)
            c = Dropout(0.25)(c)
            c = Conv2D(filters=16, kernel_size=(5,5), padding="same", activation="relu")(c)
            c = BatchNormalization(axis=-1)(c)
            c = MaxPooling2D(pool_size=(2,2))(c)
            c = Flatten()(c)
            c = Dense(6, activation="relu")(c)

            # build model for img4
            inputs4 = Input(shape=self.input_shapes[3])
            d = Conv2D(filters=16, kernel_size=(3,3), padding="same", activation="relu")(inputs4)
            d = BatchNormalization(axis=-1)(d)
            d = Dropout(0.25)(d)
            d = Conv2D(filters=16, kernel_size=(5,5), padding="same", activation="relu")(d)
            d = BatchNormalization(axis=-1)(d)
            d = MaxPooling2D(pool_size=(2,2))(d)
            d = Flatten()(d)
            d = Dense(6, activation="relu")(d)

            # build model for mlp
            inputs5 = Input(shape=self.input_shapes[4])
            e = Dense(25, activation="relu")(inputs5)
            #e = Dropout(0.25)(e) # removing this yieled about 58% accuracy
            #e = Dense(4, activation="relu")(e)
            #e = Dense(10, activation="relu")(e)

            # concatenate
            z = concatenate([a, b, c, d, e])

            # softmax activation to find output
            z = Dense(self.output_shape[0][0], activation="softmax")(z)

            # create model
            self.model = Model([inputs1, inputs2, inputs3, inputs4, inputs5], z)

        else:
            raise ValueError("Unrecognized mode for model generation")



        if auto_compile:
            self.compile_model()

    def build_default_mlp(self, input_size):
        inputs = Input(shape=input_size[0])
        x = Dense(8, activation="relu")(inputs)
        x = Dense(4, activation="relu")(x)
        x = Dense(10, activation="relu")(x)
        #x = Dense(self.output_shape[0][0], activation="softmax")(x)
        return inputs, x

    def build_default_cnn(self, input_size):
        input_size = input_size + (1,) # our images are all b&w so we need to attach the single layer here
        inputs = Input(shape=input_size)
        x = Conv2D(16, (3,3), padding="same", activation="relu")(inputs)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Dropout(0.5)(x)

        x = Conv2D(32, (3,3), padding="same", activation="relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(6, activation="relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dropout(0.25)(x)

        return inputs, x

    def build_custom_mlp(self, input_size, neurons):
        inputs = Input(shape=input_size[0])
        x = Dense(8, activation="relu")(inputs)
        for n in neurons:
            x = Dense(n, activation="relu")(x)
        #x = Dense(self.output_shape[0][0], activation="softmax")(x)
        
        return inputs, x

    def build_custom_cnn(self, layer_defs): # This will only add layers, input/output should be defined in the build_model function
        for l in layer_defs:
            self.model.add(self.decode_cnn_layer(l))
            self.model.add(BatchNormalization(axis=-1))

    def decode_cnn_layer(self, op):
        print(op)
        k = op.keys()
        if "conv" in k:
            # Generate convolutional layer
            f = op['filters']
            s = op['conv']
            return Conv2D(f, (s,s), padding='same', activation="relu")

        if "pooling" in k:
            # Generate pooling layer
            s = op['pooling']
            return MaxPooling2D((s,s))

    def compile_model(self):
        if self.model is not None:
            self.model.compile(loss=self.loss_func, optimizer=self.optimizer, metrics='accuracy') 
            return True
        return False

    def train(self, x, y, output_model_file, save_at_checkpoints=True, early_stopping=True, use_existing=True):
        if self.model is not None:
            
            # TODO: Implement loading from existing model

            callbacks = []
            if early_stopping:
                callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, verbose=self.verbose))

            # TODO: Implement saving at checkpoints

            print(f"Training {self.mode} model...")
            history = self.model.fit(x, y, 
                validation_split=self.validation_percentage, 
                epochs=self.epochs,
                shuffle=True, 
                verbose=self.verbose,
                callbacks=callbacks)
            self.is_trained = True

            return history

    def print_model(self):
        if self.model is not None:
            print(self.model.summary())
        else:
            raise ValueError("Model is undefined, cannot print")

    def save_model(self):
        pass

    def load_model(self):
        pass

    def test(self, x, y):
        if not self.is_trained:
            raise ValueError("Model is not yet trained")

        results = self.model.evaluate(x, y, return_dict=True)
        return results