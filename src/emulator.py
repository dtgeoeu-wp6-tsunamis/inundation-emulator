from keras import layers, models, Input, regularizers, optimizers
from src.data_reader import DataReader
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from pandas import DataFrame

class Emulator:
    def __init__(self, generated_dir, topofile, topomask, rundir=None):
        self.reg = 1e-15
        self.pois = range(30,45)
        self.n_pois = len(self.pois)
        self.generated_dir = generated_dir
        self.topofile = topofile
        self.topomask = topomask
    
        # Create rundir if does not exist. Else load model.
        if rundir:
            self.rundir=rundir
            self.model_file = os.path.join(rundir, "model.h5")
            self.model = self.load_model()
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.rundir = os.path.join(generated_dir, f"emulator_{timestamp}")
            os.makedirs(self.rundir, exist_ok=True)
            self.model = self.create_model()
            self.model_file = os.path.join(self.rundir, "model.h5")

        self.logdir = os.path.join(generated_dir, "logs", os.path.split(self.rundir)[-1])
    
    def create_model(self):
        reg = 1e-5 # Parameter penalization factor.
        
        encode = models.Sequential([
            Input(shape=(self.n_pois, 481, 1)),
            layers.Conv2D(32, (3, 3), activation='relu', strides=(1, 1), use_bias=False, kernel_regularizer=regularizers.l2(reg)),
            layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 2)),
            layers.Conv2D(64, (3, 5), activation='relu', strides=(1, 1), use_bias=False, kernel_regularizer=regularizers.l2(reg)),
            layers.MaxPooling2D(pool_size=(3, 5), strides=(2, 3)),
            layers.Conv2D(128, (3, 5), activation='relu', strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(reg)),
            layers.MaxPooling2D(pool_size=(3, 5), strides=(2, 3)),
            layers.Conv2D(32, (1, 1), activation='relu', strides=(1, 1), use_bias=True, kernel_regularizer=regularizers.l2(reg)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(16, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(reg))
        ])
        
        # Decoder
        decode = models.Sequential([
            layers.Dense(32, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(reg)),
            layers.Dense(418908, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(reg))
        ])
        
        # Complete Model
        model = models.Sequential([encode, decode])
        return model

    def train_model(self, train_dir, train_scenarios, validation_dir, validation_scenarios, batch_size=32, epochs=30):

        with open(train_scenarios, 'r') as file:
            nr_of_scenarios = sum(1 for line in file if line.strip())
            print(f"Number of training scenarios: {nr_of_scenarios}")

        data_config = {
            "train": {
                "scenarios_file": train_scenarios,
                "datadir": train_dir
            },
            "val": { 
                "scenarios_file": validation_scenarios,
                "datadir": validation_dir
            }
        }
        
        readers = {}
        for key, config in data_config.items():
            readers[key] = DataReader( 
                scenarios_file=config["scenarios_file"],
                pois=self.pois,
                datadir=config["datadir"],
                topofile=self.topofile,
                topo_mask_file=self.topomask,
                shuffle_on_load=False, 
                reload=False
            )
        

        datasets = {}
        # Create datasets from generators
        output_signature = (
            tf.TensorSpec(shape=(self.n_pois, 481), dtype=tf.int32),
            tf.TensorSpec(shape=(readers["train"].topo_mask.sum()), dtype=tf.int32)
        )
        datasets["train"] = tf.data.Dataset.from_generator(
                generator=readers["train"].generator,
                output_signature=output_signature
        ).cache().shuffle(buffer_size=nr_of_scenarios).batch(batch_size)
        
        datasets["val"] = tf.data.Dataset.from_generator(
                generator=readers["val"].generator,
                output_signature=output_signature
        ).cache().batch(batch_size)

        # Compile the model with a loss function and optimizer
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), 
            loss="mse",
            metrics=['mse']
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir, histogram_freq=1)
        
        # Fit the model
        self.history = self.model.fit(datasets["train"], 
                                      epochs=epochs, 
                                      callbacks=[tensorboard_callback],
                                      validation_data=datasets["val"])
        
        hist_df = DataFrame(self.history.history) 
        with open(os.path.join(self.rundir, "train_summary.csv"), "w") as outfile:
            hist_df.to_csv(outfile)

    def predict(self):
        pass
        #pred_val = conv_model.predict(x=datasets.batch(40))
        #target_val = readers['validate'].get_targets()

    def plot_train_summary(self):
        #Plot the training history
        plt.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig()

    def load_model(self):
        return(models.load_model(self.model_file))

    def save_model(self):
        self.model.save(self.model_file)
