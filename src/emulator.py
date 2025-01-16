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

    def train_model(self, datadir, scenarios_file, batch_size, epochs):
        
        with open(scenarios_file, 'r') as file:
            nr_of_scenarios = sum(1 for line in file if line.strip())
        print(f"Number of scenarios: {nr_of_scenarios}")
        
        reader = DataReader(
            scenarios_file=scenarios_file,
            pois=self.pois,
            datadir=datadir,
            topofile=self.topofile,
            topo_mask_file=self.topomask,
            shuffle_on_load=False, 
            reload=False
        )
        
        log_dir = "logs/fit/" + tf.timestamp().numpy().astype(str)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
                generator=reader.generator,
                output_signature=(
                        tf.TensorSpec(shape=(self.n_pois, 481), dtype=tf.int32),
                        tf.TensorSpec(shape=(reader.topo_mask.sum()), dtype=tf.int32)
                )
        ).cache().shuffle(buffer_size=nr_of_scenarios)
        batched_dataset = dataset.batch(batch_size)#.prefetch(tf.data.AUTOTUNE)


        # Compile the model with a loss function and optimizer
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001, 
                                                     beta_1=0.9, 
                                                     beta_2=0.999), 
            loss="mse",
            metrics=['mse']
        )

        # Fit the model
        self.history = self.model.fit(batched_dataset, epochs=epochs, callbacks=[tensorboard_callback])
        
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