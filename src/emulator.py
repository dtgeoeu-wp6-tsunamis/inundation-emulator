from keras import layers, models, Input, regularizers, optimizers
from src.data_reader import DataReader
from datetime import datetime
import tensorflow as tf
import os
import csv
import numpy as np
import xarray as xr
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
        #reg = 1e-5 # Parameter penalization factor.
        alpha = 0.01
        
        encode = models.Sequential([
            Input(shape=(self.n_pois, 481)),
            layers.Reshape((15, 481, 1)),  # Optionally add channels later
            layers.Conv2D(8, (3, 3), strides=(1, 1), use_bias=False),
            layers.LeakyReLU(alpha=alpha),
            layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 2)),
            layers.Conv2D(16, (3, 5), strides=(1, 1), use_bias=False),
            layers.LeakyReLU(alpha=alpha),
            layers.MaxPooling2D(pool_size=(3, 5), strides=(2, 3)),
            layers.Conv2D(32, (3, 5), strides=(1, 1), padding='same', use_bias=False),
            layers.LeakyReLU(alpha=alpha),
            layers.MaxPooling2D(pool_size=(3, 5), strides=(2, 3)),
            layers.Conv2D(32, (1, 1), strides=(1, 1), use_bias=True),
            layers.LeakyReLU(alpha=alpha),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(8, use_bias=True),
            layers.LeakyReLU(alpha=alpha),
        ])
        
        # Decoder
        decode = models.Sequential([
            layers.Dense(8, use_bias=True),
            layers.LeakyReLU(alpha=alpha),
            layers.Dense(64, use_bias=True),
            layers.LeakyReLU(alpha=alpha),
            layers.Dense(418908, use_bias=True),
            layers.LeakyReLU(alpha=alpha),
        ])
        
        # Complete Model
        model = models.Sequential([encode, decode])
        return model

    def train_model(self, train_dir, train_scenarios, validation_dir, validation_scenarios, 
                    batch_size=32, epochs=30, l2_callback_frequency=30, save_model_frequency=50):

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
            tf.TensorSpec(shape=(self.n_pois, 481), dtype=tf.int32),                # eta
            tf.TensorSpec(shape=(readers["train"].topo_mask.sum()), dtype=tf.int32),# flow_depth
            tf.TensorSpec(shape=(), dtype=tf.string)                                # scenario
        )
        
        datasets["train"] = tf.data.Dataset.from_generator(
                generator=readers["train"].generator,
                output_signature=output_signature
        ).cache()
        
        
        # You may not want to cache both datasets..
        datasets["val"] = tf.data.Dataset.from_generator(
                generator=readers["val"].generator,
                output_signature=output_signature
        ).cache()

        # Compile the model with a loss function and optimizer
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), 
            loss="mse",
            metrics=['mse']
        )

        # Callbacks
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir, histogram_freq=1)
        l2_callback = L2MetricsCallback(
            train_dataset=datasets["train"],
            validation_dataset=datasets["val"],
            callback_frequency=l2_callback_frequency,
            output_csv=os.path.join(self.rundir, 'l2_metrics.csv')
        )
        save_model_callback = SaveModelEveryNEpochs(self.rundir, n_epochs=save_model_frequency)

        
        # Fit the model
        self.history = self.model.fit(datasets["train"].map(lambda x, y, _: (x, y)).shuffle(buffer_size=nr_of_scenarios).batch(batch_size), 
                                      epochs=epochs, 
                                      callbacks=[tensorboard_callback, l2_callback, save_model_callback],
                                      validation_data=datasets["val"].map(lambda x, y, _: (x, y)).batch(batch_size))
        
        hist_df = DataFrame(self.history.history) 
        with open(os.path.join(self.rundir, "train_summary.csv"), "a") as outfile:
            hist_df.to_csv(outfile)

    def predict(self, scenarios_file, input_dir, output_dir):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        with open(scenarios_file, 'r') as file:
            nr_of_scenarios = sum(1 for line in file if line.strip())
            print(f"Number of scenarios for prediction: {nr_of_scenarios}")
        
        reader = DataReader( 
                scenarios_file=scenarios_file,
                pois=self.pois,
                datadir=input_dir,
                topofile=self.topofile,
                topo_mask_file=self.topomask,
                shuffle_on_load=False,
                target=False,
                reload=False
        )
        
        output_signature = (
            tf.TensorSpec(shape=(self.n_pois, 481), dtype=tf.int32),                # eta
            tf.TensorSpec(shape=(), dtype=tf.int32),                                # No flow_depth
            tf.TensorSpec(shape=(), dtype=tf.string)                                # scenario
        )
        
        dataset = tf.data.Dataset.from_generator(
                generator=reader.generator,
                output_signature=output_signature
        )

        for x, scenario_id in dataset:
            # Make predictions
            #TODO: Use CT_mask..
            y_pred = self.model.predict(x, verbose=0)

            # Convert to NumPy for easier manipulation
            y_pred_np = y_pred.numpy() if hasattr(y_pred, 'numpy') else y_pred

            # Create a NetCDF file for each scenario
            scenario_id_str = scenario_id.numpy().decode('utf-8')  # Decode scenario_id to string
            file_path = os.path.join(output_dir, f"{scenario_id_str}.nc")

            # Create an xarray DataArray for the prediction
            data = xr.DataArray(
                y_pred_np,
                dims=["dim_0", "dim_1"],  # Update these based on the shape of your predictions
                name="prediction",
            )

            # Add metadata or attributes if necessary
            data.attrs["scenario"] = scenario_id_str
            data.attrs["description"] = "Model predictions for scenario"

            # Save to NetCDF
            data.to_netcdf(file_path)
            print(f"Saved predictions for scenario '{scenario_id_str}' to {file_path}")
            
            #pred_val = conv_model.predict(x=datasets.batch(40))
            #target_val = readers['validate'].get_targets()

    def load_model(self):
        return(models.load_model(self.model_file))

    def save_model(self):
        self.model.save(self.model_file)


class L2MetricsCallback(tf.keras.callbacks.Callback):
    """
    Computes L2 norm and L2 error for each scenario in the training and test sets
    at the end of each epoch, then writes results to a CSV.
    """
    def __init__(self, 
                 train_dataset,
                 validation_dataset,
                 callback_frequency,
                 output_csv='metrics.csv'):
        """
        Args:
            train_dataset (tf.data.Dataset): Dataset for training scenarios.
                                          Should yield (x, y_true, scenario) pairs.
            validation_dataset (tf.data.Dataset): Dataset for test/validation scenarios.
                                         Should yield (x, y_true, scenario) pairs.
            callback_frequency (int): Nr of epochs between each calculation.
            output_csv (str): Name/path of the CSV file where metrics are stored.
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.callback_frequency = callback_frequency
        self.output_csv = output_csv
        
        # Initialize CSV file with header if it doesn't exist or is empty
        if not os.path.exists(self.output_csv) or os.path.getsize(self.output_csv) == 0:
            with open(self.output_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'dataset', 'scenario', 'l2_norm', 'l2_error'])

    def on_epoch_end(self, epoch, logs=None):
        if (epoch +1) %  self.callback_frequency == 0:
            # Evaluate for all training scenarios
            
            self.compute_and_log_metrics(
                epoch=epoch, 
                dataset_name='train', 
                dataset=self.train_dataset, 
            )

            # Evaluate for all test scenarios
            self.compute_and_log_metrics(
                epoch=epoch, 
                dataset_name='validation', 
                dataset=self.validation_dataset, 
            )

    def compute_and_log_metrics(self, epoch, dataset_name, dataset):
        """
        Loop through each scenario in a dataset, compute the L2 norm & L2 error,
        and append results to the CSV file.
        """
        with open(self.output_csv, 'a', newline='') as f:
            writer = csv.writer(f)

            for x_batch, y_batch, scenario_id_batch in dataset.batch(30):
                y_pred_batch = self.model.predict(x_batch, verbose=0)
                l2_norm = np.sqrt(np.mean(y_batch**2, axis=1))                    # ||y_true||_2
                l2_error = np.sqrt(np.mean((y_pred_batch - y_batch)**2, axis=1))  # ||y_pred - y_true||_2
                
                # Process batch
                for i, scenario_id in enumerate(scenario_id_batch.numpy()):
                    # Write row: [epoch+1, dataset_name, scenario_id, l2_norm, l2_error]
                    writer.writerow([epoch + 1, dataset_name, scenario_id.decode('utf-8'), l2_norm[i], l2_error[i]])


class SaveModelEveryNEpochs(tf.keras.callbacks.Callback):
    def __init__(self, rundir, n_epochs=100):
        """
        Callback to save the model every `n_epochs`.

        Args:
            rundir (str): Rundirectory.
            n_epochs (int): Frequency of epochs to save the model.
        """
        super().__init__()
        self.save_path = os.path.join(rundir, "model_checkpoints")
        self.n_epochs = n_epochs

        # Ensure the save directory exists
        os.makedirs(self.save_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """
        Save the model every `n_epochs`.
        """
        if (epoch + 1) % self.n_epochs == 0:
            model_filename = os.path.join(self.save_path, f"model_epoch_{epoch + 1}.h5")
            self.model.save(model_filename)
            print(f"Model saved at: {model_filename}")
