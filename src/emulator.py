from keras import layers, models, Input, optimizers
import json
from src.data_reader import DataReader
import tensorflow as tf
import os
import csv
import numpy as np
from pandas import DataFrame
from src.logger_setup import get_logger
from netCDF4 import Dataset
import time

@tf.keras.utils.register_keras_serializable()
def asym_loss(y_true, y_pred): #added to implement asym loss function for underprediction issues
    error = y_pred - y_true
    squared_error = tf.square(error)

    # If underprediction (error < 0), scale by y_true; else, use scale = 1.0
    scale = tf.where(error < 0, tf.abs(y_true) + 1, tf.ones_like(y_true))

    return tf.reduce_mean(squared_error * scale)

class Emulator:
    def __init__(self, generated_dir, rundir, epoch_checkpoint=None):
        self.rundir=rundir
        self.generated_dir = generated_dir
        self.topofile = os.path.join(self.rundir, "topography.grd") 
        self.topomask_file = os.path.join(rundir, "topomask.npy")
        self.grid_info_file = os.path.join(self.rundir, "grid_info.json")
        with open(self.grid_info_file) as f:
            self.grid_info = json.load(f)
        self.topomask = np.load(self.topomask_file)
        
        # Input-Output dimensions 
        # Should be stored in a config file created by datareader.
        self.pois = range(30,45)
        self.n_pois = len(self.pois)
        self.input_time_steps = 481
        self.grid_lat = self.grid_info['dimensions']['grid_lat']
        self.grid_lon = self.grid_info['dimensions']['grid_lon']
        self.nr_pixel_out = self.topomask.sum()

        self.id=os.path.split(rundir)[-1]
        if epoch_checkpoint:
            self.model_file = os.path.join(self.rundir, "model_checkpoints", f"model_epoch_{epoch_checkpoint}.keras")
        else:
            self.model_file = os.path.join(rundir, "model.keras")
        if os.path.isfile(self.model_file):
            self.model = self.load_model()
        else:
            self.model = self.create_model()
        
        self.logger = get_logger(__name__, self.rundir)
        
        # For tensorboard
        self.logdir = os.path.join(generated_dir, "logs", os.path.split(self.rundir)[-1])
        
        # Write model config to log.
        self.logger.info("Encoder:")
        self.model.layers[0].summary(print_fn=self.logger.info)
        self.logger.info("Decoder:")
        self.model.layers[1].summary(print_fn=self.logger.info)
    
    def create_model(self):
        #reg = 1e-5 # Parameter penalization factor.
        alpha = 0.01 # Leaky relu parameter ().
        
        encode = models.Sequential([
            Input(shape=(self.n_pois, self.input_time_steps)),
            layers.Reshape((15, self.input_time_steps, 1)),  # Optionally add channels later
            layers.Conv2D(8, (3, 3), strides=(1, 1), activation="linear", use_bias=False),
            layers.LeakyReLU(alpha=alpha),
            layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 2)),
            layers.Conv2D(16, (3, 5), strides=(1, 1), activation="linear", use_bias=False),
            layers.LeakyReLU(alpha=alpha),
            layers.MaxPooling2D(pool_size=(3, 5), strides=(2, 3)),
            layers.Conv2D(32, (3, 5), strides=(1, 1), activation="linear", padding='same', use_bias=False),
            layers.LeakyReLU(alpha=alpha),
            layers.MaxPooling2D(pool_size=(3, 5), strides=(2, 3)),
            layers.Conv2D(32, (1, 1), strides=(1, 1), activation="linear", use_bias=True),
            layers.LeakyReLU(alpha=alpha),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(16, activation="linear", use_bias=True),
            layers.LeakyReLU(alpha=alpha),
        ])
        
        # Decoder
        decode = models.Sequential([
            layers.Dense(16, activation="linear", use_bias=True),
            layers.LeakyReLU(alpha=alpha),
            layers.Dense(64, activation="linear", use_bias=True),
            layers.LeakyReLU(alpha=alpha),
            layers.Dense(self.nr_pixel_out, activation="linear", use_bias=True),
            layers.LeakyReLU(alpha=alpha),
        ])
        
        # Complete Model
        model = models.Sequential([encode, decode])
        return model

    def train_model(self, train_dir, train_scenarios, validation_dir, validation_scenarios, 
                    batch_size=20, epochs=600, l2_callback_frequency=100, save_model_frequency=200):

        with open(train_scenarios, 'r') as file:
            nr_of_scenarios = sum(1 for line in file if line.strip())
            self.logger.info(f"Number of training scenarios: {nr_of_scenarios}")

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
                rundir=self.rundir,
                scenarios_file=config["scenarios_file"],
                datadir=config["datadir"],
                pois=self.pois,
                shuffle_on_load=False, 
                reload=False
            )
        
        datasets = {}
        # Create datasets from generators
        output_signature = (
            tf.TensorSpec(shape=(self.n_pois, self.input_time_steps), dtype=tf.float32),        # eta
            tf.TensorSpec(shape=(self.nr_pixel_out), dtype=tf.float32),                         # flow_depth
            tf.TensorSpec(shape=(), dtype=tf.string)                                            # scenario
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
            loss=asym_loss, #was "mse",
            metrics=['mse'], #for reporting
        )

        # Callbacks
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir, histogram_freq=1)
        l2_callback = L2MetricsCallback(
            train_dataset=datasets["train"],
            validation_dataset=datasets["val"],
            callback_frequency=l2_callback_frequency,
            output_csv=os.path.join(self.rundir, 'l2_metrics.csv')
        )
        save_model_callback = SaveModelEveryNEpochs(self.rundir, logger=self.logger, n_epochs=save_model_frequency)
        logger_callback = LoggingCallback(self.logger)
        
        # Fit the model
        self.history = self.model.fit(datasets["train"].map(lambda x, y, _: (x, y)).shuffle(buffer_size=nr_of_scenarios).batch(batch_size), 
                                      epochs=epochs, 
                                      callbacks=[tensorboard_callback, 
                                                 l2_callback, 
                                                 save_model_callback, 
                                                 logger_callback],
                                      validation_data=datasets["val"].map(lambda x, y, _: (x, y)).batch(batch_size))
        
        hist_df = DataFrame(self.history.history) 
        with open(os.path.join(self.rundir, "train_summary.csv"), "a") as outfile:
            hist_df.to_csv(outfile)

    def predict(self, scenarios_file, input_dir, output_dir, batch_size=100):
        """Make predictions and write each to netcdf file.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # load the stored grid information
        with open(self.grid_info_file, "r") as f:
            grid_info = json.load(f)
        
        with open(scenarios_file, 'r') as file:
            nr_of_scenarios = sum(1 for line in file if line.strip())
            self.logger.info(f"Number of scenarios for prediction: {nr_of_scenarios}")
        
        reader = DataReader(
                rundir=self.rundir, 
                scenarios_file=scenarios_file,
                datadir=input_dir,
                pois=self.pois,
                shuffle_on_load=False,
                target=False,
                reload=False
        )
        
        output_signature = (
            tf.TensorSpec(shape=(self.n_pois, self.input_time_steps), dtype=tf.float32),    # eta
            tf.TensorSpec(shape=(0,), dtype=tf.float32),                                # No flow_depth
            tf.TensorSpec(shape=(), dtype=tf.string)                                # scenario
        )
        
        dataset = tf.data.Dataset.from_generator(
                generator=reader.generator,
                output_signature=output_signature
        ).batch(batch_size, drop_remainder=False)

        # Make predictions
        preds = np.zeros((batch_size, *self.topomask.shape))
        flow_depths = np.zeros((batch_size, *self.topomask.shape))

        for eta, flow_depth, scenario_id in dataset.take(-1):
            #add wait of 1 sec to avoid segmentation fault
            time.sleep(1)
            db = eta.shape[0] # Dynamic batch size
            preds[:db, self.topomask] = self.model(eta, training=False)
            #flow_depths[:db,self.topomask] = flow_depth
            self.write_predictions(preds[:db], scenario_id, grid_info, output_dir)
        
    def write_predictions(self, preds, scenario_id, grid_info, output_dir):
        """ Write predictions to file.
        """
        for scenario, pred in zip(scenario_id, preds):
            # Create a NetCDF file for each scenario
            scenario_id_str = scenario.numpy().decode('utf-8') 
            pred_file = os.path.join(output_dir, f"{scenario_id_str}_CT_10m_PR.nc")
            
            with Dataset(pred_file, mode='w', format="NETCDF4") as dst:
                dst.scenario = scenario_id_str
                dst.model_id = self.id
                dst.description = "Predicted maximal flow depth."
                for dim_name, dim_size in grid_info["dimensions"].items():
                    dst.createDimension(dim_name, dim_size)
                for var_name, var_info in grid_info["variables"].items():
                    dst_var = dst.createVariable(var_name, np.dtype(var_info["datatype"]), var_info["dimensions"])
                    for attr, value in var_info["attributes"].items():
                        dst_var.setncattr(attr, value)
                    dst_var[:] = np.array(var_info["data"])
                # Add predicted
                prediction = dst.createVariable("predicted", "f4", ("grid_lat", "grid_lon"))
                prediction.units = "meter"
                prediction.description = "Maximum flow depth."
                prediction[:,:] = pred
            self.logger.info(f"NetCDF file created: {pred_file}")
        
    # def load_model(self, model_file=None):
    #     if model_file:
    #         return(models.load_model(model_file))
    #     else:
    #         return(models.load_model(self.model_file))

    def load_model(self, model_file=None):
        if model_file:
            return models.load_model(model_file, custom_objects={"asym_loss": asym_loss})
        else:
            return models.load_model(self.model_file, custom_objects={"asym_loss": asym_loss})

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
    def __init__(self, rundir, logger, n_epochs=100):
        """
        Callback to save the model every `n_epochs`.

        Args:
            rundir (str): Rundirectory.
            n_epochs (int): Frequency of epochs to save the model.
        """
        super().__init__()
        self.save_path = os.path.join(rundir, "model_checkpoints")
        self.logger = logger
        self.n_epochs = n_epochs

        # Ensure the save directory exists
        os.makedirs(self.save_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """
        Save the model every `n_epochs`.
        """
        if (epoch + 1) % self.n_epochs == 0:
            model_filename = os.path.join(self.save_path, f"model_epoch_{epoch + 1}.keras")
            self.model.save(model_filename)
            self.logger.info(f"Model saved at: {model_filename}")


class LoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logger):
        """
        Custom callback to log Keras training progress using a logger.

        Args:
            logger (logging.Logger): The logger instance to send training logs to.
        """
        super().__init__()
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        """
        Logs metrics at the end of each epoch.
        """
        if logs is not None:
            log_message = f"Epoch {epoch + 1}: " + ", ".join(
                [f"{key}={value:.4f}" for key, value in logs.items()]
            )
            self.logger.info(log_message)

    def on_train_begin(self, logs=None):
        """
        Logs the start of training.
        """
        self.logger.info("Training started.")

    def on_train_end(self, logs=None):
        """
        Logs the end of training.
        """
        self.logger.info("Training completed.")
