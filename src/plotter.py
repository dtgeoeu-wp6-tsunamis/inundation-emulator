import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from src.logger_setup import get_logger

# Example usage
def main():
    rundir = "/home/ebr/projects/inundation-emulator/generated/emulator_20250121_100344"
    plotter = Plotter(rundir)
    
    plotter.plot_l2_metrics_overlay()
    plotter.plot_training_history()
    

class Plotter:
    """ Class for visualization of emulator results.
    """
    
    def __init__(self, rundir):
        self.rundir = rundir
        self.output_dir = os.path.join(self.rundir, "plots")
        self.l2_metrics_file = os.path.join(self.rundir, "l2_metrics.csv")  # Path to the metrics CSV file
        self.summary_file = os.path.join(self.rundir, "train_summary.csv")
        self.logger = get_logger(__name__, self.rundir)
        
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)


    def plot_l2_metrics_overlay(self):
        """
        Creates scatterplots of L2 norm vs. L2 error for all epochs, overlaying
        training and validation datasets with different colors.
        
        Args:
            csv_file (str): Path to the CSV file containing L2 metrics.
            output_dir (str): Directory to save the scatterplots.
        """

        # Load the CSV data
        df = pd.read_csv(self.l2_metrics_file)

        # Get all unique epoch values
        unique_epochs = df["epoch"].unique()

        for epoch in unique_epochs:
            # Filter data for the current epoch
            df_epoch = df[df["epoch"] == epoch]

            # Separate data for train and test datasets
            train_data = df_epoch[df_epoch["dataset"] == "train"]
            test_data = df_epoch[df_epoch["dataset"] == "validation"]

            # Scatterplot: Overlay train and test datasets
            plt.figure(figsize=(8, 6))

            if not train_data.empty:
                plt.scatter(train_data["l2_norm"], train_data["l2_error"], 
                            label="Train", alpha=0.7, color="blue")
            
            if not test_data.empty:
                plt.scatter(test_data["l2_norm"], test_data["l2_error"], 
                            label="Validation", alpha=0.7, color="orange")

            # Add title, labels, legend, and grid
            plt.title(f"L2 Norm vs. L2 Error (Epoch {epoch})")
            plt.xlabel("L2 Norm")
            plt.ylabel("L2 Error")
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()
            
            plt.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.7)
            plt.grid(True, which="major", linestyle="-", linewidth=0.8)
            
            # Add diagonal line y = x
            min_val = max(min(df_epoch["l2_norm"].min(), df_epoch["l2_error"].min()), 1e-3)
            max_val = max(df_epoch["l2_norm"].max(), df_epoch["l2_error"].max())
            x = np.logspace(np.log10(min_val), np.log10(max_val), 100)
            plt.plot(x, x, linestyle="--", color="red", label="y = x")

            # Save the plot
            plot_path = os.path.join(self.output_dir, f"scatter_epoch_{epoch}.png")
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Saved scatterplot for Epoch {epoch} at {plot_path}")


    def plot_training_history(self):
        """
        Loads training history from a CSV file and plots loss and MSE metrics
        with y-axis in log scale.

        Args:
            csv_file (str): Path to the CSV file containing training history.
            output_file (str, optional): Path to save the plot. If None, displays the plot interactively.
        """
        # Load the CSV file into a DataFrame
        history = pd.read_csv(self.summary_file)

        # Check for required columns
        required_columns = ['loss', 'mse', 'val_loss', 'val_mse']
        for col in required_columns:
            if col not in history.columns:
                raise ValueError(f"Missing required column: {col} in the CSV file.")

        # Plot loss metrics
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Training Loss', color='blue')
        plt.plot(history['val_loss'], label='Validation Loss', color='orange')
        plt.yscale('log')
        plt.title("Loss (Log Scale)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        filename_loss = os.path.join(self.rundir,"train_summary_loss.png")
        plt.savefig(filename_loss)
        self.logger.info(f"Saved train loss plot: {filename_loss}")
        plt.close()

        # Plot MSE metrics
        plt.figure(figsize=(10, 6))
        plt.plot(history['mse'], label='Training MSE', color='green')
        plt.plot(history['val_mse'], label='Validation MSE', color='red')
        plt.yscale('log')
        plt.title("Mean Squared Error (Log Scale)")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        filename_mse = os.path.join(self.rundir,"train_summary_mse.png")
        plt.savefig(filename_mse)
        self.logger.info(f"Saved train mse plot: {filename_mse}")
        plt.close()


if __name__ == "__main__":
    main()