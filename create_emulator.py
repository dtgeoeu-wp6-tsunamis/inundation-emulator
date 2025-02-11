import os
import shutil
from src.emulator import Emulator
from src.plotter import Plotter
from src.data_reader import DataReader
from src.logger_setup import get_logger
from datetime import datetime

"""
# Instantiate and build the model
# -emulator$ poetry run python -m create_emulator
"""
GENERATED_DIR = "/home/ebr/projects/inundation-emulator/generated"
NAME="emulator"
RUNDIR = None

TOPO_FILE = '/home/ebr/data/PTHA2020_runs_UMA/Catania/C_CT.grd'
TOPO_MASK = '/home/ebr/data/PTHA2020_runs_UMA/Catania/ct_mask.npy'  # Needs to be a .npy file. Created from training data if it does not exist.
GRID_INFO_FILE = "/home/ebr/data/PTHA2020_runs_UMA/Catania/grid_info.json"

TRAIN_SCENARIOS = "/home/ebr/data/PTHA2020_runs_UMA/train_591/scenarios.txt"
TRAIN_DIR = '/home/ebr/data/PTHA2020_runs_UMA/train_591'
VALIDATION_SCENARIOS = '/home/ebr/data/PTHA2020_runs_UMA/test/scenarios.txt'
VALIDATION_DIR = "/home/ebr/data/PTHA2020_runs_UMA/test"

# Test data
#TEST_DATA = '/home/ebr/projects/inundation-emulator/article_data/bottom_UMAPS_shuf.txt'
#TEST_DATA_DIR = "/mnt/NGI_disks/ebr/T/Tsunami/PTHA2020_runs_UMA"

#RUNDIR = "/home/ebr/projects/inundation-emulator/generated/emulator_20250123_125901"

def main():
    global RUNDIR # Use global variable
    # Initialize model
    
    if RUNDIR is None:
        # Create rundir
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ID = f"{NAME}_{timestamp}"
        RUNDIR = os.path.join(GENERATED_DIR, ID)
        os.makedirs(RUNDIR, exist_ok=True)
        
        # Copy config files to rundir: train_scenarios, validation_scenarios, topofile
        shutil.copy(TOPO_FILE, os.path.join(RUNDIR, "topography.grd"))
        shutil.copy(TRAIN_SCENARIOS, os.path.join(RUNDIR, "train_scenarios.txt"))
        shutil.copy(VALIDATION_SCENARIOS, os.path.join(RUNDIR, "validation_scenarios.txt"))
        
        if os.path.isfile(TOPO_MASK):
            shutil.copy(TOPO_MASK, os.path.join(RUNDIR, "topomask.npy"))

        # Creates topomask on initialization if it does noyt exists.
        reader = DataReader(
            rundir=RUNDIR,
            scenarios_file=TRAIN_SCENARIOS,
            datadir=TRAIN_DIR,
        )
        
        if os.path.isfile(GRID_INFO_FILE):
            shutil.copy(GRID_INFO_FILE, os.path.join(RUNDIR, "grid_info.json"))
        else: 
            reader.store_grid_info(GRID_INFO_FILE)
    
    
    # All config files should be available in the rundir.
    emulator = Emulator(GENERATED_DIR, RUNDIR)
    
    emulator.train_model(train_dir = TRAIN_DIR, 
                         train_scenarios = os.path.join(RUNDIR, "train_scenarios.txt"),
                         validation_dir = VALIDATION_DIR,
                         validation_scenarios = os.path.join(RUNDIR, "validation_scenarios.txt"),
                         batch_size=20,
                         epochs=10,
                         l2_callback_frequency=5,
                         save_model_frequency=5)
    emulator.save_model()
    
    # Create plots.
    plotter = Plotter(emulator.rundir)
    plotter.plot_l2_metrics_overlay()
    plotter.plot_training_history()


if __name__ == "__main__":
    main()