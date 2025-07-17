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
GENERATED_DIR = "/home/nrr/projects/inundation-emulator/generated"
NAME="emulator"
RUNDIR = None
POIS = range(30, 45)  # Points of Interest, can be modified based on the site

TOPO_FILE = '/home/ebr/data/PTHA2020_runs_UMA/Catania/C_CT.grd'
TRAIN_SCENARIOS = "/home/nrr/projects/inundation-emulator/scenario.txt"
TRAIN_DIR = '/home/nrr/NGI/P/2022/02/20220296/Calculations/temp_emulator' #4196 set
VALIDATION_SCENARIOS = '/home/ebr/data/PTHA2020_runs_UMA/test/scenarios.txt'
VALIDATION_DIR = "/home/ebr/data/PTHA2020_runs_UMA/test"

# Optional
TOPO_MASK = '/home/ebr/data/PTHA2020_runs_UMA/Catania/ct_mask.npy'  # Needs to be a .npy file. Created from training data if it does not exist.
GRID_INFO_FILE = "/home/ebr/data/PTHA2020_runs_UMA/Catania/grid_info.json"
#TOPO_MASK = None
#GRID_INFO_FILE = None

# Test data replace with PTF events from Manuela
# TEST_DATA = '/home/ebr/projects/inundation-emulator/article_data/bottom_UMAPS_shuf.txt'
# TEST_DATA_DIR = "/mnt/NGI_disks/ebr/T/Tsunami/PTHA2020_runs_UMA"

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
        
        # Creates topomask on initialization if it does noyt exists.
        reader = DataReader(
            rundir=RUNDIR,
            scenarios_file=TRAIN_SCENARIOS,
            datadir=TRAIN_DIR,
        )
        
        if TOPO_MASK and os.path.isfile(TOPO_MASK):
            if not TOPO_MASK.endswith(".npy"):
                raise ValueError(f"Invalid file extension: {TOPO_MASK}. Expected a .npy file.")
            shutil.copy(TOPO_MASK, os.path.join(RUNDIR, "topomask.npy"))
        else: 
            reader.logger.warning(f"TOPO_MASK ({TOPO_MASK}) not found. Creating a new mask.")
            reader.create_mask()
        
        if GRID_INFO_FILE and os.path.isfile(GRID_INFO_FILE):
            shutil.copy(GRID_INFO_FILE, os.path.join(RUNDIR, "grid_info.json"))
        else: 
            reader.logger.warning(f"GRID_INFO_FILE ({GRID_INFO_FILE}) not found. Creating a new mask.")
            reader.store_grid_info()
    
    
    # All config files should be available in the rundir.
    emulator = Emulator(GENERATED_DIR, RUNDIR, POIS)
    
    emulator.train_model(train_dir = TRAIN_DIR, 
                         train_scenarios = os.path.join(RUNDIR, "train_scenarios.txt"),
                         validation_dir = VALIDATION_DIR,
                         validation_scenarios = os.path.join(RUNDIR, "validation_scenarios.txt"),
                         batch_size=20,
                         epochs=300,
                         l2_callback_frequency=20,
                         save_model_frequency=50)
    emulator.save_model()
    
    # Create plots.
    plotter = Plotter(emulator.rundir)
    plotter.plot_l2_metrics_overlay()
    plotter.plot_training_history()


if __name__ == "__main__":
    main()