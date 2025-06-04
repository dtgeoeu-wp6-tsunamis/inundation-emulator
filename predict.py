
import os
from src.emulator import Emulator
from datetime import datetime

"""
# Use model to create predictions.
# -emulator$ poetry run python -m predict
"""
GENERATED_DIR = "/home/nrr/projects/inundation-emulator/generated"
NAME="emulator"
RUNDIR ="/home/nrr/projects/inundation-emulator/generated/emulator_20250516_073408"
PREDICTION_DIR = "/home/nrr/projects/inundation-emulator/generated/predictions" 

TRAIN_SCENARIOS = "/home/nrr/projects/inundation-emulator/scenario.txt" #4196 training set
TRAIN_DIR = '/home/nrr/NGI/P/2022/02/20220296/Calculations/temp_emulator' 

VALIDATION_SCENARIOS = '/home/ebr/data/PTHA2020_runs_UMA/test/scenarios.txt' #about 100 events
VALIDATION_DIR = "/home/ebr/data/PTHA2020_runs_UMA/test"

# Test data replace with PTF events from Manuela
# TEST_DATA = '/home/ebr/projects/inundation-emulator/article_data/bottom_UMAPS_shuf.txt'
# TEST_DATA_DIR = "/mnt/NGI_disks/ebr/T/Tsunami/PTHA2020_runs_UMA"

TEST_SCENARIOS = '/home/nrr/projects/inundation-emulator/tests/data/scenarios.txt' #for ptf about 1066
TEST_DIR = "/home/nrr/projects/inundation-emulator/tests/data"

# TEST_SCENARIOS = '/home/nrr/NGI/P/2022/02/20220296/Calculations/from_Leonardo/20250512/data/scenarios.txt' #for ptf about 1066, not used as reading files from network caused segmentation fault
# TEST_DIR = "/home/nrr/NGI/P/2022/02/20220296/Calculations/from_Leonardo/20250512/data"

def main():
    emulator = Emulator(GENERATED_DIR, RUNDIR)
    
    # Create output folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(PREDICTION_DIR, emulator.id, f"preds_{timestamp}")

    # emulator.predict(                          #small test set for 6.4 deliverable
    #     scenarios_file=VALIDATION_SCENARIOS,
    #     input_dir=VALIDATION_DIR,
    #     output_dir=output_dir
    # )

    emulator.predict(                            #for main Catania SD 
        scenarios_file=TEST_SCENARIOS,
        input_dir=TEST_DIR,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()