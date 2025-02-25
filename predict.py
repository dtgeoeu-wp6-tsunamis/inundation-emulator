
import os
from src.emulator import Emulator
from datetime import datetime

"""
# Use model to create predictions.
# -emulator$ poetry run python -m predict
"""
GENERATED_DIR = "/home/ebr/projects/inundation-emulator/generated"
NAME="emulator"
#RUNDIR ="/home/ebr/projects/inundation-emulator/generated/emulator_20250211_121156"
RUNDIR ="/home/ebr/projects/inundation-emulator/generated/emulator_20250225_125539"
PREDICTION_DIR = "/home/ebr/projects/inundation-emulator/generated/predictions" 

TRAIN_SCENARIOS = "/home/ebr/data/PTHA2020_runs_UMA/train_591/scenarios.txt"
TRAIN_DIR = '/home/ebr/data/PTHA2020_runs_UMA/train_591'
VALIDATION_SCENARIOS = '/home/ebr/data/PTHA2020_runs_UMA/test/scenarios.txt'
VALIDATION_DIR = "/home/ebr/data/PTHA2020_runs_UMA/test"

# Test data
#TEST_DATA = '/home/ebr/projects/inundation-emulator/article_data/bottom_UMAPS_shuf.txt'
#TEST_DATA_DIR = "/mnt/NGI_disks/ebr/T/Tsunami/PTHA2020_runs_UMA"

def main():
    emulator = Emulator(GENERATED_DIR, RUNDIR)
    
    # Create output folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(PREDICTION_DIR, emulator.id, f"preds_{timestamp}")

    emulator.predict(
        scenarios_file=VALIDATION_SCENARIOS,
        input_dir=VALIDATION_DIR,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()