from src.emulator import Emulator

"""
# Instantiate and build the model
# -emulator$ poetry run python -m create_emulator
        center_longitude = 15.5
        center_latitude = 38.0
        projection = ccrs.ObliqueMercator(
            central_latitude=center_latitude,
            central_longitude=center_longitude,
            scale_factor=1.0
            )
        #projection = ccrs.UTM(zone=33, southern_hemisphere=False)
"""
GENERATED_DIR = "/home/ebr/projects/inundation-emulator/generated"
TOPO_FILE = '/home/ebr/data/PTHA2020_runs_UMA/Catania/C_CT.grd'
TOPO_MASK = '/home/ebr/data/PTHA2020_runs_UMA/Catania/ct_mask.txt'
TRAIN_SCENARIOS ="/home/ebr/data/PTHA2020_runs_UMA/train_591/scenarios.txt"
TRAIN_DIR = '/home/ebr/data/PTHA2020_runs_UMA/train_591'
VALIDATION_SCENARIOS = '/home/ebr/data/PTHA2020_runs_UMA/test/scenarios.txt'
VALIDATION_DIR = "/home/ebr/data/PTHA2020_runs_UMA/test"

# Test data
#TEST_DATA = '/home/ebr/projects/inundation-emulator/article_data/bottom_UMAPS_shuf.txt'
#TEST_DATA_DIR = "/mnt/NGI_disks/ebr/T/Tsunami/PTHA2020_runs_UMA"


def main():
    emulator = Emulator(GENERATED_DIR, TOPO_FILE, TOPO_MASK)
    model = emulator.model
    print("\nEncoder:")
    model.layers[0].summary()
    print("\nDecoder:")
    model.layers[1].summary()
    
    emulator.train_model(train_dir = TRAIN_DIR, 
                         train_scenarios = TRAIN_SCENARIOS,
                         validation_dir = VALIDATION_DIR,
                         validation_scenarios =  VALIDATION_SCENARIOS,
                         batch_size=32,
                         epochs=100)
    emulator.save_model()

if __name__ == "__main__":
    main()