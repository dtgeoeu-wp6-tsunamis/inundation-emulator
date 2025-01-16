from src.emulator import Emulator

"""
# Instantiate and build the model
# -emulator$ poetry run -m tests.test_model
"""
GENERATED_DIR = "/home/ebr/projects/inundation-emulator/generated"

# Training data
DATA_DIR = '/home/ebr/data/PTHA2020_runs_UMA/train_164'
TOPO_FILE = '/home/ebr/data/PTHA2020_runs_UMA/Catania/C_CT.grd'
TOPO_MASK = '/home/ebr/data/PTHA2020_runs_UMA/Catania/ct_mask.txt'
SCENARIOS_FILE ="home/ebr/data/PTHA2020_runs_UMA/train_164/scenarios.txt"

# Test data
TEST_DATA = '/home/ebr/projects/inundation-emulator/article_data/bottom_UMAPS_shuf.txt'
TEST_DATA_DIR = "/mnt/NGI_disks/ebr/T/Tsunami/PTHA2020_runs_UMA"

def main():
    emulator = Emulator(GENERATED_DIR, TOPO_FILE, TOPO_MASK)
    model = emulator.model
    print("\nEncoder:")
    model.layers[0].summary()
    print("\nDecoder:")
    model.layers[1].summary()
    
    emulator.train_model(datadir = DATA_DIR, 
                         scenarios_file = SCENARIOS_FILE, 
                         batch_size=32, 
                         epochs=30)

if __name__ == "__main__":
    main()