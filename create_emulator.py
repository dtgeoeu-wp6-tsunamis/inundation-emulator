from src.emulator import Emulator
from src.plotter import Plotter

"""
# Instantiate and build the model
# -emulator$ poetry run python -m create_emulator
"""
GENERATED_DIR = "/home/ebr/projects/inundation-emulator/generated"
TOPO_FILE = '/home/ebr/data/PTHA2020_runs_UMA/Catania/C_CT.grd'
TOPO_MASK = '/home/ebr/data/PTHA2020_runs_UMA/Catania/ct_mask.txt'
TRAIN_SCENARIOS ="/home/ebr/data/PTHA2020_runs_UMA/train_591/scenarios.txt"
TRAIN_DIR = '/home/ebr/data/PTHA2020_runs_UMA/train_591'
#TRAIN_SCENARIOS ="/home/ebr/data/PTHA2020_runs_UMA/train_164/scenarios.txt"
#TRAIN_DIR = '/home/ebr/data/PTHA2020_runs_UMA/train_164'
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
                         batch_size=10,
                         epochs=2000,
                         l2_callback_frequency=400,
                         save_model_frequency=400)
    emulator.save_model()
    
    # Create plots.
    plotter = Plotter(emulator.rundir)
    plotter.plot_l2_metrics_overlay()
    plotter.plot_training_history()
    

if __name__ == "__main__":
    main()