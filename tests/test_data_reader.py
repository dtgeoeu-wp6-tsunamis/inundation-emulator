from data_reader import DataReader
import tensorflow as tf

def main():
    """ Test for validation of DataReader.
    """
    DATA_DIR = '/home/ebr/data/PTHA2020_runs_UMA/train_164'
    TOPOFILE = '/home/ebr/data/PTHA2020_runs_UMA/Catania/C_CT.grd'
    TOPO_MASK = '/home/ebr/data/PTHA2020_runs_UMA/Catania/ct_mask.txt'

    pois = range(30,45)
    n_pois = len(pois)

    reader = DataReader(
        scenarios_file="/home/ebr/projects/inundation-emulator/training_set/scenarios.txt",
        pois=pois,
        datadir=DATA_DIR,
        topofile=TOPOFILE,
        topo_mask_file=TOPO_MASK,
        shuffle_on_load=False, 
        reload=False
    )
    
    dataset = tf.data.Dataset.from_generator(
            generator=reader.generator,
            output_signature=(
                    tf.TensorSpec(shape=(n_pois,481), dtype=tf.int32),
                    tf.TensorSpec(shape=(reader.topo_mask.sum()), dtype=tf.int32)
            )
    )

    for i, batch in enumerate(dataset.batch(10)):
        eta, flow_depth = batch
        print(f"Batch: {i}, eta.shape: {eta.shape}, flow_depth.shape: {flow_depth.shape}")
        

if __name__ == "__main__":
    main()