from data_reader import DataReader
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import uuid

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
GENERATED_DIR = os.path.join(ROOT_DIR, 'generated')
SUMMARIES_TRAIN = os.path.join(DATA_DIR, 'train_test/train.txt')
SUMMARIES_VAL = os.path.join(DATA_DIR, 'train_test/validate.txt')

readers = {'train': DataReader(summaries_file=SUMMARIES_TRAIN, point_of_interest=35, shuffle_on_load=True, reload=True),
           'validate': DataReader(summaries_file=SUMMARIES_VAL, point_of_interest=35)}

# Create datasets
datasets = {}
for key, value in readers.items():
    print(key)
    datasets[key] = tf.data.Dataset.from_generator(
            generator=value.generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([481, 3]), tf.TensorShape([2]))
    )

conv_model = tf.keras.Sequential([
    # input [batch, timesteps (481), channels (3)]
    tf.keras.layers.Conv1D(filters=8,
                           kernel_size=20,
                           activation='relu',
                           input_shape=(481, 3),
                           use_bias=False),
    # output [batch, input-kernel_size + 1/strides, filters]
    tf.keras.layers.MaxPool1D(pool_size=11, strides=8),
    # output [batch, input-pool_size +1 /strides, filters]
    tf.keras.layers.Flatten(),
    # output [batch, input * filters]
    tf.keras.layers.Dense(units=10, use_bias=False, activation='relu'),
    # output [batch, units]
    tf.keras.layers.Dense(units=2, use_bias=False, activation='relu')
])

conv_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.005
    ),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError()]
)

# Only running trough training once.
history = conv_model.fit(
    x=datasets['train'].batch(40),
    epochs=10,
    #validation_data=datasets['validate'].batch(30),
    steps_per_epoch=80
)

# Create predictions and save model and predictions to folder

model_path = os.path.join(GENERATED_DIR, str(uuid.uuid4()))
os.mkdir(model_path)
print('Created folder: {}'.format(model_path))

pred_val = conv_model.predict(x=datasets['validate'].batch(40))
target_val = readers['validate'].get_targets()

np.savetxt(os.path.join(model_path, 'predictions_validate.txt'), pred_val)
conv_model.save(os.path.join(model_path, 'model'))



def plot_predictions(targets, predictions):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.scatter(x=targets[:, 0], y=predictions[:, 0], alpha=0.3)
    ax2.scatter(x=targets[:, 1], y=predictions[:, 1], alpha=0.3)

    ax1.set_title('Inundation area')
    ax2.set_title('Max height')
    ax1.set_xlabel('Targets')
    ax1.set_ylabel('Predictions')
    fig.savefig(fname=os.path.join(GENERATED_DIR, 'pred_plot'))

def plot_weights(layer):
    weights = conv_model.layers[0].weights[0]
    fig, axs = plt.subplots(3)

    axs[0].plot(weights[:, 0, layer])
    axs[1].plot(weights[:, 1, layer])
    axs[2].plot(weights[:, 2, layer])

    plt.show()