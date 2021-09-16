import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display


# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def get_audio_data(data_dir):
    # data_dir = pathlib.Path('data/mini_speech_commands')
    if not data_dir.exists():
        tf.keras.utils.get_file(
          'mini_speech_commands.zip',
          origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
          extract=True,
          cache_dir='.', cache_subdir='data')


# !rm -rf '/content/data/mini_speech_commands/no/'
# !rm -rf '/content/data/mini_speech_commands/go/'
# !rm -rf '/content/data/mini_speech_commands/up/'


def preprocess_data(data_dir):
    # experiment only on selected words
    commands = 'down left yes right stop'.split()

    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    print('Number of total examples:', num_samples)
    print('Number of examples per label:',
      len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
    print('Example file tensor:', filenames[0])

    return filenames


# Train-Val-Test Split
def split_data(filenames):
    train_files = filenames[:3600]
    val_files = filenames[3600: 3600 + 800]
    test_files = filenames[-600:]

    print('Training set size', len(train_files))
    print('Validation set size', len(val_files))
    print('Test set size', len(test_files))

    return train_files, val_files, test_files


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


# AUTOTUNE = tf.data.AUTOTUNE
# #creating data in tensor format
# files_ds = tf.data.Dataset.from_tensor_slices(train_files)
# waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
#
#
# # displaying waveform for the data
# rows = 2
# cols = 2
# n = rows*cols
# fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
# for i, (audio, label) in enumerate(waveform_ds.take(n)):
#     r = i // cols
#     c = i % cols
#     ax = axes[r][c]
#     ax.plot(audio.numpy())
#     ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
#     label = label.numpy().decode('utf-8')
#     ax.set_title(label)
#
# # plt.show()


def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram


# for waveform, label in waveform_ds.take(1):
#     label = label.numpy().decode('utf-8')
#     spectrogram = get_spectrogram(waveform)
#
# print('Label:', label)
# print('Waveform shape:', waveform.shape)
# print('Spectrogram shape:', spectrogram.shape) # spectrogram as 2D image
# print('Audio playback')
# display.display(display.Audio(waveform, rate=16000))

# def plot_spectrogram(spectrogram, ax):
#     # Convert to frequencies to log scale and transpose so that the time is
#     # represented in the x-axis (columns).
#     log_spec = np.log(spectrogram.T)
#     height = log_spec.shape[0]
#     width = log_spec.shape[1]
#     X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
#     Y = range(height)
#     ax.pcolormesh(X, Y, log_spec, shading='auto')
#
#
#     fig, axes = plt.subplots(2, figsize=(12, 8))
#     timescale = np.arange(waveform.shape[0])
#     axes[0].plot(timescale, waveform.numpy())
#     axes[0].set_title('Waveform')
#     axes[0].set_xlim([0, 16000])
#     plot_spectrogram(spectrogram.numpy(), axes[1])
#     axes[1].set_title('Spectrogram')
#     # plt.show()


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


# def plot_spectrogram2(waveform_ds):
#     spectrogram_ds = waveform_ds.map(
#         get_spectrogram_and_label_id, num_parallel_calls=tf.data.AUTOTUNE)
#
#     rows = 2
#     cols = 2
#     n = rows*cols
#     fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
#     for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
#         r = i // cols
#         c = i % cols
#         ax = axes[r][c]
#         plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
#         ax.set_title(commands[label_id.numpy()])
#         ax.axis('off')
#     # plt.show()


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=tf.data.AUTOTUNE)
    return output_ds


def create_batches_for_train(data, batch_size):
    # creating batches and passing onto the model architecture
    # batch_size = 32

    data = data.batch(batch_size)
    # reduce read latency
    data = data.cache().prefetch(tf.data.AUTOTUNE)
    return data


def create_model(input_shape, train_ds, num_labels):
    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(train_ds.map(lambda x, _: x))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        preprocessing.Resizing(32, 32),
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    print(model.summary())
    return model


def compile_fit(model, train_ds, val_ds):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    EPOCHS = 20
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=3)
    )


def evaluate_test(model, test_ds):
    test_audio = []
    test_labels = []

    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)
    # print(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    # print(y_pred)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')


global commands
commands = 'down left yes right stop'.split()


def main():
    pass


    # install data
    data_dir = pathlib.Path('data/mini_speech_commands')
    get_audio_data(data_dir)

    commands = 'down left yes right stop'.split()

    # pre-process data
    files = preprocess_data(data_dir)

    # train-val-test split
    train_files, val_files, test_files = split_data(files)

    # these files are like dataframe, X being the data and Y is the label
    # both in tensor format
    train_ds = preprocess_dataset(train_files)
    # train_ds = spectrogram_ds
    val_ds = preprocess_dataset(val_files)
    test_ds = preprocess_dataset(test_files)

    for spectrogram, _ in train_ds.take(1):
        input_shape = spectrogram.shape
        # print(input_shape)

    num_labels = len(commands)

    train_ds_batch = create_batches_for_train(train_ds, 32)
    val_ds_batch = create_batches_for_train(val_ds, 32)
    # print(train_ds_batch)

    model = create_model(input_shape, train_ds,num_labels)

    #compile and fit
    compile_fit(model, train_ds_batch, val_ds_batch)
    model.save('model_save')

    # evaluate
    evaluate_test(model,test_ds)


if __name__ == "__main__":
    main()




