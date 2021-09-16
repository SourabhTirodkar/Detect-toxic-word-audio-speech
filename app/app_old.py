from flask import Flask, render_template, request
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models


app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello_world():
    acc = main()
    # return "<p>Hello, World!</p>"
    # return str(acc)
    return render_template('index.html')



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
          cache_dir='..', cache_subdir='data')


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


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=tf.data.AUTOTUNE)
    return output_ds



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

    return test_acc


global commands
commands = 'down left yes right stop'.split()


def main():

    # install data
    data_dir = pathlib.Path('../data/mini_speech_commands')
    get_audio_data(data_dir)

    commands = 'down left yes right stop'.split()

    # pre-process data
    files = preprocess_data(data_dir)

    # train-val-test split
    train_files, val_files, test_files = split_data(files)
    test_ds = preprocess_dataset(test_files)

    new_model = tf.keras.models.load_model('model_save')
    # evaluate
    test_acc = evaluate_test(new_model, test_ds)

    return test_acc