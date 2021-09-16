import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import models


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


audio_binary = tf.io.read_file('data/mini_speech_commands/stop/1aed7c6d_nohash_0.wav')
waveform = decode_audio(audio_binary)
print(waveform)


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


spectrogram = get_spectrogram(waveform)

print('Waveform shape:', waveform.shape)
print('Spectrogram shape:', spectrogram.shape) # spectrogram as 2D image
print('Audio playback')
# display.display(display.Audio(waveform, rate=16000))

new_model = tf.keras.models.load_model('model_save')


def evaluate_test(model, test_ds):
    test_audio = []
    test_labels = []


    toxic = set(['down', 'stop'])
    my_dict = {}
    my_dict['down'] = 0
    my_dict['left'] = 1
    my_dict['yes'] = 2
    my_dict['right'] = 3
    my_dict['stop'] = 4

    for audio in test_ds:
        test_audio.append(audio.numpy())
        # test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    # test_labels = np.array(test_labels)
    # # print(test_labels)
    print(test_audio.shape)
    check = np.reshape(test_audio, (1, 124, 129, 1))

    print(check.shape)
    y_pred = np.argmax(model.predict(check), axis=1)
    print(y_pred)


    for k, v in my_dict.items():
        if v == y_pred[0]:
            # print(v,k)
            if k in toxic:
                print('Yes, it is Toxic word')
            else:
                print('No, it is not toxic word')
    # y_true = test_labels
    #
    # test_acc = sum(y_pred == y_true) / len(y_true)
    # print(f'Test set accuracy: {test_acc:.0%}')


evaluate_test(new_model, spectrogram)