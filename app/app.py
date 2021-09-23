from flask import Flask, render_template, request, url_for, redirect
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# os.chdir('app')
app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():
    # return str(res)
    request_type_str = request.method

    if request_type_str == 'GET':
        return render_template('index.html', href1='static/images/Toxic_Positivity.png', href4=url_for('music'))
    else:
        global file
        file = request.form['file']
        res, value, waveform = main(str(file))
        # print(waveform)
        # print(str(file))
        # return redirect(url_for('music'))
        if value == 0 or value == 4:
            return render_template('index.html', href1='static/images/positive.jfif', href2='Yes, it is Toxic word', href3=url_for('music2'), href4 = url_for('music'))
        else:
            return render_template('index.html', href1='static/images/negative.jpg', href2='No, it is not Toxic Word', href3=url_for('music2'), href4 = url_for('music'))


        # return str(res)


@app.route("/music", methods=['GET', 'POST'])
def music():
    request_type_str = request.method

    if request_type_str == 'GET':
        return render_template('music.html')
    else:
        file = request.form['file']
        return render_template('music.html', href3='static/audio_files/' + str(file), href6=url_for('index'))


@app.route("/music2", methods=['GET', 'POST'])
def music2():
    return render_template('music2.html', href3='static/audio_files/' + str(file), href5='static/spectrogram/file.png', href6=url_for('index'))


def evaluate_test(model, test_ds):
    test_audio = []

    toxic = set(['down', 'stop'])
    my_dict = {'down': 0, 'left': 1, 'yes': 2, 'right': 3, 'stop': 4}

    for audio in test_ds:
        test_audio.append(audio.numpy())

    test_audio = np.array(test_audio)
    # test_labels = np.array(test_labels)
    # # print(test_labels)
    # print(test_audio.shape)
    check = np.reshape(test_audio, (1, 124, 129, 1))

    # print(check.shape)
    y_pred = np.argmax(model.predict(check), axis=1)
    # print(y_pred)

    for k, v in my_dict.items():
        if v == y_pred[0]:
            # print(v,k)
            if k in toxic:
                return 'Yes, it is Toxic word', v
            else:
                return 'No, it is not toxic word', v


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)

    return waveform, spectrogram


def plot_spectrogram(spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns).
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec, shading='auto')


def main(filepath):
    audio_binary = tf.io.read_file('../test_files/' + filepath)
    waveform = decode_audio(audio_binary)

    waveform, spectrogram = get_spectrogram(waveform)

    fig, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(waveform.shape[0])
    axes[0].plot(timescale, waveform.numpy())
    axes[0].set_title('Waveform')
    axes[0].set_xlim([0, 16000])
    plot_spectrogram(spectrogram.numpy(), axes[1])
    axes[1].set_title('Spectrogram')
    strFile = "static/spectrogram/file.png"
    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig('static/spectrogram/file.png', dpi=100)
    # plt.show()

    # load model
    new_model = tf.keras.models.load_model('../model_save')

    # test
    result, v = evaluate_test(new_model, spectrogram)
    return result, v, waveform


if __name__ == '__main__':
    # main()
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=80)
