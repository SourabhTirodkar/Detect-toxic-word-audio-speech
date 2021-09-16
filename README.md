# Detect-toxic-word-audio-speech

## This project detects toxic words when a audio sample of 1 sec is provided.


### Model Training:

Classes: 5 (down, left, right, stop, yes)<br>
[Model Training Script](model_train.py) <br>
[Model Test Script](test_script.py) <br>
[Saved Model](model_save)
------------------------

### Flask Application
Assumed  <b>'down'</b> and <b>'stop'</b> as toxic words<br>
[Application](app/app.py)