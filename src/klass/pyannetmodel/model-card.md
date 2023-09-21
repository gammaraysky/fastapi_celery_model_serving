# SincNet Model Card

Last Updated: _August 2023_

<br>


# Model Details
The SincNet model is made up of a convolutional layer, 2 bidrectional LSTM layers and 2 linear layers with approximately 700k trainable parameters. The model objective is for Voice Activity Detection.

<br>

### _Model Size_
- approx. 700k parameters


<br>

### _Papers_
1. [Speaker Recognition from raw waveform with SincNet](https://arxiv.org/pdf/1808.00158.pdf)
2. [End-to-end speaker segmentation for overlap-aware resegmentation](https://arxiv.org/pdf/2104.04045.pdf)

<br>

### _Input_
Audio file(s) of .wav format

<br>

### _Output_

For each audio file used for inference, a .rttm file will be produced with the annotated voice/speech regions

<br>

# Intended Use
The model is designed for predicting voice/speech portions in an audio clip recorded by a far-field microphone, within a meeting room setting. It takes a raw audio as input and outputs a prediction of the voice/speech regions through a .rttm file.

<br>

# Metrics
Model performance was measured using the F1-Score metric for predicting the binary categories of voice vs noise.

<br>

# Training Data

The Model is trained on a total of 100 hours of meeting recordings in both English and Mandarin, provided by two publicly available datasets: (1) a subset of [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/), filtered for microphone recordings to represent far-field speech and (2) a subset of [Alimeeting dataset](http://openslr.org/119/) (far-field speech). Both these datasets contributed 50 hours of training data.

The training data is typically mono-channel and at a sample rate of 16kHz with accompanying annotations in .rttm format.

<br>


# Evaluation Data
The Model is evaluated on both the far-field evaluation set of (1) [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) and (2) [Alimeeting dataset](http://openslr.org/119/).

<br>

# Limitations

As with many deep learning models, the model performance heavily relies on the quality and distribution of the training data. The SincNet model is no exception and may be biased towards the training data. The model may also struggle to predict out-of-distribution data not found in the training set.

### _Interpretability_
There is limited ability to interpret how the model makes it's prediction, although basic error analysis has been done at the model training stage.

### _Training data_
There is no clear indication on the distance between the microphone and the speakers in the training data corpus, assumptions have been made that the environmental set-up in the training data corpus will not be significantly different compared to the model's use case.
