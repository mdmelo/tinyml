#!/usr/bin/env python
# coding: utf-8

# Useful links
#    https://petewarden.com/2023/07/29/accelerating-ai-with-the-raspberry-pi-picos-dual-cores/
#    https://github.com/usefulsensors/pico-tflmicro
#    https://blog.tensorflow.org/2021/09/TinyML-Audio-for-everyone.html
#    https://github.com/tensorflow/io?tab=readme-ov-file#tensorflow-version-compatibility
#    https://github.com/ArmDeveloperEcosystem/ml-audio-classifier-example-for-pico
#    https://github.com/ArmDeveloperEcosystem/ml-audio-classifier-example-for-pico/blob/main/ml_audio_classifier_example_for_pico.ipynb
#    https://towardsdatascience.com/fixed-point-dsp-for-data-scientists-d773a4271f7f
#    https://www.tensorflow.org/api_docs/python/tf/signal/stft
#    https://en.wikipedia.org/wiki/Short-time_Fourier_transform

# TODO
# Add microphone to Pico-W and test
#
# Add (external?) LED feedback via PWM to /files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/src/main.cpp
#
# Or... check that printf in the inference application is available via /dev/ttyACM0.
#       See 'build C pico-w application and flash it' for required CMakeLists.txt changes.
#       Looks like it might just work: /files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/CMakeLists.txt
#
# Understand the inference application code, does this do audio acquisition as well?
#
# Test ability to train with locally-acquired audio
#
# Write script to auto-create the inference-app/src/tflite_model.h file etc
#
# Understand/fix "triggered tf.function retracing" warning
#
# Fix issues preventing use of newer TensorFlow, eg 2.12.x
#
# Understand/fix quantize warning "Statistics for quantized inputs were expected, but not specified"

import sys
import os
import zipfile
import pickle
import pprint

# using TF2.x with Keras 2.x see https://keras.io/getting_started/ and https://github.com/tensorflow/tensorflow/issues/63849
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# avoid noisy log messages (this filters to Warning and above) - see https://github.com/tensorflow/tensorflow/issues/59779
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from numpy import pi as PI

import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_model_optimization as tfmot
import tensorflow.lite as tflite
import librosa
import cmsisdsp

# sudo apt-get install libportaudio2
import sounddevice as sd

# set True to display matplotlib plots
show_graphs = False

# set True to hear audio clips
listen_audio = False

# set True to use saved models
use_savedbaselinemodel = True
use_savedfinetunemodel = True

# set True to record and train with locally generated audio
trainwithlocalaudio = False

# access datafiles
basedir = '/files/pico/ML/audio-arm/'

# audio sampling params
SAMPLE_RATE = 16000
CHANNELS = 1

RANDOM_SEED = 42

# ## Introduction
#
# This tutorial will guide you through how to train a TensorFlow based audio classification Machine Learning (ML)
# model to detect a fire alarm sound. We’ll show you how to use TensorFlow Lite for Microcontrollers with Arm CMSIS-NN
# accelerated kernels to deploy the ML model to an Arm Cortex-M0+ based microcontroller (MCU) board for local on-device
# ML interferencing. Arm’s CMSIS-DSP library, which provides optimized Digital Signal Processing (DSP) function
# implementations for Arm Cortex-M  processors, will also be used to extract features from the real-time audio data prior
# to inference.
#
# See https://www.tensorflow.org/lite/microcontrollers)
# See https://arm-software.github.io/CMSIS_5/NN/html/index.html)
# See https://developer.arm.com/ip-products/processors/cortex-m/cortex-m0-plus
# See https://arm-software.github.io/CMSIS_5/DSP/html/index.html
# See https://developer.arm.com/ip-products/processors/cortex-m)
#
# While this guide focuses on detecting a fire alarm sound, it can be adapted for other sound classification tasks.
# You may need to adapt the feature extraction stages and/or adjust ML model architecture for your use case.


# ## What you need to to get started
#
# ### Development Environment
#
#  * Google Chrome
#  * Google Colab (https://colab.research.google.com/notebooks/)
#
# ### Hardware
#
# You’ll need one of the following development boards that are based on Raspberry Pi’s RP2040 MCU chip
# (https://www.raspberrypi.org/products/rp2040/) that was released early in 2021.
#
# #### SparkFun RP2040 MicroMod and MicroMod ML Carrier
#
# This is recommended for people who are new to electronics and microcontrollers. While it does cost a bit more
# than the option below, it is easier to assemble and does not require a soldering iron, knowing how to solder and
# how to wire up breadboards.
#
#  * [SparkFun MicroMod RP2040 Processor](https://www.sparkfun.com/products/17720)
#
# #### Raspberry Pi Pico and PDM microphone board
#
# This option is slightly lower in cost, however it requires a soldering iron and knowledge of how to wire a breadboard
# with electronic components.
#
#  * [Raspberry Pi Pico](https://www.raspberrypi.org/products/raspberry-pi-pico/)
#  * [Adafruit PDM MEMS Microphone Breakout](https://www.adafruit.com/product/3492)
#  * Half size or full size breadboard
#  * Jumper wires
#  * A USB-B micro cable to connect the board to your computer
#  * Soldering iron
#
#
# #### More information
#
# Both of the options above will allow you to collect real-time 16 kHz audio from a digital microphone and process the audio
# signal in real-time on the development board’s Arm Cortex-M0+ processor which operates at 125 MHz. The application running on
# rhe Arm Cortex-M0+ will have a Digital Signal Processing (DSP) stage to extract features from the audio signal, the extracted
# features will then be fed into a neural network to perform a classification task to determine if a fire alarm sound is present
# in the board’s environment.
#
#
# ### Hardware Setup
#
# #### Raspberry Pi Pico
#
# Follow the instructions from the Hardware Setup section of the "Create a USB Microphone with the Raspberry Pi Pico".
# See https://www.hackster.io/sandeep-mistry/create-a-usb-microphone-with-the-raspberry-pi-pico-cc9bd5#toc-hardware-setup-5
# for assembly instructions.
#

# ## Install dependencies

# ### Python Libraries

# get_ipython().system('pip install librosa matplotlib pandas "tensorflow==2.8.*" "tensorflow-io==0.24.*" "tensorflow-model-optimization==0.7.2"')
#
# get_ipython().system('pip install git+https://github.com/ARM-software/CMSIS_5.git@5.8.0#egg=CMSISDSP\\&subdirectory=CMSIS/DSP/PythonWrapper')


# ### Command line tools
#
# Install the command line tools we will need to build applications for the Raspberry Pi RP2040:
#
# tf.keras.utils.get_file('cmake-3.21.0-linux-x86_64.tar.gz',
#                         'https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-linux-x86_64.tar.gz',
#                         cache_dir='./',
#                         cache_subdir='tools',
#                         extract=True)
#
# tf.keras.utils.get_file('gcc-arm-none-eabi-10-2020-q4-major-x86_64-linux.tar.bz2',
#                         'https://developer.arm.com/-/media/Files/downloads/gnu-rm/10-2020q4/gcc-arm-none-eabi-10-2020-q4-major-x86_64-linux.tar.bz2',
#                         cache_dir='./',
#                         cache_subdir='tools',
#                         extract=True)
#
# get_ipython().system('apt-get install -y xxd')


# Add the downloaded and extracted tools to the `PATH` environmental variable, so we can use them
# later on without specifying the full path to them:

os.environ['PATH'] = f"{os.getcwd()}/tools/cmake-3.21.0-linux-x86_64/bin:{os.environ['PATH']}"
os.environ['PATH'] = f"{os.getcwd()}/tools/gcc-arm-none-eabi-10-2020-q4-major/bin:{os.environ['PATH']}"


# ### Raspberry Pi Pico SDK
#
# Use `git` to get the `v1.2.0` of the [Raspberry Pi Pico SDK - https://github.com/raspberrypi/pico-sdk
#
# get_ipython().run_cell_magic('shell', '', 'git clone --branch 1.2.0 https://github.com/raspberrypi/pico-sdk.git\ncd pico-sdk\ngit submodule init\ngit submodule update\n')


# Set the `PICO_SDK_PATH` environment variable to specify the location of the `pico-sdk`
#
# os.environ['PICO_SDK_PATH'] = f"{os.getcwd()}/pico-sdk"


# Change the code below to select the board you will be using for the remainder of the tutorial.
#
# By default the `PICO_BOARD` environment variable is set to `sparkfun_micromod` for the SparkFun RP2040 MicroMod.
# Set the value to `pico` if you are using a Raspberry Pi Pico board.

# for SparkFun MicroMod
# os.environ['PICO_BOARD'] = 'sparkfun_micromod'

# for Raspberry Pi Pico (uncomment next line)
os.environ['PICO_BOARD'] = 'pico'

print(f"PICO_BOARD env. var. set to '{os.environ['PICO_BOARD']}'")


# ### Project Files
#
# The source code for the inference application and Python utilities for Google Colab cloned using `git`:
#
# get_ipython().run_cell_magic('shell', '', 'git clone --recurse-submodules https://github.com/ArmDeveloperEcosystem/ml-audio-classifier-example-for-pico.git\n')


# Create symbolic links for the project files that we've cloned to the root Google Colab folder:
#
# get_ipython().run_cell_magic('shell', '', '
#                  ln -s ml-audio-classifier-example-for-pico/colab_utils colab_utils\n
#                  ln -s ml-audio-classifier-example-for-pico/inference-app inference-app\n')


# ## Baseline model
#
# Start by training a generic sound classifier model with TensorFlow using this dataset:
# ESC-50: Dataset for Environmental Sound Classification from https://github.com/karolpiczak/ESC-50.
#
# This will allow us to create a more generic model that is trained on a broader dataset, and then use
# Transfer Learning later on to fine tune it for our specific audio classification task.
#
# This model will be trained on the ESC-50 dataset, which contains 50 types of sounds; each sound category has
# 40 audio files that are 5 seconds each in length. Each audio file will be split into 1 second soundbites,
# and any soundbites that contain pure silence will be discarded.



# ### Prepare dataset
#
# #### Download and extract
#
# The ESC-50 dataset is downloaded and extracted to the `datasets` folder using tf.keras.utils.get_file API
# see https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file.
#
# tf.keras.utils.get_file('esc-50.zip',
#                         'https://github.com/karoldvl/ESC-50/archive/master.zip',
#                         cache_dir='./',
#                         cache_subdir='datasets',
#                         extract=True)

def load_wav(filename, desired_sample_rate, desired_channels):
    try:
        file_contents = tf.io.read_file(filename)
        wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=desired_channels)
        wav = tf.squeeze(wav, axis=-1)
    except:
        # fallback to librosa if the wav file cannot be read with TF
        filename = tf.cast(filename, tf.string)
        wav, sample_rate = librosa.load(filename.numpy().decode('utf-8'), sr=None, mono=(desired_channels == 1))

    wav = tfio.audio.resample(wav, rate_in=tf.cast(sample_rate, dtype=tf.int64), rate_out=tf.cast(desired_sample_rate, dtype=tf.int64))
    return wav

def load_wav_for_map(fullpath, label, fold):
    wav = tf.py_function(load_wav, [fullpath, SAMPLE_RATE, CHANNELS], tf.float32)
    return wav, label, fold

@tf.function
def split_wav(wav, width, stride):
    return tf.map_fn(fn=lambda t: wav[t * stride:t * stride + width], elems=tf.range((tf.shape(wav)[0] - width) // stride), fn_output_signature=tf.float32)

@tf.function
def wav_not_empty(wav):
    return tf.experimental.numpy.any(wav)

def split_wav_for_flat_map(wav, label, fold):
    wavs = split_wav(wav, width=16000, stride=4000)
    labels = tf.repeat(label, tf.shape(wavs)[0])
    folds = tf.repeat(fold, tf.shape(wavs)[0])
    return tf.data.Dataset.from_tensor_slices((wavs, labels, folds))

@tf.function
def create_spectrogram(samples):
    return tf.abs(
        # compute short-time Fourier Transform of the signal
        # https://en.wikipedia.org/wiki/Short-time_Fourier_transform
        # https://www.tensorflow.org/api_docs/python/tf/signal/stft
        tf.signal.stft(samples, frame_length=256, frame_step=128)
    )

# create `plot_spectrogram` function to plot the spectrogram representation using `matplotlib`
def plot_spectrogram(spectrogram, vmax=None, title=None):
    transposed_spectrogram = tf.transpose(spectrogram)

    fig = plt.figure(figsize=(8,6))
    height = transposed_spectrogram.shape[0]
    X = np.arange(transposed_spectrogram.shape[1])
    Y = np.arange(height * int(SAMPLE_RATE / 256), step=int(SAMPLE_RATE / 256))

    im = plt.pcolormesh(X, Y, tf.transpose(spectrogram), vmax=vmax)
    fig.colorbar(im)
    if title:
        plt.title(title)
    plt.show()


def plot_spectrogram_log(spectrogram, fig, ax, title=None):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
        
    # Convert the frequencies to log scale and transpose, so 
    # that the time is represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)
    
    im = plt.pcolormesh(X, Y, tf.transpose(spectrogram), vmax=None)
    fig.colorbar(im)
    if title:
        ax.set_title(title)


def create_spectrogram_for_map(samples, label, fold):
    return create_spectrogram(samples), label, fold


def get_arm_spectrogram(waveform):
    # fixed point equivalent of TF short-time fourier transform
    num_frames = int(1 + (len(waveform) - window_size) // step_size)
    fft_size = int(window_size // 2 + 1)

    # Convert the audio to q15
    waveform_q15 = cmsisdsp.arm_float_to_q15(waveform)

    # Create empty spectrogram array
    spectrogram_q15 = np.empty((num_frames, fft_size), dtype = np.int16)

    start_index = 0

    for index in range(num_frames):
        # Take the window from the waveform.
        window = waveform_q15[start_index:start_index + window_size]

        # Apply the Hanning Window.
        window = cmsisdsp.arm_mult_q15(window, hanning_window_q15)

        # Calculate the FFT, shift by 7 according to docs
        window = cmsisdsp.arm_rfft_q15(rfftq15, window)

        # Take the absolute value of the FFT and add to the Spectrogram.
        spectrogram_q15[index] = cmsisdsp.arm_cmplx_mag_q15(window)[:fft_size]

        # Increase the start index of the window by the overlap amount.
        start_index += step_size

    # Convert to numpy output ready for keras
    return cmsisdsp.arm_q15_to_float(spectrogram_q15).reshape(num_frames,fft_size) * 512


@tf.function
def create_arm_spectrogram_for_map(wav, label, fold):
    spectrogram = tf.py_function(get_arm_spectrogram, [wav], tf.float32)

    return spectrogram, label, fold


def add_white_noise(audio):
    # generate noise and the scalar multiplier
    noise = tf.random.uniform(shape=tf.shape(audio), minval=-1, maxval=1)
    noise_scalar = tf.random.uniform(shape=[1], minval=0, maxval=0.2)

    # add them to the original audio
    audio_with_noise = audio + (noise * noise_scalar)

    # final clip the values to ensure they are still between -1 and 1
    audio_with_noise = tf.clip_by_value(audio_with_noise, clip_value_min=-1, clip_value_max=1)

    return audio_with_noise


def add_random_silence(audio):
    audio_mask = tf.random.categorical(tf.math.log([[0.2, 0.8]]), num_samples=tf.shape(audio)[0])
    audio_mask = tf.cast(audio_mask, dtype=tf.float32)
    audio_mask = tf.squeeze(audio_mask, axis=0)

    # multiply the audio input by the mask
    augmented_audio = audio * audio_mask
    return augmented_audio


def add_audio_mixup(audio, mixup_audio):
    # randomly generate a scalar
    noise_scalar = tf.random.uniform(shape=[1], minval=0, maxval=1)

    # add the background noise to the audio
    augmented_audio = audio + (mixup_audio * noise_scalar)

    #final clip the values so they are stil between -1 and 1
    augmented_audio = tf.clip_by_value(augmented_audio, clip_value_min=-1, clip_value_max=1)

    return augmented_audio


def play_audio_data(audiodata, sample_rate=SAMPLE_RATE):
    sd.play(audiodata, sample_rate)
    sd.wait()


def calculate_ds_len(ds):
    count = 0
    for _, _, _ in ds:
        count += 1
    return count


def split_full_dataset(full_ds, prefetch=False):
    full_ds_size = calculate_ds_len(full_ds)
    print(f'full_ds_size = {full_ds_size}')

    full_ds = full_ds.shuffle(full_ds_size)

    train_ds_size = int(0.60 * full_ds_size)
    val_ds_size = int(0.20 * full_ds_size)
    test_ds_size = int(0.20 * full_ds_size)

    train_ds = full_ds.take(train_ds_size)

    remaining_ds = full_ds.skip(train_ds_size)
    val_ds = remaining_ds.take(val_ds_size)
    test_ds = remaining_ds.skip(val_ds_size)

    # remove the folds column as it's no longer needed
    remove_fold_column = lambda spectrogram, label, fold: (tf.expand_dims(spectrogram, axis=-1), label)

    # datasets are 'tensorflow.python.data.ops.dataset_ops.MapDataset'
    train_ds = train_ds.map(remove_fold_column)
    val_ds = val_ds.map(remove_fold_column)
    test_ds = test_ds.map(remove_fold_column)

    # dataset.prefetch() will load a PrefetchDataset:
    # when the GPU is working on forward / backward propagation on the current 
    # batch, we want the CPU to process the next batch of data so that it is 
    # immediately ready. As the most expensive part of the computer, we want the 
    # GPU to be fully used all the time during training. We call this consumer/
    # producer overlap, where the consumer is the GPU and the producer is the CPU.
    #
    # Prefetch overlaps the preprocessing and model execution of a training step. 
    # While the model is executing training step s, the input pipeline is reading 
    # the data for step s+1. Doing so reduces the step time to the maximum (as 
    # opposed to the sum) of the training and the time it takes to extract the data.
    # see https://www.tensorflow.org/guide/data_performance#prefetching
    if prefetch:
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    else:
        # The tf.data.Dataset.cache transformation can cache a dataset, either in 
        # memory or on local storage. This will save some operations (like file opening 
        # and data reading) from being executed during each epoch. The next epochs 
        # will reuse the data cached by the cache transformation.
        # see https://www.tensorflow.org/guide/data_performance#caching
        train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds



if  not use_savedbaselinemodel:
    # #### Load dataset metadata
    #
    #  read the `datasets/ESC-50-master/meta/esc50.csv` file which contains the
    # metadata for the audio files in the ESC-50 dataset:

    esc50_complete_csv = basedir + 'datasets/ESC-50-master/meta/esc50.csv'
    esc50_small_csv = basedir + 'datasets/ESC-50-master/meta/esc50-small.csv'
    esc50_single_csv = basedir + 'datasets/ESC-50-master/meta/esc50-single.csv'
    base_data_path = basedir + 'datasets/ESC-50-master/audio/'
    usecsv = esc50_complete_csv

    df = pd.read_csv(usecsv)
    print("read {}\nshape: ".format(usecsv), df.shape, "\n", df.head())


    # add new column with the `fullpath` of the wave files:
    base_data_path = basedir + 'datasets/ESC-50-master/audio/'
    df['fullpath'] = df['filename'].map(lambda x: os.path.join(base_data_path, x))
    print("\n\nadded fullpath column: \n", df.head())


    # #### Load wave file data
    #
    # Define a new function named `load_wav` to load audio samples from a wave file using TensorFlow's
    # tf.io.read_file API - see https://www.tensorflow.org/api_docs/python/tf/io/read_file and
    # tf.audio.decode_wav API - see https://www.tensorflow.org/api_docs/python/tf/audio/decode_wav.
    #
    # The tfio.audio.resample API - see https://www.tensorflow.org/io/api_docs/python/tfio/audio/resample -
    # will be used to resample the audio samples at the specified sampling rate.
    #
    # librosa is a Python module for audio procession (see https://librosa.org/).  The librosa load API - see
    # https://librosa.org/doc/main/generated/librosa.load.html - will be used as a fallback if TensorFlow is unable
    # to decode the wave file.


    # load the first wave file, which is a sound of a dog barking, from the pandas `DataFrame`, and
    # plot it overtime using matplotlib.
    test_wav_file_path = df['fullpath'][0]
    test_wav_data = load_wav(test_wav_file_path, SAMPLE_RATE, CHANNELS)

    if show_graphs:
        plt.plot(test_wav_data)
        plt.title("test wav file: {}".format(test_wav_file_path))
        plt.show()

    if listen_audio:
        play_audio_data(test_wav_data)

    # zoom in and only plot samples `32000` to `48000`so we get a closer plot of the audio samples
    # in the wave file in the 2 to 3 second span:
    if show_graphs:
        _ = plt.plot(test_wav_data[32000:48000])
        plt.title("samples 32000 -> 48000".format(test_wav_file_path))
        plt.show()


    # Use the tf.data.Dataset TensorFlow API - see https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
    # to create a pipeline that loads all wave file data from the dataset.
    fullpaths = df['fullpath']
    targets = df['target']
    folds = df['fold']

    fullpaths_ds = tf.data.Dataset.from_tensor_slices((fullpaths, targets, folds))
    print("fullpath dataset: \n", fullpaths_ds.element_spec)

    # Map each `fullpath` value to wave file samples:
    wav_ds = fullpaths_ds.map(load_wav_for_map)
    print("wav dataset: \n", wav_ds.element_spec)


    # #### Split Wave file data
    #
    # We would like to train the model on 1 second soundbites, so we must split up the 5 seconds of
    # audio per item in the ESC-50 dataset to slices of 16000 samples. We will also stride the original audio
    # samples `4000` samples at a time, and filter out any sound chunks that contain pure silence.

    # avoid noisy log messages - see https://github.com/tensorflow/tensorflow/issues/59779
    #
    # 2024-04-14 08:46:36.457393: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0]
    # (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT:
    # You must feed a value for placeholder tensor 'args_0' with dtype float [[{{node args_0}}]]
    split_wav_ds = wav_ds.flat_map(split_wav_for_flat_map)
    split_wav_ds = split_wav_ds.filter(lambda x, y, z: wav_not_empty(x))
    print("completed splitting ESC-50 5 second soundbites into 1 second bites")


    # plot the first 5 soundbites over time using `matplotlib`:
    for ndx, data in enumerate(split_wav_ds.take(5)):
        wav, _, _ = data
        if show_graphs:
            plt.plot(wav)
            plt.title("soundbite #{}".format(ndx+1))
            plt.show()


    # #### Create Spectrograms
    #
    # Rather than passing in the time series data directly into our TensorFlow model, we will transform the audio
    # data into an audio spectrogram representation. This will create a 2D representation of the audio signal’s frequency
    # content over time.
    #
    # The input audio signal we will use will have a sampling rate of 16kHz, this means one second of audio will contain 16,000
    # samples. Using TensorFlow’s tf.signal.stft function - see https://www.tensorflow.org/api_docs/python/tf/signal/stft -
    # we will transform a 1 second audio signal into a 2D tensor representation. We will choose a frame length of 256 and a
    # frame step of 128, so the output of this feature extraction stage will be a Tensor that has a shape of `(124, 129)`.

    # STFT = short-time Fourier transform  - a Fourier-related transform used to determine the sinusoidal frequency
    # and phase content of local sections of a signal as it changes over time. In practice, the procedure for computing STFTs
    # is to divide a longer time signal into shorter segments of equal length and then compute the Fourier transform separately
    # on each shorter segment. This reveals the Fourier spectrum on each shorter segment. One then usually plots the changing
    # spectra as a function of time, known as a spectrogram or waterfall plot.

    # take the same 2 - 3 second interval of the first dog barking wave file and create it's spectrogram representation
    spectrogram = create_spectrogram(test_wav_data[32000:48000])
    print("created spectrogram, shape:\n", spectrogram.shape)

    if show_graphs:
        fig, axes = plt.subplots(2, figsize=(12, 8))
        timescale = np.arange(test_wav_data[32000:48000].shape[0])
        axes[0].plot(timescale, test_wav_data[32000:48000].numpy())
        axes[0].set_title('audio waveform')
        axes[0].set_xlim([0, 16000])

        plot_spectrogram_log(spectrogram.numpy(), fig, axes[1], "spectrogram")
        plt.suptitle("2.0s->3.0s audio")
        plt.show()

    # map each split wave item to a spectrogram:
    spectrograms_ds = split_wav_ds.map(create_spectrogram_for_map)
    print(spectrograms_ds.element_spec)

    # plot the first 5 spectrograms in the dataset:
    if show_graphs:
        for ndx, data in enumerate(spectrograms_ds.take(5)):
            s, _, _ = data
            plot_spectrogram(s, title="spectrogram #{}".format(ndx+1))


    # ### Split Dataset
    #
    # Before we start training the ML classifier model, we must split the dataset up in three parts: training, validation, and test.
    #
    # We will use the same technique in TensorFlow's Transfer learning with YAMNet for environmental sound classification guide -
    # see https://www.tensorflow.org/tutorials/audio/transfer_learning_audio#split_the_data - and use the `fold` column of the
    # ESC-50 dataset to determine the split.
    #
    # Before splitting the dataset, set a random seed for reproducibility:
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Entries with a `fold` value of less than 4 will used for training, the ones with a `value` will be used for validation, and
    # finally the remaining items with be used for testing.
    #
    # The `fold` column will be removed as it is no longer needed, and the dimensions of the spectrogram shape will be expanded
    # from `(124, 129)` to `(124, 129, 1)`. The training items will also be shuffled.
    cached_ds = spectrograms_ds.cache()

    train_ds = cached_ds.filter(lambda spectrogram, label, fold: fold < 4)
    val_ds = cached_ds.filter(lambda spectrogram, label, fold: fold == 4)
    test_ds = cached_ds.filter(lambda spectrogram, label, fold: fold > 4)

    # remove the folds column as it's no longer needed
    remove_fold_column = lambda spectrogram, label, fold: (tf.expand_dims(spectrogram, axis=-1), label)

    train_ds = train_ds.map(remove_fold_column)
    val_ds = val_ds.map(remove_fold_column)
    test_ds = test_ds.map(remove_fold_column)

    train_ds = train_ds.cache().shuffle(1000, seed=RANDOM_SEED).batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
    print("training, validation and test datasets are now created")

    # ### Train Model
    #
    # Now that we have the features extracted from the audio signal, we can create a model using TensorFlow’s Keras  API.
    # The model will consist of 8 layers:
    #
    #  1. An input layer.
    #  2. A preprocessing layer, that will resize the input tensor from 124x129x1 to 32x32x1.
    #  3. A normalization layer, that will scale the input values between -1 and 1
    #  4. A 2D convolution layer with: 8 filters, a kernel size of 8x8, and stride of 2x2, and ReLU activation function.
    #  5. A 2D max pooling layer with size of 2x2
    #  6. A flatten layer to flatten the 2D data to 1D
    #  7. A dropout layer, that will help reduce overfitting during training
    #  8. A dense layer with 50 outputs and a softmax activation function, which outputs the likelihood of the sound category (between 0 and 1).
    #

    # Before we build the model using Tensorflow Keras API's - see https://www.tensorflow.org/api_docs/python/tf/keras,
    # we will create normalization layer and feed in all the spectrogram dataset items.

    for spectrogram, _, _ in cached_ds.take(1):
        input_shape = tf.expand_dims(spectrogram, axis=-1).shape
        print('Input shape:', input_shape)

    print("creating normalization layer...")

    # The normalization layer will shift and scale inputs into a distribution centered around
    # 0 with standard deviation 1. It accomplishes this by precomputing the mean and variance
    # of the data, and calling `(input - mean) / sqrt(var)` at runtime.
    #
    # The mean and variance values for the layer must be either supplied on
    # construction or learned via `adapt()`. `adapt()` will compute the mean and
    # variance of the data and store them as the layer's weights. `adapt()` should
    # be called before `fit()`, `evaluate()`, or `predict()`.
    #
    # For an overview and full list of preprocessing layers, see the preprocessing
    # [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()

    # WARNING:tensorflow:5 out of the last 5 calls to <function pfor.<locals>.f at 0x7efc7c15e5c0> triggered
    # tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to
    #  (1) creating @tf.function repeatedly in a loop,
    #  (2) passing tensors with different shapes,
    #  (3) passing Python objects instead of tensors.
    #
    # For (1), please define your @tf.function outside of the loop.
    # For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing.
    # For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and
    #          https://www.tensorflow.org/api_docs/python/tf/function for  more details.

    norm_layer.adapt(cached_ds.map(lambda x, y, z: tf.reshape(x, input_shape)))
    print("normalization layer done")


    # Define the sequential 8 layer model described above:
    print("creating the TF model...")
    baseline_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.experimental.preprocessing.Resizing(32, 32, interpolation="nearest"),
        norm_layer,
        tf.keras.layers.Conv2D(8, kernel_size=(8,8), strides=(2, 2), activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(50, activation='softmax')
    ])

    print("TF model create finished")
    baseline_model.summary()

    # Compile the model with `accuracy` metrics, an Adam optimizer and a sparse categorical crossentropy loss function.
    # Specify early stopping and dynamic learning rate scheduler callbacks for training.

    METRICS = [
        "accuracy",
    ]

    # Adam optimization is a stochastic gradient descent method that is based on adaptive
    # estimation of first-order and second-order moments,
    #
    # SparseCategoricalCrossentropy computes the crossentropy loss between the labels
    # and predictions.
    baseline_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=METRICS,
    )

    def scheduler(epoch, lr):
        if epoch < 100:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(verbose=1, patience=25),
        tf.keras.callbacks.LearningRateScheduler(scheduler)
    ]


    # Train the model:
    print("model training started...")
    EPOCHS = 250
    history = baseline_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    
    # show model performance
    metrics = history.history
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Baseline Model Loss [CrossEntropy]')
    
    plt.subplot(1,2,2)
    plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
    plt.legend(['accuracy', 'val_accuracy'])
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Baseline Model Accuracy [%]')
    plt.show()
    plt.close()

    # Evaluate the loss and accuracy of the test dataset:
    print("model training finshed, evaluating loss and accuracy using test dataset")

    # Epoch 41: early stopping
    # model training finshed, evaluating loss and accuracy using test dataset

    #   Accuracy
    #
    #     Accuracy is a method for measuring a classification model’s performance. It is typically expressed as a percentage.
    #     Accuracy is the percentage of predictions where the predicted value is equal to the true value. It is binary (true/false)
    #     for a particular sample. Accuracy is often graphed and monitored during the training phase though the value is often
    #     associated with the overall or final model accuracy. Accuracy is easier to interpret than loss.
    #
    #   Loss
    #
    #     A loss function, also known as a cost function, takes into account the probabilities or uncertainty of a prediction based on
    #     how much the prediction varies from the true value. This gives us a more nuanced view into how well the model is performing.
    #
    #     Unlike accuracy, loss is not a percentage — it is a summation of the errors made for each sample in training or validation sets.
    #     Loss is often used in the training process to find the "best" parameter values for the model (e.g. weights in neural network).
    #     During the training process the goal is to minimize this value.
    #
    #     Unlike accuracy, loss may be used in both classification and regression problems.
    #
    #       1/Unknown - 1s 715ms/step   - loss: 3.0822 - accuracy: 0.4062
    #      11/Unknown - 1s   5ms/step   - loss: 3.2630 - accuracy: 0.2841
    #      20/Unknown - 1s   6ms/step   - loss: 4.1075 - accuracy: 0.2000
    #      29/Unknown - 1s   6ms/step   - loss: 4.9819 - accuracy: 0.1907
    #      38/Unknown - 1s   6ms/step   - loss: 5.0779 - accuracy: 0.1924
    #      47/Unknown - 1s   6ms/step   - loss: 4.5339 - accuracy: 0.2427
    #      56/Unknown - 1s   6ms/step   - loss: 4.4093 - accuracy: 0.2282
    #      66/Unknown - 1s   6ms/step   - loss: 4.2783 - accuracy: 0.2268
    #      75/Unknown - 1s   6ms/step   - loss: 4.1174 - accuracy: 0.2396
    #      84/Unknown - 1s   6ms/step   - loss: 4.2615 - accuracy: 0.2385
    #      93/Unknown - 1s   6ms/step   - loss: 4.4185 - accuracy: 0.2352
    #     102/Unknown - 1s   6ms/step   - loss: 4.4326 - accuracy: 0.2390
    #     112/Unknown - 1s   6ms/step   - loss: 4.5757 - accuracy: 0.2450
    #     122/Unknown - 1s   6ms/step   - loss: 4.3965 - accuracy: 0.2569
    #     132/Unknown - 1s   6ms/step   - loss: 4.3541 - accuracy: 0.2526
    #     141/Unknown - 2s   6ms/step   - loss: 4.2947 - accuracy: 0.2473
    #     150/Unknown - 2s   6ms/step   - loss: 4.2106 - accuracy: 0.2475
    #     160/Unknown - 2s   6ms/step   - loss: 4.1213 - accuracy: 0.2502
    #     168/Unknown - 2s   6ms/step   - loss: 4.1048 - accuracy: 0.2504
    #     178/Unknown - 2s   6ms/step   - loss: 4.0331 - accuracy: 0.2598
    #
    # 187/187 [==============================] - 2s 6ms/step - loss: 3.9881 - accuracy: 0.2619
    # [3.9881081581115723, 0.2618529200553894]

    print(baseline_model.evaluate(test_ds))

    # The baseline model has a relatively low accuracy ~24%, however in the next steps we will use
    # it as a starting point to fine tune a more accurate model for our use case.

    # Save the model:
    baseline_model.save("baseline_model")


    # Create a zip file of the saved model, for download purposes
    # ipython only
    # get_ipython().system('zip -r baseline_model.zip baseline_model')
    with zipfile.ZipFile('baseline_model.zip', 'w', zipfile.ZIP_DEFLATED) as myzip:
        myzip.write('baseline_model')

else:
    # load the saved baseline model
    # Model: "sequential"
    # Layer (type)                   Output Shape              Param #
    # =================================================================
    # resizing (Resizing)            (None, 32, 32, 1)           0
    #
    # normalization (Normalization)  (None, 32, 32, 1)           3

    # conv2d (Conv2D)                (None, 13, 13, 8)         520

    # max_pooling2d (MaxPooling2D)   (None, 6, 6, 8)             0

    # flatten (Flatten)              (None, 288)                 0

    # dropout (Dropout)              (None, 288)                 0

    # dense (Dense)                  (None, 50)              14450

    #=================================================================
    # Total params: 14,973
    # Trainable params: 14,970
    # Non-trainable params: 3
    #_________________________________________________________________

    # (venv310) mike@debian-x250:/files/pico/ML/audio-arm$ tree baseline_model
    # baseline_model
    # ├── assets
    # ├── keras_metadata.pb
    # ├── saved_model.pb
    # └── variables
    #    ├── variables.data-00000-of-00001
    #    └── variables.index

    print("loading saved baseline model")
    baseline_model = tf.keras.models.load_model('baseline_model')
    baseline_model.summary()


# ## Transfer Learning
#
# Now we will use Transfer Learning and change the classification head of the model to
# train a binary classification model for fire alarm sounds.
#
# Transfer Learning is the process of retraining a model that has been developed for a
# task to complete a new similar task. The idea is that the model has learned transferable
# "skills" and the weights and biases can be used in other models as a starting point.
#
# Transfer learning is very common in computer vision. Big data companies spend weeks
# training models on ImageNet, this is not possible for most people and so people reuse
# the models built in these research companies to complete their own tasks. A model
# designed to recognise 1000 different objects in a image can be adapted to recognise
# other or similar objects.
#
# As humans we use transfer learning too. The skills you developed to learn to walk
# could also be used to learn to run later on.
#
# In a neural network, the first few layers of a model start to perform a "feature extraction"
# such as finding shapes, edges and colors. The layers later on are used as classifiers;
# they take the extracted features and classify them.
#
# You can find more information and visualizations about this here https://yosinski.com/deepvis.
#
# Because of this, we can assume the first few layers have learned quite general feature
# extraction techniques that can be applied to all similar tasks and so we can freeze
# all these layers. The classifier layer will need to be trained based on the new classes.
#
# To do this, we break the process into two steps:
#
# 1. Freeze the "backbone" of the model and train the head with a fairly high learning rate.
#    We slowly reduce the learning rate.
#
# 2. Unfreeze the "backbone" and fine-tune the model with a low learning rate.




# ### Dataset
#
# We have collected 10 fire alarm clips from freesound.org (https://freesound.org/) and
# BigSoundBank.com (https://bigsoundbank.com/).  Background noise clips from the SpeechCommands
# dataset (https://www.tensorflow.org/datasets/catalog/speech_commands) will be used for non-fire
# alarm sounds. This dataset is small and represents the sort of data you might expect to
# see in the real world. Data augmentation techniques will be used to supplement the
# training data we’ve collected.

# ### Download datasets
#
# We've created an archive with the following wave files:
#
#  * https://freesound.org/people/rayprice/sounds/155006/ ([CC BY 3.0 license](https://creativecommons.org/licenses/by/3.0/))
#
#  * https://freesound.org/people/deleted_user_2104797/sounds/164686/ ([CC0 1.0 license](https://creativecommons.org/publicdomain/zero/1.0/))
#
#  * https://freesound.org/people/AdamWeeden/sounds/255180/ ([CC BY 3.0 license](https://creativecommons.org/licenses/by/3.0/))
#
#  * https://freesound.org/people/MoonlightShadow/sounds/325367/([CC0 1.0 license](https://creativecommons.org/publicdomain/zero/1.0/))
#
#  * https://freesound.org/people/SpliceSound/sounds/369847/ ([CC0 1.0 license](https://creativecommons.org/publicdomain/zero/1.0/))
#
#  * https://freesound.org/people/SpliceSound/sounds/369848/ ([CC0 1.0 license](https://creativecommons.org/publicdomain/zero/1.0/))
#
#  * https://bigsoundbank.com/detail-0800-smoke-detector-alarm.html ([free of charge and royalty free.](https://bigsoundbank.com/droit.html))
#
#  * https://bigsoundbank.com/detail-1151-smoke-detector-alarm-2.html ([free of charge and royalty free.](https://bigsoundbank.com/droit.html))
#
#  * https://bigsoundbank.com/detail-1153-smoke-detector-alarm-3.html ([free of charge and royalty free.](https://bigsoundbank.com/droit.html))
#

# tf.keras.utils.get_file('fire_alarms.tar.gz',
#                         'https://github.com/ArmDeveloperEcosystem/ml-audio-classifier-example-for-pico/archive/refs/heads/fire_alarms.tar.gz',
#                         cache_dir='./',
#                         cache_subdir='datasets',
#                         extract=True)



# Since we only need the files in the _background_noise_ folder of the dataset
# use the curl command to download the archive file and then manually extract
# using the tar command, instead of using tf.keras.utils.get_file(...)
# in Python

# get_ipython().system('mkdir -p datasets/speech_commands')
# get_ipython().system('curl -L -o datasets/speech_commands_v0.02.tar.gz http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz')
# get_ipython().system("tar --wildcards --directory datasets/speech_commands -xzvf datasets/speech_commands_v0.02.tar.gz './_background_noise_/*'")


# ### Load dataset
#
# Instead of using a pandas DataFrame to load the dataset, we will load the fire alarm files and
# background noise files separately. The `label` and `fold` values will be mapped manually.


if not use_savedfinetunemodel:

    wavfiles = basedir +  "datasets/ml-audio-classifier-example-for-pico-fire_alarms/*.wav"
    fire_alarm_files_ds = tf.data.Dataset.list_files(wavfiles, shuffle=False)
    fire_alarm_files_ds = fire_alarm_files_ds.map(lambda x: (x, 1, -1))

    noisefiles = basedir + "datasets/speech_commands/_background_noise_/*.wav"
    background_noise_files_ds = tf.data.Dataset.list_files(noisefiles, shuffle=False)
    background_noise_files_ds = background_noise_files_ds.map(lambda x: (x, 0, -1))


    fire_alarm_wav_ds = fire_alarm_files_ds.map(load_wav_for_map)
    fire_alarm_wav_ds = fire_alarm_wav_ds.cache()

    background_noise_wav_ds = background_noise_files_ds.map(load_wav_for_map)
    background_noise_wav_ds = background_noise_wav_ds.cache()


    # Plot and listen to the first fire alarm file:

    if show_graphs:
        for wav_data, _, _ in fire_alarm_wav_ds.take(1):
            plt.plot(wav_data)
            plt.ylim([-1, 1])
            plt.title("fire alarm wav")
            plt.show()

            if listen_audio:
                play_audio_data(wav_data)

    # Do the same for the first background noise file:
    if show_graphs:
        for wav_data, _, _ in background_noise_wav_ds.take(1):
            plt.plot(wav_data)
            plt.ylim([-1, 1])
            plt.title("background noise wav")
            plt.show()

            if listen_audio:
                play_audio_data(wav_data)

    # Split the audio samples into 1 second soundbites:

    split_fire_alarm_wav_ds = fire_alarm_wav_ds.flat_map(split_wav_for_flat_map)
    split_fire_alarm_wav_ds = split_fire_alarm_wav_ds.filter(lambda x, y, z: wav_not_empty(x))

    split_background_noise_wav_ds = background_noise_wav_ds.flat_map(split_wav_for_flat_map)
    split_background_noise_wav_ds = split_background_noise_wav_ds.filter(lambda x, y, z: wav_not_empty(x))


    # TensorFlow Lite for Microcontroller (TFLu) provides only a subset of TensorFlow operations,
    # so we are unable to use the tf.signal.stft API we’ve used for feature extraction of the
    # baseline model on our MCU. However, we can leverage Arm’s CMSIS-DSP library to generate
    # spectrograms on the MCU. CMSIS-DSP contains support for both floating-point and fixed-point
    # DSP operations which are optimized for Arm Cortex-M processors, including the Arm Cortex-M0+
    # that we will be deploying the ML model to. The Arm Cortex-M0+ does not contain a floating-point
    # unit (FPU) so it would be better to leverage a 16-bit fixed-point DSP based feature
    # extraction pipeline on the board.
    #
    # We can leverage CMSIS-DSP’s Python Wrapper to perform the same operations on our
    # training pipeline using 16-bit fixed-point math. We will replicate the TensorFlow
    # short-time fourier transform API with the following CMSIS-DSP based operations:
    #
    #  1. Manually creating a Hanning Window of length 256 using the Hanning Window formula along with CMSIS-DSP’s `arm_cos_f32` API.
    #  2. Creating a CMSIS-DSP `arm_rfft_instance_q15` instance and initializing it using CMSIS-DSP’s `arm_rfft_init_q15` API.
    #  3. Looping through the audio data 256 samples at a time, with a stride of 128 (this matches the parameters we’ve passed into the TF sft API)
    #    a. Multiplying the 256 samples by the Hanning Window, using CMSIS-DSP’s `arm_mult_q15` API
    #    b. Calculating the FFT of the output of the previous step, using CMSIS-DSP’s `arm_rfft_q15` API
    #    c. Calculating the magnitude of the previous step, using CMSIS-DSP’s `arm_cmplx_mag_q15` API
    #  4. Each audio soundbites’s FFT magnitude represents the one column of the spectrogram.
    #  5. Since our baseline model expects a floating-point input, instead of the 16-bit quantized value we were using,
    #     the CMSIS-DSP `arm_q15_to_float` API can be used to convert the spectrogram data from a 16-bit fixed-point value
    #     to a floating-point value for training.
    #
    # For description of how to create audio spectrograms using fixed-point operations with CMSIS-DSP,
    # see the Towards Data Science “Fixed-point DSP for Data Scientists” guide at
    # https://towardsdatascience.com/fixed-point-dsp-for-data-scientists-d773a4271f7f.

    window_size = 256
    step_size = 128

    hanning_window_f32 = np.zeros(window_size)
    for i in range(window_size):
        hanning_window_f32[i] = 0.5 * (1 - cmsisdsp.arm_cos_f32(2 * PI * i / window_size ))

    hanning_window_q15 = cmsisdsp.arm_float_to_q15(hanning_window_f32)

    rfftq15 = cmsisdsp.arm_rfft_instance_q15()
    status = cmsisdsp.arm_rfft_init_q15(rfftq15, window_size, 0, 1)


    # Create a spectrogram representation for all of the fire alarm soundbites
    # plot the first spectrogram.
    fire_alarm_spectrograms_ds = split_fire_alarm_wav_ds.map(create_arm_spectrogram_for_map)
    fire_alarm_spectrograms_ds = fire_alarm_spectrograms_ds.cache()

    if show_graphs:
        for spectrogram, _, _ in fire_alarm_spectrograms_ds.take(1):
            plot_spectrogram(spectrogram, title="fire alarm")


    # Do the same for the background noise soundbites:
    background_noise_spectrograms_ds = split_background_noise_wav_ds.map(create_arm_spectrogram_for_map)
    background_noise_spectrograms_ds = background_noise_spectrograms_ds.cache()

    if show_graphs:
        for spectrogram, _, _ in background_noise_spectrograms_ds.take(1):
            plot_spectrogram(spectrogram, title="background noise")

    # Calculate the lengths of each dataset to see how balanced they are
    num_fire_alarm_spectrograms = calculate_ds_len(fire_alarm_spectrograms_ds)
    num_background_noise_spectrograms = calculate_ds_len(background_noise_spectrograms_ds)

    # num_fire_alarm_spectrograms = 1067
    # num_background_noise_spectrograms = 1572
    print(f"num_fire_alarm_spectrograms = {num_fire_alarm_spectrograms}")
    print(f"num_background_noise_spectrograms = {num_background_noise_spectrograms}")


    # Since there are more background noise samples than fire alarm samples
    # we will use data augmentation to balance them.

    # ### Data Augmentation

    # Data augmentation is a set of techniques used to increase the size of a dataset. This is
    # done by slightly modifying samples from the dataset or by creating synthetic data. In
    # this situation we are using audio and we will create a few functions to augment the
    # different samples. We will use three techniques:
    #
    #  * adding white noise to the audio samples
    #  * adding random silence to the audio
    #  * mixing two audio samples together
    #
    # As well as increasing the size of the dataset, data augmentation also helps to reduce
    # overfitting by training the model on different (not perfect) data samples. For example,
    # on a microcontroller you are unlikely to have perfect high quality audio, and so a
    # technique like adding white noise can help the model work in situations where the
    # microphone might intermittently be noisy.
    #
    # Plot the time representation of the first fire alarm soundbite over time along with
    # it's spectrogram representation so we can compare against the augmented versions.

    print("starting data augmentation...")

    for wav, _, _ in split_fire_alarm_wav_ds.take(1):
        test_fire_alarm_wav = wav

    if show_graphs:
        # plt.plot(test_fire_alarm_wav)
        # plt.ylim([-1, 1])
        # plt.title("fire alarm")
        # plt.show()

        plot_spectrogram(get_arm_spectrogram(test_fire_alarm_wav), vmax=25, title="fire alarm")

    if listen_audio:
        play_audio_data(test_fire_alarm_wav)

    # #### White Noise

    # TensorFlow tf.random.uniform API - see https://www.tensorflow.org/api_docs/python/tf/random/uniform
    # can be used generate a Tensor of equal shape to the original audio. This Tensor can then be multiplied
    # by a random scalar, and then added to the original audio samples.
    # The tf.clip_by_value API - see https://www.tensorflow.org/api_docs/python/tf/clip_by_value
    # will be used to ensure the audio remains in the range of -1.0 to 1.0.

    # Apply the white noise to the fire alarm sound and then plot it to compare.
    test_fire_alarm_with_white_noise_wav = add_white_noise(test_fire_alarm_wav)

    if show_graphs:
        # plt.plot(test_fire_alarm_with_white_noise_wav)
        # plt.ylim([-1, 1])
        # plt.title("fire alarm with white noise")
        # plt.show()

        plot_spectrogram(get_arm_spectrogram(test_fire_alarm_with_white_noise_wav), vmax=25, title="fire alarm with white noise")

    if listen_audio:
        play_audio_data(test_fire_alarm_with_white_noise_wav)


    # #### Random Silence
    #
    # TensorFlow's tf.random.categorical API - see https://www.tensorflow.org/api_docs/python/tf/random/categorical
    # can be used generate a Tensor of equal shape to the original audio containing mask of `True` or `False`. This
    # mask can then be casted to a float type of 1.0 or 0.0, so that it can be multiplied by the original audio single
    # to create random periods of silence.


    # Apply the random silence to the fire alarm sound and then plot it to compare.
    test_fire_alarm_with_random_silence_wav = add_random_silence(test_fire_alarm_wav)

    if show_graphs:
        # plt.plot(test_fire_alarm_with_random_silence_wav)
        # plt.ylim([-1, 1])
        # plt.title("fire alarm with random silence")
        # plt.show()

        plot_spectrogram(get_arm_spectrogram(test_fire_alarm_with_random_silence_wav), vmax=25, title="fire alarm with random silence")

        if listen_audio:
            play_audio_data(test_fire_alarm_with_random_silence_wav)


    # #### Audio Mixups
    #
    # Combine a fire alarm soundbite with a background noise soundbite to create a mixed up
    # version of the two.

    for wav, _, _ in split_background_noise_wav_ds.take(1):
        test_background_noise_wav = wav

    if show_graphs:
        # plt.plot(test_background_noise_wav)
        # plt.ylim([-1, 1])
        # plt.title("background noise to create audio mixup")
        # plt.show()

        plot_spectrogram(get_arm_spectrogram(test_background_noise_wav), title="background noise to create audio mixup")

        if listen_audio:
            play_audio_data(test_background_noise_wav)

    # Multiply the background noise soundbite with a random scalar before adding it
    # to the original fire alarm soundbite. Then ensure the mixed up value is between
    # the range of -1.0 and 1.0.

    # Apply the audio mixup to the fire alarm sound and then plot it to compare.
    test_fire_alarm_with_mixup_wav = add_audio_mixup(test_fire_alarm_wav, test_background_noise_wav)

    if show_graphs:
        # plt.plot(test_fire_alarm_with_mixup_wav)
        # plt.ylim([-1, 1])
        # plt.title("fire alarm with audio mixup")
        # plt.show()

        plot_spectrogram(get_arm_spectrogram(test_fire_alarm_with_mixup_wav), vmax=25, title="fire alarm with audio mixup")

        if listen_audio:
            play_audio_data(test_fire_alarm_with_mixup_wav)

    if show_graphs:
        fig, axs = plt.subplots(5, figsize=(16, 10))
        fig.suptitle('audio signals: augmented data')

        axs[0].plot(test_fire_alarm_wav)
        axs[0].set_ylim(-1, 1)
        axs[0].title.set_text('fire alarm')

        axs[1].plot(test_fire_alarm_with_white_noise_wav)
        axs[1].set_ylim(-1, 1)
        axs[1].title.set_text('fire alarm with white noise')

        axs[2].plot(test_fire_alarm_with_random_silence_wav)
        axs[2].set_ylim(-1, 1)
        axs[2].title.set_text('fire alarm with random silence')

        axs[3].plot(test_background_noise_wav)
        axs[3].set_ylim(-1, 1)
        axs[3].title.set_text('background noise')

        axs[4].plot(test_fire_alarm_with_mixup_wav)
        axs[4].set_ylim(-1, 1)
        axs[4].title.set_text('fire alarm with background noise')

        plt.tight_layout()
        plt.show()


    # ### Create the Augmented Dataset
    #
    # Combine all three augmententation techniques to balance our dataset.
    # Calculate how many augmented files we need to generate
    num_augmented_fire_alarm_spectrograms = num_background_noise_spectrograms - num_fire_alarm_spectrograms
    print(f'num_augmented_fire_alarm_spectrograms = {num_augmented_fire_alarm_spectrograms}')


    # Divide by 3 to calculate how many augmented soundbites per technique to generate:
    num_white_noise_fire_alarm_spectrograms = num_augmented_fire_alarm_spectrograms // 3
    num_random_silence_fire_alarm_spectrograms = num_augmented_fire_alarm_spectrograms // 3
    num_audio_mixup_fire_alarm_spectrograms = num_augmented_fire_alarm_spectrograms // 3

    print(f'num_white_noise_fire_alarm_spectrograms = {num_white_noise_fire_alarm_spectrograms}')
    print(f'num_random_silence_fire_alarm_spectrograms = {num_random_silence_fire_alarm_spectrograms}')
    print(f'num_audio_mixup_fire_alarm_spectrograms = {num_audio_mixup_fire_alarm_spectrograms}')

    # Select and shuffle the number of soundbites required
    split_fire_alarm_wav_ds = split_fire_alarm_wav_ds.cache()
    preaugmented_split_fire_alarm_wav = split_fire_alarm_wav_ds.shuffle(num_augmented_fire_alarm_spectrograms, seed=RANDOM_SEED).take(num_augmented_fire_alarm_spectrograms)

    # Create the white noise augmented soundbites
    def add_white_noise_for_map(wav, label, fold):
        return add_white_noise(wav), label, fold

    white_noise_fire_alarm_wav_ds = preaugmented_split_fire_alarm_wav.take(num_white_noise_fire_alarm_spectrograms)
    white_noise_fire_alarm_wav_ds = white_noise_fire_alarm_wav_ds.map(add_white_noise_for_map)

    # Create the random noise augmented soundbites
    def add_random_silence_for_map(wav, label, fold):
        return add_random_silence(wav), label, fold

    random_silence_fire_alarm_wav_ds = preaugmented_split_fire_alarm_wav.skip(num_white_noise_fire_alarm_spectrograms)
    random_silence_fire_alarm_wav_ds = random_silence_fire_alarm_wav_ds.take(num_random_silence_fire_alarm_spectrograms)
    random_silence_fire_alarm_wav_ds = random_silence_fire_alarm_wav_ds.map(add_random_silence_for_map)

    # Create the audio mixup augmented soundbites
    audio_mixup_background_noise_ds = split_background_noise_wav_ds.shuffle(num_audio_mixup_fire_alarm_spectrograms).take(num_audio_mixup_fire_alarm_spectrograms)
    audio_mixup_background_noise_iter = iter(audio_mixup_background_noise_ds.map(lambda x, y, z: x))

    def add_audio_mixup_for_map(wav, label, fold):
        return add_audio_mixup(wav, next(audio_mixup_background_noise_iter)), label, fold

    audio_mixup_split_fire_alarm_wav_ds = preaugmented_split_fire_alarm_wav.skip(num_white_noise_fire_alarm_spectrograms + num_random_silence_fire_alarm_spectrograms)
    audio_mixup_split_fire_alarm_wav_ds = audio_mixup_split_fire_alarm_wav_ds.take(num_audio_mixup_fire_alarm_spectrograms)
    audio_mixup_split_fire_alarm_wav_ds = audio_mixup_split_fire_alarm_wav_ds.map(add_audio_mixup_for_map)

    # Combine all the augmented soundbites together and map them to their spectrogram representations
    augment_split_fire_alarm_wav_ds = tf.data.Dataset.concatenate(white_noise_fire_alarm_wav_ds, random_silence_fire_alarm_wav_ds)
    augment_split_fire_alarm_wav_ds = tf.data.Dataset.concatenate(augment_split_fire_alarm_wav_ds, audio_mixup_split_fire_alarm_wav_ds)

    augment_fire_alarm_spectrograms_ds = augment_split_fire_alarm_wav_ds.map(create_arm_spectrogram_for_map)
    print("finished data augmentation.")



    # ### Split Dataset
    #
    # Combine the spectrogram datasets, and split them into training, validation, and test sets.
    # Instead of using the `fold` value to split them, we will shuffle all the items, and then
    # split by percentage.
    print("creating spectrogram train/validate/test datasets")

    # full_ds is 'tensorflow.python.data.ops.dataset_ops.ConcatenateDataset'
    full_ds = tf.data.Dataset.concatenate(fire_alarm_spectrograms_ds, background_noise_spectrograms_ds)
    full_ds = tf.data.Dataset.concatenate(full_ds, augment_fire_alarm_spectrograms_ds)
    full_ds = full_ds.cache()

    # save the full dataset - see tensorflow/python/data/experimental/ops/io.py
    print("saving full dataset...")

    tf.data.experimental.save(
        dataset=full_ds, path=basedir + "datasets/model-saved/full_ds", compression='GZIP'
    )
    with open(basedir +  "datasets/model-saved/full_ds" + '.pickle', 'wb') as out_:
        pickle.dump(full_ds.element_spec, out_)

    train_ds, val_ds, test_ds = split_full_dataset(full_ds, prefetch=False)

    # ### Replace Baseline Model Classification Head and Train Model
    #
    # The model we previously trained on the ESC-50 dataset, predicted the presence of 50
    # sound types, and which resulted in the final dense layer of the model having 50 outputs.
    # The new model we would like to create is a binary classifier, and needs to have a single
    # output value.
    #
    # We will load the baseline model, and swap out the final dense layer to match our needs:
    print("creating new tinyML model")

    # we need a new head with one neuron.
    model_body = tf.keras.Model(inputs=baseline_model.input, outputs=baseline_model.layers[-2].output)

    classifier_head = tf.keras.layers.Dense(1, activation="sigmoid")(model_body.output)

    fine_tune_model = tf.keras.Model(model_body.input, classifier_head)
    fine_tune_model.summary()


    # To freeze a layer in TensorFlow we set `layer.trainable = False`. Loop through
    # all the layers and do this:
    for layer in fine_tune_model.layers:
        layer.trainable = False

    # and now unfreeze the last layer (the head):
    fine_tune_model.layers[-1].trainable = True

    # Compile the model, this time with using a binary crossentropy loss function as this model
    # contains a single output.

    METRICS = [
        "accuracy",
    ]

    fine_tune_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=METRICS,
    )

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(verbose=1, patience=5),
        tf.keras.callbacks.LearningRateScheduler(scheduler)
    ]

    # Start the  training
    print("training tinyML model...")
    EPOCHS = 25

    history_1 = fine_tune_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Unfreeze all the layers, and train for a few more epochs:
    for layer in fine_tune_model.layers:
        layer.trainable = True

    fine_tune_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=METRICS,
    )

    def scheduler(epoch, lr):
        return lr * tf.math.exp(-0.1)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(verbose=1, patience=5),
        tf.keras.callbacks.LearningRateScheduler(scheduler)
    ]


    EPOCHS = 10
    history_2 = fine_tune_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    metrics = history_2.history
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(history_2.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylim([0, max(plt.ylim())])    
    plt.xlabel('Epoch')
    plt.ylabel('Finetune Model Loss [CrossEntropy]')
    
    plt.subplot(1,2,2)
    plt.plot(history_2.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
    plt.legend(['accuracy', 'val_accuracy'])
    plt.ylim([0, 100])    
    plt.xlabel('Epoch')
    plt.ylabel('Finetune Model Accuracy [%]')
    plt.show()

    print("training done, saving the TinyML (fine-tuned) model")
    fine_tune_model.save("fine_tuned_model")

else:
    # load the saved fine-tuned model
    print("loading saved fine-tuned model")
    fine_tune_model = tf.keras.models.load_model('fine_tuned_model')
    fine_tune_model.summary()




# ## Training with local audio (optional)

# We now have an ML model which can classify the presence of fire alarm sound. However
# this model was trained on publicly available sound recordings which might not match
# he sound characteristics of the hardware microphone we will use for inferencing.
#
# The Raspberry Pi RP2040 MCU has a native USB feature that allows it to act like a
# USB microphone. We can flash an application to the board to enable it to act like a
# USB microphone to our PC. Then we can extend Google Colab’s capabilities with the Web Audio
# API - see https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API - on a modern Web
# browser like Google Chrome to collect live data samples all from within Google Colab.

# ### Record your own audio - Software Setup
#
# Use the USB microphone example from the [Microphone Library for Pico - see
# https://github.com/ArmDeveloperEcosystem/microphone-library-for-pico.
# The example application can be compiled using `cmake` and `make`. Then we can flash
# the example application to the board over USB by putting the board into “boot ROM mode”
# which will allow us to upload an application to the board.
#
# Use `git` to clone the library source code and accompanying examples:
# get_ipython().run_cell_magic('shell', '', 'git clone https://github.com/ArmDeveloperEcosystem/microphone-library-for-pico.git\n')

# create a `build` folder to run `cmake` on:
# get_ipython().run_cell_magic('shell', '', 'cd microphone-library-for-pico\nmkdir -p build\ncd build\ncmake .. -DPICO_BOARD=${PICO_BOARD}\n')

# run `make` to compile the example:
# get_ipython().run_cell_magic('shell', '', 'cd microphone-library-for-pico/build\n\nmake -j usb_microphone\n')


# #### Flashing the board
#
# If you are using a [WebUSB API](https://wicg.github.io/webusb/) enabled browser like Google Chrome,
# you can directly flash the image onto the board from within Google Collab (Otherwise, you can manually
# download the .uf2 file to your computer and then drag it onto the USB disk for the RP2040 board.)
#
# or use picotool...
#
# **Note for Windows**: If you are using Windows you must install WinUSB drivers in order to use WebUSB,
# you can do so by following the instructions here:
# https://github.com/ArmDeveloperEcosystem/ml-audio-classifier-example-for-pico/blob/main/windows.md
#
# **Note for Linux**: If you are using Linux you must configure udev in order to use WebUSB, you can do
# so by following the instructions here: https://github.com/ArmDeveloperEcosystem/ml-audio-classifier-example-for-pico/blob/main/linux.md
#
# First you must place the board in USB Boot ROM mode, as follows:
#
#  * SparkFun MicroMod
#   * Plug the USB-C cable into the board and your PC to power the board
#   * While holding down the BOOT button on the board, tap the RESET button
#
#  * Raspberry Pi Pico
#   * Plug the USB Micro cable into your PC, but do NOT plug in the Pico side.
#   * While holding down the white BOOTSEL button, plug in the micro USB cable to the Pico


# Run the code cell below and then click the "Flash" button to upload the USB microphone
# example application to the board over USB.
#
# from colab_utils.pico import flash_pico
#
# flash_pico('microphone-library-for-pico/build/examples/usb_microphone/usb_microphone.bin')



if  trainwithlocalaudio:

    # Record local fire alarm sounds
    # from colab_utils.audio import record_wav_file
    # os.makedirs('datasets/custom/fire_alarm', exist_ok=True)
    # record_wav_file('datasets/custom/fire_alarm')

    # Record local background noise
    # os.makedirs('datasets/custom/background_noise', exist_ok=True)
    # record_wav_file('datasets/custom/background_noise')


    # We can zip up the recorded wave files to download and use again:
    # get_ipython().system('zip -r custom.zip datasets/custom')
    with zipfile.ZipFile('custom.zip', 'w', zipfile.ZIP_DEFLATED) as myzip:
        topdir = 'datasets/custom'
        for folder_name, subfolders, filenames in os.walk(topdir):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                myzip.write(file_path, arcname=os.path.relpath(file_path, topdir))


    # ### Load dataset
    # Load and transform the custom recorded dataset using the same pipeline as above
    custom_fire_alarm_ds = tf.data.Dataset.list_files("datasets/custom/fire_alarm/*.wav", shuffle=False)
    custom_fire_alarm_ds = custom_fire_alarm_ds.map(lambda x: (x, 1, -1))
    custom_fire_alarm_ds = custom_fire_alarm_ds.map(load_wav_for_map)
    custom_fire_alarm_ds = custom_fire_alarm_ds.flat_map(split_wav_for_flat_map)
    custom_fire_alarm_ds = custom_fire_alarm_ds.map(create_arm_spectrogram_for_map)

    custom_background_noise_ds = tf.data.Dataset.list_files("datasets/custom/background_noise/*.wav", shuffle=False)
    custom_background_noise_ds = custom_background_noise_ds.map(lambda x: (x, 0, -1))
    custom_background_noise_ds = custom_background_noise_ds.map(load_wav_for_map)
    custom_background_noise_ds = custom_background_noise_ds.flat_map(split_wav_for_flat_map)
    custom_background_noise_ds = custom_background_noise_ds.map(create_arm_spectrogram_for_map)

    custom_ds = tf.data.Dataset.concatenate(custom_fire_alarm_ds, custom_background_noise_ds)
    custom_ds = custom_ds.map(lambda x, y, z: (tf.expand_dims(x, axis=-1), y, z))
    custom_ds_len = calculate_ds_len(custom_ds)

    print(f'{custom_ds_len}')

    custom_ds = custom_ds.map(lambda x, y,z: (x, y))
    custom_ds = custom_ds.shuffle(custom_ds_len).cache()

    # Evaluate dataset performance before training:
    fine_tune_model.evaluate(custom_ds.batch(1))

    # ### Fine tune model
    EPOCHS = 25

    for layer in fine_tune_model.layers:
        layer.trainable = False

    fine_tune_model.layers[-1].trainable = True

    fine_tune_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=METRICS,
    )

    history3 = fine_tune_model.fit(
        custom_ds.take(int(custom_ds_len * 0.8)).batch(1),
        validation_data=custom_ds.skip(int(custom_ds_len * 0.8)).batch(1),
        epochs=EPOCHS,
        # callbacks=callbacks,
    )

    fine_tune_model.evaluate(custom_ds.batch(1))
    fine_tune_model.save('fine_tuned_model')



# ## Model Optimization
#
# Quantization brings improvements via model compression and latency reduction. With the API 
# defaults, the model size shrinks by 4x, and we typically see between 1.5 - 4x improvements 
# in CPU latency in the tested backends. Eventually, latency improvements can be seen on 
# compatible machine learning accelerators, such as the EdgeTPU and NNAPI.

print("starting model optimization [quantization] for Pico-W microcontroller...")

# To optimize the model to run on the Arm Cortex-M0+ processor, we will use a process
# called model quantization. Model quantization converts the model’s weights and biases
# from 32-bit floating-point values to 8-bit values. The  pico-tflmicro library - see
# https://github.com/raspberrypi/pico-tflmicro - which is a port of TFLu for the RP2040’s 
# Pico SDK contains Arm’s CMSIS-NN library, which supports optimized kernel operations for 
# quantized 8-bit weights on Arm Cortex-M processors.

# ### Quantization Aware Training
#
# There are two forms of quantization: post-training quantization and quantization aware training. 
# Post-training quantization is easier to use, though quantization aware training is often better for 
# model accuracy.
#
# Post-training quantization techniques can be performed on an already-trained float TensorFlow
# model and applied during TensorFlow Lite conversion.
#
# Quantization aware training emulates inference-time quantization, creating a model that downstream 
# tools will use to produce actually quantized models. The quantized models use lower-precision (e.g. 
# 8-bit instead of 32-bit float), leading to benefits during deployment.


# We will use TensorFlow’s Quantization Aware Training (QAT) feature - see 
# https://www.tensorflow.org/model_optimization/guide/quantization/training -
# which can  easily convert the floating-point model to quantized fixed-point.
final_model = tf.keras.models.load_model("fine_tuned_model")

# load the saved fine-tuned transfer-model datasets
# loaded datasets are tensorflow.python.data.ops.dataset_ops.PrefetchDataset
print("loading full dataset...")

with open(basedir + "datasets/model-saved/full_ds" + '.pickle', 'rb') as in_:
    es = pickle.load(in_)
full_ds = tf.data.experimental.load(
    basedir + "datasets/model-saved/full_ds", es, compression='GZIP'
)
train_ds, val_ds, test_ds = split_full_dataset(full_ds, prefetch=False)

def apply_qat_to_dense_and_cnn(layer):
    if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer

annotated_model = tf.keras.models.clone_model(
    fine_tune_model,
    clone_function=apply_qat_to_dense_and_cnn,
)

quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)


#   Layer (type)                       Output Shape              Param #
#=======================================================================
#   input_1 (InputLayer)              [(None, 124, 129, 1)]        0
#
#   resizing (Resizing)               (None, 32, 32, 1)            0
#
#   normalization (Normalization)     (None, 32, 32, 1)            3
#
#   quant_conv2d (QuantizeWrapperV2)  (None, 13, 13, 8)          539
#
#   max_pooling2d (MaxPooling2D)      (None, 6, 6, 8)              0
#
#   flatten (Flatten)                 (None, 288)                  0
#
#   quant_dropout (QuantizeWrapperV2) (None, 288)                  1
#
#   quant_dense (QuantizeWrapperV2)   (None, 1)                  294
#
#=======================================================================
#   Total params: 837
#   Trainable params: 809
#   Non-trainable params: 28
quant_aware_model.summary()

METRICS = [
    "accuracy",
]

quant_aware_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=METRICS,
)

EPOCHS=1
quant_aware_history = quant_aware_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

metrics = quant_aware_history.history

# quantized model performance:
#  'accuracy':     [0.9734748005867004],
#  'loss':         [0.0793059915304184],
#  'val_accuracy': [0.9761146306991577],
#  'val_loss':     [0.0819682851433754]
print("quantized model performance:")
pprint.pprint(metrics)

# ### Saving model in TFLite format

# Use tf.lite.TFLiteConverter.from_keras_model API - see
# https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#from_keras_model -
# to convert the quantized Keras model to TF Lite format and save it to disk as a .tflite file.

converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    for input_value, output_value in train_ds.unbatch().batch(1).take(100):
        # Model has only one input so each data point has one element.
        yield [input_value]

converter.representative_dataset = representative_data_gen

# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# ??? what are these warnings from the quantizer ???

# WARNING:absl:Found untraced functions such as conv2d_layer_call_fn, conv2d_layer_call_and_return_conditional_losses, dropout_layer_call_fn, dropout_layer_call_and_return_conditional_losses, dense_layer_call_fn 
# while saving (showing 5 of 6). These functions will not be directly callable after loading.
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


# /files/pico/ML/audio-arm/venv310/lib/python3.10/site-packages/tensorflow/lite/python/convert.py:746: 
#  UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
# fully_quantize: 0, inference_type: 6, input_inference_type: 9, output_inference_type: 9
# non issue?  see https://github.com/tensorflow/tensorflow/issues/60574

tflite_model_quant = converter.convert()

with open("tflite_model.tflite", "wb") as f:
    f.write(tflite_model_quant)


# ### Test TF Lite model
#
# Since TensorFlow also supports loading TF Lite models using tensorflow.lite API -
# see https://www.tensorflow.org/api_docs/python/tf/lite, we can also verify the
# functionality of the quantized model and compare its accuracy with the regular
# unquantized model.

# Load the interpreter and allocate tensors.  See tensorflow/lite/python/interpreter.py
interpreter = tflite.Interpreter("tflite_model.tflite")
interpreter.allocate_tensors()

# Load input and output details
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
print("quantized model: input details")
pprint.pprint(input_details)
print("quantized model: output details")
pprint.pprint(output_details)

# Set quantization values
input_scale, input_zero_point = input_details["quantization"]
output_scale, output_zero_point = output_details["quantization"]

# Calculate the number of correct predictions
correct = 0
test_ds_len = 0

# Loop through entire test set
for x, y in test_ds.unbatch():
    # original shape is [124, 129, 1] expand to [1, 124, 129, 1]
    x = tf.expand_dims(x, 0).numpy()

    # quantize the input value
    if (input_scale, input_zero_point) != (0, 0):
        x = x / input_scale + input_zero_point
    x = x.astype(input_details['dtype'])

    # add the input tensor to interpreter
    interpreter.set_tensor(input_details["index"], x)

    # run the model
    interpreter.invoke()

    # Get output data from model and convert to fp32
    output_data = interpreter.get_tensor(output_details["index"])
    output_data = output_data.astype(np.float32)

    # Dequantize the output
    if (output_scale, output_zero_point) != (0.0, 0):
        output_data = (output_data - output_zero_point) * output_scale

    # convert output to category
    if output_data[0][0] >= 0.5:
        category = 1
    else:
        category = 0

    # add 1 if category = y
    correct += 1 if category == y.numpy() else 0

    test_ds_len += 1


accuracy = correct / test_ds_len

# Accuracy for quantized model is 96.19% on test set.
print(f"Accuracy for quantized model is {accuracy*100:.2f}% on test set.")


# ## Deploy on Device

# #### Convert `.tflite` to `.h` file
#
# The RP2040 MCU on the boards we are deploying to, does not have a built-in file
# system, which means we cannot use the .tflite file directly on the board. However,
# we can use the Linux `xxd` command to convert the .tflite file to a .h file which
# can then be compiled in the inference application in the next step.

# get_ipython().run_cell_magic('shell', '', 'echo "alignas(8) const unsigned char tflite_model[] = {" > tflite_model.h\ncat tflite_model.tflite | xxd -i                        >> tflite_model.h\necho "};"                                               >> tflite_model.h\n')
print('---> convert tflite model file to a C-language header file using xxd')
print('--->    execute: echo "alignas(8) const unsigned char tflite_model[] = {" > /tmp/tflite_model.h')
print('--->    execute: cat tflite_model.tflite | xxd -i >> /tmp/tflite_model.h')
print('--->    execute: echo "};" >> /tmp/tflite_model.h')
print('---> see /files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/src/tflite_model.h')




# #### Inference Application
#
# We now have a model that is ready to be deployed to the device! We’ve created an application
# template for inference which can be compiled with the .h file that we’ve generated for the model.
#
# The C++ application uses the `pico-sdk` as the base, along with the `CMSIS-DSP`, `pico-tflmicro`,
# and `Microphone Libary for Pico` libraries. It’s general structure is as follows:
#
#  1. Initialization
#
#   a. Configure the board's built-in LED for output. The application will map the brightness of the LED to the output of the model. (0.0 LED off, 1.0 LED on with full brightness)
#   b. Setup the TF Lite library and TF Lite model for inference
#   c. Setup the CMSIS-DSP based DSP pipeline
#   d. Setup and start the microphone for real-time audio
#
#  2. Inference loop
#
#   a. Wait for 128 * 4 = 512 new audio samples from the microphone
#   b. Shift the spectrogram array over by 4 columns
#   c. Shift the audio input buffer over by 128 * 4 = 512 samples and copy in the new samples
#   d. Calculate 4 new spectrogram columns for the updated input buffer
#   e. Perform inference on the spectrogram data
#   f. Map the inference output value to the on-board LED’s brightness and output the status to the USB port
#
# In-order to run in real-time each cycle of the inference loop must take under
# (512 / 16000) = 0.032 seconds or 32 milliseconds. The model we’ve trained and
# converted takes 24 ms for inference, which gives us ~8 ms for the other operations
# in the loop.
#
# 128 was used above to match the stride of 128 used in the training pipeline for the
# spectrogram. We used a shift of 4 in the spectrogram to fit within the real-time
# constraints we had.
#
# The source code for the inference application can be found on GitHub:
# https://github.com/ArmDeveloperEcosystem/ml-audio-classifier-example-for-pico/tree/main/inference-app
#
# Note this project is already cloned so these files are local.


# Copy the updated `tflite_model.h` file over:
# get_ipython().system('cp tflite_model.h inference-app/src/tflite_model.h')
destfile =  '/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/src/tflite_model.h'
print('---> copy C-language header created above for building Pico application')
print('--->    execute: cp  /tmp/tflite_model.h  {}'.format(destfile))



# #### Compile Inference Application

# use `cmake` to setup project before compiling it, then make to build
# get_ipython().run_cell_magic('shell', '', 'cd inference-app\nmkdir -p build\ncd build\ncmake .. -DPICO_BOARD=${PICO_BOARD}\n')
# get_ipython().run_cell_magic('shell', '', 'cd inference-app/build\n\nmake -j\n')

# 1. edit micro_utils.h:           see https://github.com/onnx/onnx-tensorrt/issues/474
#   $:/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app$ more  ./lib/pico-tflmicro/src/tensorflow/lite/micro/micro_utils.h
#   ...
#   #ifndef TENSORFLOW_LITE_MICRO_MICRO_UTILS_H_
#   #define TENSORFLOW_LITE_MICRO_MICRO_UTILS_H_

#   #include <algorithm>
#   #include <cmath>
#   #include <cstdint>
#  +#include <stdexcept>
#  +#include <limits>


# 2. edit src/main.cpp
#  (venv310) mike@debian-x250:/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app$ diff src/main.cpp src/main.cpp.ORIG
#  71,74d70
#  < #ifndef PICO_DEFAULT_LED_PIN
#  < #warning pio/hello_pio example requires a board with a regular LED
#  <
#  < #else
#  85d80
#  < #endif
#  148,150d142
#  < #ifndef PICO_DEFAULT_LED_PIN
#  < #warning pio/hello_pio example requires a board with a regular LED
#  < #else
#  152d143
#  < #endif


# 3. cmake
# mike@debian-x250:/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build$ cmake .. -DPICO_BOARD=pico_w -DCMAKE_BUILD_TYPE=Debug
# ...

# 4. make
# (venv310) mike@debian-x250:/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build$ make
# ...




# #### Flash inferencing application to board
#
# put the board into USB boot ROM mode again to load the new application to it. If you are using a
# WebUSB API enabled browser like Google Chrome, you can directly flash the image onto the board from
# within Google Collab! Otherwise, you can manually download the .uf2 file to your computer and
# then drag it onto the USB disk for the RP2040 board.
#
#  * SparkFun MicroMod
#   * Plug the USB-C cable into the board and your PC to power the board
#   * While holding down the BOOT button on the board, tap the RESET button
#
#  * Raspberry Pi Pico
#   * Plug the USB Micro cable into your PC, but do NOT plug in the Pico side.
#   * While holding down the white BOOTSEL button, plug in the micro USB cable to the Pico
print("Use picotool to flash the inference application to Pico-W and analyze audio...")
print(" ---->   sudo picotool load -x -t uf2 /files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico//inference-app/build/pico_inference_app.uf2")
print(" ---->   monitor via:   minicom -b 9600 -D /dev/ttyACM0")


# Then run the code cell below and click the "flash" button.
# from colab_utils.pico import flash_pico
# flash_pico('inference-app/build/pico_inference_app.bin')


# ### Monitoring the Inference on the board
#
# Now that the inference application is running on the board you can observe it in action in two ways:
#
#  1. Visually by observing the brightness of the LED on the board. It should remain off or dim
#     when no fire alarm sound is present - and be on when a fire alarm sound is present.
#
#  2. Connecting to the board’s USB serial port to view output from the inference application.
#
# #### Test Audio
#
# Use the code cell below to playback the fire alarms sounds used during training from your computer.
# You may need to adjust the speaker volume from your computer.

if  listen_audio:
    for wav, _, _ in fire_alarm_wav_ds:
        play_audio_data(wav)


# #### Serial Monitor
#
# Run the code cell below and then click the "Connect Port" button to view the serial output from the board:
#
# from colab_utils.serial_monitor import run_serial_monitor
#
# run_serial_monitor()


# ## Improving the model
#
# You now have the first version of the model deployed to the board, and it is performing inference
# on live 16,000 kHz audio data!
#
# Test out various sounds to see if the model has the expected output. Maybe the fire alarm sound
# is being falsely detected (false positive) or not detected when it should be (false negative).
#
# If this occurs, you can record more new audio data for the scenario(s) by flashing the USB microphone
# application firmware to the board, recording the data for training, re-training the model and
# converting to TF lite format, and re-compiling + flashing the inference application to the board.
#
# Supervised machine learning models can generally only be as good as the training data they are
# trained with, so additional training data for these scenarios might help. You can also try to
# experiment with changing the model architecture or feature extraction process - but keep in mind
# that your model must be small enough and fast enough to run on the RP2040 MCU!
#
# ## Conclusion
#
# This guide covered an end-to-end flow of how to train a custom audio classifier model to run
# locally on a development board that uses an Arm Cortex-M0+ processor. Google Colab was used to train
# the model using Transfer Learning techniques along with a smaller dataset and data augmentation
# techniques. We also collected our own data from the microphone that is used at inference time by
# loading an USB microphone application to the board, and extending Colab’s features with the Web
# Audio API and custom JavaScript
#
# The training side of the project combined Google’s Colab service and Chrome browser, with the open
# source TensorFlow library. The inference application captured audio data from a digital microphone,
# used Arm’s CMSIS-DSP library for the feature extraction stage, then used TensorFlow Lite for
# Microcontrollers with Arm CMSIS-NN accelerated kernels to perform inference with a 8-bit quantized
# model that classified a real-time 16 kHz audio input on an Arm Cortex-M0+ processor.
#
# The Web Audio API, Web USB API, and Web Serial API features of Google Chrome were used to extend
# Google Colab’s functionality to interact with the development board. This allowed us to experiment
# with and develop our application entirely with a web browser and deploy it to a constrained development
# board for on-device inference.
#
# Since the ML processing was performed on the development boards RP2040 MCU, privacy was preserved
# as no raw audio data left the device at inference time.

