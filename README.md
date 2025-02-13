# Animal Classification

## Group Members

Dimitri Eilinger \\
Timon Wieland \\
Yassin Joubli \\

## Data 

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/caoofficial/animal-sounds?resource=download).

## The Project

This project's model performs a classification task. It classifies an animal sound into one of 10 animal sounds: 'Bird', 'Cat', 'Chicken', 'Cow', 'Dog', 'Donkey', 'Frog', 'Lion', 'Monkey', and 'Sheep'. 

To enable the model to perform transforms and convolutions, the audio files are transformed into image form, specifically spectrograms. A spectrogram is a 2D visual representation of sound, where:
- The x-axis represents time.
- The y-axis represents pitch.
- The brightness of the lines represents volume; the brighter the line, the louder the sound.

## Project Setup

### Requirements

Make sure to create and activate the environment before running the project:

```bash
conda env create -f environment.yml
conda activate eth-pythonML24
