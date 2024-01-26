# PROJET_SAM

This project is part of the SAM (Signal and Multimedia Learning) course in the Master 2 IAAA (Artificial Intelligence & Automatic Learning) at Aix-Marseille University.


## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Notebooks Directory](#notebooks-directory)
  - [Src Directory](#src-directory)
- [License](#license)
- [Acknowledgments](#acknowledgments)


## Project Overview

This project aims to apply the concepts learned in the SAM (Signal and Multimedia Learning) course. It involves the creation multimodal models such as early fusion and late fusion using videos, audio and text data. 

The goal is to compare the different ways to process multimodality (mainly Early VS Late fusion) and to familiarize ourselves with audio, text, and videos processing in deep learning models.

## Getting Started

### Prerequisites

- numpy==1.23.5 or more.
- pandas==1.5.2 or more.
- scikit-learn==1.2.0 or more.
- matplotlib==3.6.2 or more.
- torch==1.13.1 or more.
- torchaudio==0.13.1+cu116 or more.
- keras==2.11.0 or more.

### Installation

1. Clone the repository:

git clone https://github.com/Paulpey13/PROJET_SAM

2. Navigate to the project directory:

cd PROJET_SAM

3. Install the required packages:

pip install -r requirements.txt

## Usage

The project is structured into two main directories: `notebooks` and `src`. Each directory serves a specific purpose with various files facilitating the project's implementation. You need to import paco-cheese folder in order to use this project.
Also you can look at our experience results according to model and seed.

### Notebooks Directory

The `notebooks` directory contains Jupyter notebooks, each designed to use a specific model of the project:


Those notebook run a pipeline for every models : 

1. `audio_model.ipynb`: Implementation of the audio model.
2. `Early_fusion_model.ipynb`: Implementation of Early fusion with text and audio.
3. `late_fusion_model.ipynb`: Implementation of late fusion with text and audio.
4. `text_model.ipynb`: Implementation of the text model.

This notebook implement video mode : 
5. `video_model.ipynb`: Implementation of the video model.

6. `experiences.ipynb`: this run all of experiences.
### Src directory

The `src` directory contains various subdirectories, each dedicated to a specific component of the project:

- **Audio Directory**:
  - `audio_dataset.py`: Handles the audio dataset.
  - `audio_extract.py`: Script for audio data extraction.
  - `audio_model.py`: Contains the audio model.
  - `audio_training.py`: Script for training the audio model.

- **Early Directory**:
  - `early_dataset.py`: Manages the dataset for early fusion techniques.
  - `early_model.py`: Contains the early fusion model.
  - `early_training.py`: Script for training the early fusion model.

- **Late Directory**:
  - `combine_model.py`: Script to combine models for late fusion.
  - `late_audio_text_training.py`: Script for training the late fusion model with audio and text data.

- **Text Directory**:
  - `text_dataset.py`: Handles the text dataset.
  - `text_extract.py`: Script for text data extraction.
  - `text_model.py`: Contains the text model.
  - `text_training.py`: Script for training the text model.

- **Video Directory**:
  - `video_extract.py`: Script for video data extraction.
  - `video_model.py`: Contains the video model.
  - `video_training.py`: Script for training the video model.

Additional scripts in the `src` directory include:

- `load_data.py`: Script for loading datasets.
- `pipeline_audio_model.py`: Pipeline for processing and training the audio model.
- `pipeline_early_fusion.py`: Pipeline for processing and training the early fusion model.
- `pipeline_late_fusion.py`: Pipeline for processing and training the late fusion model.
- `pipeline_text_model.py`: Pipeline for processing and training the text model.

To work with these scripts, navigate to the `src` directory and execute the desired scripts with Python.

## Contributors

- Victor Tancrez
- Paul Peyssard
- Tony Nogneng
- Supervised by Eliot Maes & Leonor Beccera
- Aix-Marseille University

## License

This project is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
