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

This project aims to apply the concepts learned in the SAM (Signal and Multimedia Learning) course. It involves the multimodal analysis and modeling of a dataset using various machine learning techniques. 

Multimodality in this context refers to the use of multiple types of data (such as text, audio, and video) in our analysis. By leveraging the strengths of each data type, we aim to build more robust and accurate models.

The project is implemented in Python, using libraries such as pandas, numpy, scikit-learn, and matplotlib for data manipulation, analysis, and visualization. We also use libraries like PyTorch for building and training our machine learning models.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Jupyter Notebook

### Installation

1. Clone the repository:

git clone https://github.com/Paulpey13/PROJET_SAM

2. Navigate to the project directory:

cd PROJET_SAM

3. Install the required packages:

pip install -r requirements.txt

## Usage

The project is structured into two main directories: `notebooks` and `src`. Each directory serves a specific purpose with various files facilitating the project's implementation.

### Notebooks Directory

The `notebooks` directory contains Jupyter notebooks, each designed for a specific part of the project:

1. `audioOnly.ipynb`: Implementation of the audio model.
2. `Early_fusion_video.ipynb`: Implementation of Early fusion with video (work in progress).
3. `Early_fusion.ipynb`: Implementation of Early fusion with text and audio.
4. `late_fusion.ipynb`: Implementation of late fusion with text and audio.
5. `textOnly.ipynb`: Implementation of the text model.
6. `timOnly.ipynb`: Implementation of the text model based on time pauses.
7. `Video_only.ipynb`: Implementation of the video model.

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
- `pipeline_late_fusion_audio_text.py`: Pipeline for processing and training the late fusion model with audio and text.
- `pipeline_late_fusion.py`: Pipeline for processing and training the late fusion model.
- `pipeline_text_model.py`: Pipeline for processing and training the text model.
- `utils.py`: Contains utility functions used across various scripts.

To work with these scripts, navigate to the `src` directory and execute the desired scripts with Python.

## License

This project is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License

## Acknowledgments

We would like to thank the professors and teaching assistants of the SAM course for their guidance and support throughout this project.
