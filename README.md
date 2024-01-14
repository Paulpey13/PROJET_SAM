# PROJET_SAM

This project is part of the SAM (Signal and Multimedia Learning) course in the Master 2 IAAA (Artificial Intelligence & Automatic Learning) at Aix-Marseille University.

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

This project contains several Jupyter Notebook (`*.ipynb`) files, each of which serves a different purpose:

1. `timeOnly.ipynb`: This notebook focuses on the analysis and modeling of time-based features in the dataset.

2. `textOnly.ipynb`: This notebook is dedicated to the analysis and modeling of text-based features, based on natural language processing techniques.

3. `audioOnly.ipynb`: This notebook deals with audio-based features, based on signal processing and audio analysis techniques.

4. `late_fusion.ipynb`: This notebook demonstrates a late fusion approach, where predictions from multiple models (possibly trained on different types of features) are combined.

5. `Early_fusion2.ipynb`: This notebook demonstrates an early fusion approach, where different types of features are combined before training a model.

To run a notebook, navigate to the project directory in your terminal and start Jupyter Notebook with the following command:

```bash
jupyter notebook

## License

This project is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License

## Acknowledgments

We would like to thank the professors and teaching assistants of the SAM course for their guidance and support throughout this project.
