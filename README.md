# 11-785 Introduction to Deep Learning (CMU)

This repository contains course materials and student solutions for Carnegie Mellon University's 11-785 Introduction to Deep Learning. It includes assignment starter code, autograders, reference data, and working notebooks for common projects used in the class.

**Contents**
- `mlp-from-scratch/`: Homework 1 — implement a multilayer perceptron (MLP) from scratch. Includes autograder, reference weights, and helper scripts.
- `cnn-from-scratch/`: Homework 2 — implement convolutional neural network building blocks from first principles. Contains autograder, reference results, and example scripts.
- `face_verification_and_identification.ipynb`: Notebook for face verification/identification assignment — dataset loading, model training, evaluation, and submission instructions.
- `frame_level_speech_recognition.ipynb`: Notebook for frame-level speech recognition assignment — feature extraction, model training, and evaluation.

**Quick start**
1. Create and activate a Python 3.8+ virtual environment.
2. Install requirements for each assignment before running its code. For example:

```bash
python -m venv venv
source venv/bin/activate
pip install -r mlp-from-scratch/requirements.txt
pip install -r cnn-from-scratch/requirements.txt
```

3. Open the notebooks in JupyterLab/Notebook to run the assignments, or run the provided scripts in the assignment folders.

**Notes**
- Each assignment folder contains an `autograder` subfolder with unit tests and helper utilities used for grading.
- Large binary/reference files required by autograders are included in the respective folders.

If you'd like, I can expand each section with more detailed instructions, example commands, and expected outputs for each assignment.
