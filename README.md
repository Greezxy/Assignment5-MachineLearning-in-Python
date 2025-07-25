# Assignment 5 – Machine Learning in Python

This repository contains a Jupyter notebook that walks through a complete machine learning workflow using the Iris dataset and the `scikit-learn` library. The project is adapted from Jason Brownlee’s step-by-step tutorial and is designed to provide foundational experience in classification modeling, data exploration, and algorithm evaluation in Python.

---

## Table of Contents

- [Project Description] 
- [Installation Instructions] 
- [Usage]  
- [Project Structure]  
- [License]  
- [Acknowledgments]  

---

## Project Description

This notebook demonstrates the process of building a basic classification model using Python's `scikit-learn`. It includes:
- Data loading and inspection
- Exploratory visualization
- Evaluation of multiple classification algorithms
- Final model selection and prediction

The target audience is beginners in machine learning, data science students, and anyone looking for a hands-on introduction to supervised classification using real data.

---

## Installation Instructions

### Clone the Repository

```bash
git clone https://github.com/Greezxy/Assignment5-MachineLearning-in-Python.git
cd Assignment5-MachineLearning-in-Python
```

### Create and Activate the Conda Environment
```bash
conda env create -f environment-assignment5-python.yml
conda activate iris-ml-python
```  
This will install Python 3.10 along with essential ML and data libraries: pandas, numpy, scikit-learn, matplotlib, scipy, and jupyterlab.

## Usage
To launch the notebook:
```bash
jupyter lab
```

Then open the file:
Machine Learning in Python.ipynb

The notebook will:
- Load the Iris dataset
- Perform statistical summaries and visualizations
- Evaluate six ML algorithms (e.g., SVM, KNN, Logistic Regression)
- Output accuracy scores and visual comparisons
- Make predictions on a validation set

All results are fully reproducible with the provided environment and dataset.

## Project Structure
Assignment5-MachineLearning-in-Python/
├── Machine Learning in Python.ipynb     # Jupyter notebook with full ML workflow
├── environment-assignment5-python.yml           # Conda environment specification
├── README.md                 # Project overview and instructions

## License
This project is for educational use only and is part of a course assignment.

## Acknowledgments
Based on the tutorial by Jason Brownlee:  
Machine Learning in Python Step-by-Step  
© Jason Brownlee, MachineLearningMastery.com

---
