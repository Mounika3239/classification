# Logistic Regression and Support Vector Machine Implementation

This project contains an implementation of Logistic Regression and Support Vector Machines (SVM) for multi-class classification on the MNIST dataset. The project demonstrates data preprocessing, model training, evaluation, and visualization of results.

---

## Project Overview

### Implementation Details:
- **Logistic Regression**:
  - Binary and Multi-class Logistic Regression implemented using gradient descent.
  - Performance evaluated across training, validation, and test datasets.
  - Visualization of errors for different categories.

- **Support Vector Machines (SVM)**:
  - Implemented with linear and RBF kernels.
  - Performance comparison for varying `C` parameter values.
  - Graphical representation of the impact of `C` on training, validation, and testing accuracy.

---

- **`script.py`**: Python script containing:
  - Functions for data preprocessing, training, and prediction.
  - Logistic Regression implementation.
  - Multi-class Logistic Regression implementation.
  - Support Vector Machines implementation with kernel and parameter tuning.

- **`report.pdf`**: Report detailing the implementation, analysis, and comparison of Logistic Regression and SVM methods.

---

## Prerequisites

- Python 3.x
- Required libraries:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `sklearn`

Install dependencies using:
```bash
pip install numpy scipy matplotlib scikit-learn
