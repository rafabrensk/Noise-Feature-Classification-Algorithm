# Noise Feature Classification Algorithm

An experimental Machine Learning project that investigates whether removing low-importance features from a trained model can improve its predictive performance.

## Overview

Machine Learning models can be trained using a large number of features, but not every feature contributes equally to the final prediction. Some features may have little influence on the model, while others may even introduce noise or unnecessary complexity.

This project explores an iterative feature selection approach based on **feature importance**.

The general idea is:

1. Train a Machine Learning model using the complete dataset.
2. Extract the feature importance values from the trained model.
3. Identify the least important features.
4. Remove those features from the dataset.
5. Train the model again using the reduced feature set.
6. Compare the new accuracy with the original model.
7. Repeat the process to investigate how model performance changes as features are removed.

The objective is to determine whether eliminating low-importance features can produce a model with **equal or higher accuracy while using fewer features**.

## Methodology

The project follows a feature elimination loop:

```text
Dataset
   │
   ▼
Train Model
   │
   ▼
Calculate Feature Importance
   │
   ▼
Identify Low-Importance Features
   │
   ▼
Remove Features
   │
   ▼
Train Model Again
   │
   ▼
Compare Accuracy
   │
   └──────► Repeat
```

At each iteration, the model's performance is evaluated after reducing the number of features.

## Goals

* Investigate the relationship between feature importance and model performance.
* Identify potentially irrelevant or noisy features.
* Determine whether removing low-importance features can improve accuracy.
* Reduce the dimensionality of the dataset while maintaining predictive performance.
* Compare model performance before and after feature selection.
* Experiment with different feature-selection strategies.

## Example

Suppose the original dataset contains 10 features:

```text
Feature 1  → 0.31
Feature 2  → 0.24
Feature 3  → 0.18
Feature 4  → 0.10
Feature 5  → 0.07
Feature 6  → 0.04
Feature 7  → 0.03
Feature 8  → 0.02
Feature 9  → 0.01
Feature 10 → 0.00
```

The algorithm could remove the least important features and retrain the model:

```text
10 features → Accuracy: 91.2%
        ↓
8 features  → Accuracy: 92.0%
        ↓
6 features  → Accuracy: 92.8%
        ↓
4 features  → Accuracy: 89.5%
```

In this hypothetical example, the 6-feature model would be particularly interesting because it achieves higher accuracy while using fewer features.

## Technologies

* Python
* scikit-learn
* pandas
* NumPy
* Matplotlib

## Project Status

**In development**

The current implementation is focused on experimenting with feature importance and evaluating different feature-removal strategies.

Future versions may include:

* Support for different Machine Learning models.
* Multiple feature-importance methods.
* Configurable feature-removal thresholds.
* Automated comparison of different feature subsets.
* Additional evaluation metrics.
* Visualization of performance as features are removed.

## Motivation

Feature selection is an important part of Machine Learning because a model does not necessarily benefit from having access to every available feature.

This project is an attempt to explore this problem experimentally:

> **Can a model become more accurate by knowing less?**

Rather than assuming that low-importance features should be removed, the project evaluates their removal empirically by retraining the model and comparing its performance.
