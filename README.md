# California Housing Price Prediction (First ML Project)

This repository contains my first end-to-end Machine Learning project. The goal of this project is to build a complete ML pipeline to predict housing prices using the California Housing dataset.

The project demonstrates the full ML workflow including data preprocessing, feature engineering, pipeline construction, model training, hyperparameter tuning, and final evaluation.

---

## Project Overview

The model predicts median house prices based on various housing and geographical features such as income levels, population density, and location.

Key steps implemented in this project:

- Data preprocessing pipelines using Scikit-Learn
- Feature engineering with custom transformers
- Handling numerical and categorical features
- Model training using Random Forest
- Hyperparameter optimization using RandomizedSearchCV
- Final model evaluation on unseen test data
- RMSE confidence interval estimation using bootstrapping

---

## Machine Learning Pipeline

The project follows a complete ML workflow:
Raw Data
↓
Train/Test Split
↓
Feature Engineering
↓
Preprocessing Pipeline
↓
Random Forest Model
↓
Hyperparameter Tuning
↓
Final Evaluation


---

## Final Model

Algorithm used:

Random Forest Regressor

Performance on test set:

RMSE ≈ **39,547**

This means the model predicts housing prices with an average error of approximately **$39k**.

---

## Feature Engineering

Some important engineered features include:

- Rooms per household
- Bedrooms ratio
- Population per household
- Log-transformed numerical features
- Geographic cluster similarity features

These features help the model capture important relationships in the data.

---

## Technologies Used

Python  
Scikit-Learn  
Pandas  
NumPy  
SciPy  

---


---

## What I Learned

Through this project I learned how to:

- Build end-to-end ML pipelines
- Use Scikit-Learn pipelines and transformers
- Perform hyperparameter tuning
- Evaluate models properly using cross validation
- Estimate statistical confidence intervals for model performance

---

## Future Improvements

Possible improvements to this project include:

- Trying Gradient Boosting models
- Feature importance visualization
- Model deployment as an API
- Building a small web interface for predictions
