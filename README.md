# mercedes-price-prediction-dl
Mercedes car price prediction using Deep Learning regression models built with TensorFlow &amp; Keras. Includes data analysis, preprocessing, model training, and price inference using real-world vehicle dataset.

Mercedes Price Prediction (Deep Learning)

Project Overview
This project focuses on predicting Mercedes car prices using Deep Learning regression models built with TensorFlow and Keras. The model is trained on real-world automotive market data and demonstrates a complete Machine Learning pipeline from data preprocessing to model prediction.
This repository is designed as an end-to-end Deep Learning project for tabular data price prediction.

## Objectives

Predict vehicle prices using structured car listing data

Apply Deep Learning to tabular regression problems

Build a full ML workflow including preprocessing, training, and inference

## Dataset Information
Dataset includes real vehicle listing attributes:

year → Vehicle production year

transmission → Transmission type

mileage → Total mileage

tax → Vehicle tax

mpg → Fuel consumption

engineSize → Engine size

price → Vehicle price (Target Variable)

## Model Architecture
Deep Neural Network (DNN)

Sequential Model

Hidden Layers: Dense + ReLU Activation

Output Layer: 1 Neuron (Regression Output)

Optimizer: Adam

Loss Function: Mean Squared Error (MSE)

## Tech Stack

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-Learn

TensorFlow / Keras

Project Structure

mercedes-price-prediction-dl/
data/ → merc.xlsx
notebooks/ → main.ipynb
src/ → (optional future production scripts)
requirements.txt
README.md
.gitignore

Installation

Clone Repository
git clone https://github.com/username/mercedes-price-prediction-dl.git

cd mercedes-price-prediction-dl

## Install Dependencies
pip install -r requirements.txt

Usage

## Run Jupyter Notebook
jupyter notebook

## Then open:
notebooks/main.ipynb

## Model Workflow

Load Dataset

Data Analysis & Visualization

Data Preprocessing

Train-Test Split

Feature Scaling

Model Training (Deep Learning)

Model Evaluation

Price Prediction

## Example Use Case
The model can estimate the price of a vehicle using engine size, mileage, fuel efficiency, tax value, and vehicle production year.

## Future Improvements

FastAPI Model Serving

Streamlit Web Dashboard

Docker Deployment

Model Versioning (MLflow)

Hyperparameter Optimization

## Learning Outcomes

Deep Learning for Tabular Data

Regression Model Development

Data Preprocessing and Feature Scaling

Neural Network Training Workflow

Real Dataset ML Pipeline Design

Author
Developed as part of Machine Learning and AI portfolio projects.


