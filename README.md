# 🚦 Hourly Traffic Volume Forecasting 

## Project Overview
This repository contains an end-to-end machine learning internship project designed to accurately forecast hourly traffic volumes at various road junctions. By leveraging historical traffic data, temporal features, and weather conditions, this project deploys a predictive model via an interactive web dashboard.

## Features
* **Exploratory Data Analysis (EDA):** Comprehensive analysis of traffic patterns across different junctions, times of day, and weather conditions.
* **Machine Learning:** A Random Forest Regressor trained and refined through hyperparameter tuning and cross-validation to capture non-linear traffic trends.
* **Interactive Dashboard:** A live web application built with Streamlit allowing users to input specific dates, times, and weather metrics to get real-time traffic volume estimates.

## Repository Structure
* `Component_3_EDA_Data_Preparation.ipynb`: Data cleaning, temporal feature engineering, and exploratory visualizations.
* `Component_5_Trained_Models_Model_Evaluation.ipynb`: Model training, comparison (Linear Regression, Random Forest, Gradient Boosting), and final evaluation.
* `integrated_traffic_dataset.csv`: The historical dataset used for training.
* `app.py`: The Streamlit web application code.
* `traffic_volume_random_forest_refined.pkl`: The compressed, trained machine learning model.
* `requirements.txt`: Dependencies required to run the web application.

## Technologies Used
* **Python** (Data manipulation and modeling)
* **Pandas & NumPy** (Data processing)
* **Scikit-Learn** (Machine learning pipeline)
* **Streamlit** (Web application deployment)
* **Matplotlib & Seaborn** (Data visualization)