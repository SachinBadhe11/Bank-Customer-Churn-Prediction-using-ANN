Bank Customer Churn Prediction using Artificial Neural Networks

This repository implements a machine learning pipeline to predict customer churn using Artificial Neural Networks (ANNs). The project focuses on identifying customers at risk of churning and uncovering key factors that contribute to churn.

Features

Comprehensive Exploratory Data Analysis (EDA): Gain insights into the data through visualizations and descriptive statistics.
Feature Engineering: Create new features to enhance model performance.
Machine Learning Models: Develop and evaluate ANNs for customer churn prediction.
Customer Churn Insights: Identify the most significant factors influencing customer churn.
Getting Started

Prerequisites

Python 3.8 or higher
Required Libraries

pandas
numpy
matplotlib
seaborn
tensorflow # Deep learning framework for ANNs
scikit-learn # For data preprocessing and model evaluation
Running the Project

Option 1: Google Colab

Upload the Bank_Customer_Churn.ipynb notebook to your Google Drive.

Open Google Colab and import the notebook from your Drive.

Install missing dependencies using the following command:

Bash

!pip install pandas numpy matplotlib seaborn tensorflow scikit-learn
Follow the step-by-step instructions in the notebook to execute the analysis and predictions.

Option 2: Local Machine

Install required dependencies:

Bash

pip install -r requirements.txt
Open the notebook in Jupyter Notebook or your preferred IDE and run the cells.

Bash

jupyter notebook Bank_Customer_Churn.ipynb
Project Usage

Data Loading: Ensure the customer churn dataset is available in the specified path.
Exploratory Data Analysis: Explore data distributions, correlations, and missing values.
Feature Engineering: Generate new features and preprocess the data for machine learning.
Model Training: Train various ANN architectures and evaluate their performance using metrics like accuracy, precision, recall, and F1-score.
Prediction: Utilize the trained model to predict churn probabilities for new customer data.
Results Analysis: Analyze the model's performance metrics and identify the most important features impacting churn through visualizations.
Repository Structure

Bank_Customer_Churn/
├── Bank_Customer_Churn.ipynb        # Main Jupyter notebook
├── README.md                         # Project documentation
├── requirements.txt                  # Required Python packages
├── data/                             # Folder to store datasets (if applicable)
├── models/                           # Optional folder to store trained models
├── utils/                            # Optional folder for utility functions
└── ...                               # Additional project-specific folders
Contributing

We welcome contributions to this project! Feel free to submit pull requests for improvements, bug fixes, or new functionalities. Please ensure your code adheres to PEP 8 style guidelines and includes unit tests.
