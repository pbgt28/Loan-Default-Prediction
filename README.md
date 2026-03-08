Loan Default Prediction using Machine Learning

📌 Project Overview

This project aims to predict whether a borrower will default on a loan using Machine Learning techniques. Financial institutions face significant risk from loan defaults, and predictive models can help identify high-risk applicants before loan approval.

The project applies multiple machine learning algorithms and evaluates them using performance metrics such as accuracy, precision, recall, F1-score, confusion matrix, and precision-recall curve.

🎯 Objective

The main objectives of this project are:
	•	Predict the probability of loan default.
	•	Compare the performance of different machine learning models.
	•	Handle imbalanced data using techniques like SMOTE.
	•	Evaluate models using classification metrics and PR curves.

📊 Dataset: https://www.kaggle.com/datasets/nikhil1e9/loan-default

The dataset contains borrower information including financial, demographic, and credit-related features.

Example features include:
	•	Income
	•	Loan amount
	•	Credit history
	•	Loan term
	•	Employment details
	•	Debt-to-income ratio
	•	Loan status (Target Variable)

Target Variable
	•	Loan Default
	•	0 → No Default
	•	1 → Default

⚙️ Technologies Used
	•	Python
	•	NumPy
	•	Pandas
	•	Scikit-learn
	•	Matplotlib
	•	Seaborn
	•	imbalanced-learn (SMOTE)


🧠 Machine Learning Models Implemented

The following algorithms were used:
	1.	Logistic Regression
	2.	Random Forest
	3.	K-Nearest Neighbors (KNN)

These models were trained and evaluated to determine the most effective approach for predicting loan default.

🔧 Data Preprocessing

Steps performed before training the models:
	•	Handling missing values
	•	Feature encoding
	•	Feature scaling (for KNN)
	•	Handling class imbalance using SMOTE
	•	Train-test split
  
🔍 Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the structure of the dataset and identify patterns that influence loan defaults.

Key EDA Steps
	•	Checking dataset shape and structure
	•	Handling missing values
	•	Statistical summary of features
	•	Distribution analysis of numerical features
	•	Analysis of categorical variables
	•	Correlation analysis between features
	•	Identifying class imbalance in the target variable

Visualizations Used
	•	Histograms for feature distribution
	•	Bar plots for categorical features
	•	Correlation heatmap
	•	Loan default distribution pie chart

EDA helps identify important variables affecting loan default risk, such as income, credit score, and debt-to-income ratio.

📈 Model Evaluation Metrics

The models were evaluated using:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1 Score
	•	Confusion Matrix
	•	Precision-Recall Curve

Precision-Recall curves were particularly useful due to the class imbalance in loan default data.
