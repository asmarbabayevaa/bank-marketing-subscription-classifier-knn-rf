# Bank Marketing Classification — KNN & Random Forest
Predicting term deposit subscription likelihood for bank customers using machine learning. The project covers full pipeline from raw data to scoring new customers with probability estimates.

## Business Problem
A bank runs outbound marketing campaigns to get customers to subscribe to a term deposit. Not every customer is worth calling — this model scores each customer with a subscription probability so the campaign team can prioritize the most likely subscribers and reduce wasted outreach.

## Dataset

Training data: marketing.csv — historical campaign records with known outcomes
Deployment data: marketing_test.xlsx — new customers to be scored
Target variable: result (whether the customer subscribed)

Key features include customer demographics (age, job, marital status, education), financial indicators (balance, housing loan, personal loan, default status), and previous campaign interaction history.

Project Structure
├── notebook.ipynb         # Main notebook — full pipeline
├── marketing.csv          # Training dataset
├── marketing_test.xlsx    # Deployment dataset
├── scaler.pkl             # Saved StandardScaler (fitted on training data)
└── README.md

## Pipeline Overview

Data Cleaning — drop irrelevant columns, handle nulls
Feature Engineering — create domain-driven features:

financial_pressure — housing + loan burden score
wealthy_no_debt — high balance with no liabilities
risk_profile — default + financial pressure combined
previous_campaign_success — binary flag from last campaign outcome
balance_per_age — normalized wealth indicator


Model-Specific Preprocessing — separate pipelines for RF and KNN
Outlier Treatment — IQR capping for KNN pipeline
Feature Selection — Spearman correlation, inter-correlation check, VIF analysis
Modeling — baseline KNN and Random Forest
Hyperparameter Optimization — Optuna (50 trials for KNN, 30 for RF)
Model Comparison — Gini scores across all four models
Deployment — score new customers using the winning model


Model Results
ModelTrain GiniTest GiniKNN (baseline)——Random Forest (baseline)——KNN + Optuna——Random Forest + Optuna——

Winner: KNN Optuna — selected based on best Test Gini score.

Evaluation Metrics

Gini coefficient — primary metric (derived from ROC-AUC)
Precision — of predicted subscribers, how many actually subscribed
Recall — of actual subscribers, how many were captured
Confusion matrix — breakdown of prediction outcomes

