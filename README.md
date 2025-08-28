# Customer Churn Prediction

## Overview
This project predicts customer churn in telecommunications using the Telco Customer Churn dataset (Kaggle, 7,043 records, 21 features) through machine learning and data mining. Born from my passion for solving real-world problems, it addresses the critical issue of customer retention, costing telecoms billions annually (15–25% churn rates). This work sparked my interest in predictive analytics.

## Background and Problem
High churn due to unidentified dissatisfaction (e.g., high costs, short contracts) threatens telecom profitability. Manual analysis misses complex patterns, necessitating data-driven tools to predict and understand churn. This project proactively identifies at-risk customers and actionable patterns, enabling targeted retention strategies to reduce revenue loss.

## Feasibility
The project leverages the publicly available, anonymized Telco dataset, processed with standard Python libraries (`pandas`, `scikit-learn`, `networkx`, `mlxtend`). My skills from the Vellore AI Workshop (2021) and research ensured efficient implementation within weeks, using accessible tools and a clear pipeline (data cleaning, EDA, feature engineering, modeling, evaluation).

## Features and Techniques
- **Data Preprocessing**: Handles missing values, caps outliers, encodes categorical variables.
- **Exploratory Data Analysis (EDA)**: Visualizes churn distribution, tenure (boxplot, violin), charges, correlations, and pair plots to uncover patterns.
- **Feature Engineering**: Creates `Tenure_per_Charge`, `Service_Count`, `Contract_Tenure` for enhanced prediction.
- **Models**: Decision Tree, Random Forest, Logistic Regression with GridSearchCV; K-Means clustering (3 clusters) for segmentation.
- **Additions**:
  - **Association Rules**: Apriori generates rules (e.g., “if Contract=0 and TotalCharges high, then Churn=1,” confidence >0.6).
  - **Correlation Network**: NetworkX visualizes feature relationships (e.g., tenure-TotalCharges, weight ~0.83), reflecting community mining.
  - **Violin Plot**: Shows tenure distribution by churn.
- **Metrics**: Accuracy, precision, recall, F1-score, AUC; cross-validation ensures stability.

## Findings
- **Model Performance**: Logistic Regression excels with stable cross-validation.
- **Key Drivers**: `Contract_Tenure`, `tenure`, and `MonthlyCharges` are top predictors; short contracts and high charges drive churn.
- **Rules and Segments**: Rules highlight short-term contracts as churn risks; K-Means identifies distinct customer groups by tenure and charges.
- **Patterns**: Churners show shorter tenure and higher costs, suggesting cost sensitivity.

## Significance
- **Business**: Predicting churn saves millions by enabling retention strategies (e.g., discounts for short-term contract holders).
- **Research**: Association rules and graph analysis offers interpretable insights. The project’s pattern discovery mirrors my goal of uncovering IIoT sensor failure patterns, extended with Web3 for privacy.

## Narrative
This project tells my journey from tackling telecom churn to envisioning predictive maintenance. By uncovering why customers leave through rules, clusters, and networks, I saw parallels with predicting equipment failures in IIoT. I look to extend these techniques to sensor graphs with Web3 for secure, global analytics, aiming to reduce industrial downtime.

## Technical Integration
The pipeline integrates:
- **EDA and Feature Engineering**: Identifies and enhances churn drivers.
- **Classification**: Predicts churn with high accuracy.
- **Clustering**: Segments customers for targeted strategies.
- **Association Rules**: Provides interpretable insights.
- **Graph Analysis**: Visualizes feature connections, akin to IIoT sensor graphs.
These methods form a cohesive story, from raw data to actionable insights, mirroring my approach to IIoT pattern discovery.

## Tools
- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, NetworkX, mlxtend)
- Outputs in `outputs/` (visualizations, rules, metrics, GraphML)

## Results
- **Best Model**: Logistic Regression
- **Top Rules**: e.g., “if Contract=0, TotalCharges high, then Churn=1”
- **Visualizations**: Confusion matrices, ROC curves, correlation network, violin plot
