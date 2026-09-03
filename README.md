# Employee Turnover Analytics using Machine Learning

## Project Overview

This project analyzes employee turnover using an HR dataset and
machine learning techniques.

The main objective is to identify factors associated with employee
turnover and build a machine learning model to predict whether an
employee is likely to leave the organization.

## Objectives

- Perform data quality checking
- Perform Exploratory Data Analysis (EDA)
- Identify employee groups using clustering
- Handle class imbalance using SMOTE
- Evaluate models using 5-Fold Cross Validation
- Select the best machine learning model
- Suggest employee retention strategies

## Dataset

The dataset contains employee information such as:

- Satisfaction Level
- Last Evaluation
- Number of Projects
- Average Monthly Hours
- Time Spent in Company
- Work Accidents
- Promotion in Last 5 Years
- Department
- Salary
- Employee Turnover (Left)

Target variable:

- `0` = Employee stayed
- `1` = Employee left

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- K-Means Clustering
- SMOTE
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

## Project Workflow

### 1. Data Quality Check

The dataset was checked for:

- Missing values
- Duplicate records
- Data types
- Statistical summary
- Class distribution

Duplicate records were removed before machine learning.

### 2. Exploratory Data Analysis

EDA was performed to understand the relationship between employee
turnover and:

- Salary
- Department
- Satisfaction level
- Monthly working hours

### 3. Clustering

K-Means clustering was used to group employees based on:

- Satisfaction
- Evaluation
- Number of projects
- Monthly working hours
- Time spent in company

Three employee clusters were created.

### 4. SMOTE

The target variable was imbalanced because fewer employees left than
stayed.

SMOTE (Synthetic Minority Oversampling Technique) was used on the
training data to balance the classes.

### 5. 5-Fold Cross Validation

Three machine learning algorithms were evaluated:

- Logistic Regression
- Random Forest
- Support Vector Machine

Stratified 5-Fold Cross Validation was used for model evaluation.

### 6. Best Model Selection

Random Forest achieved the best cross-validation performance.

**Best Model: Random Forest**

**5-Fold Cross Validation Accuracy: 97.72%**

**Test Accuracy: Approximately 98.3%**

### 7. Employee Retention Strategies

Based on the analysis, organizations can:

- Improve employee satisfaction
- Monitor excessive working hours
- Review salary structures
- Provide career growth opportunities
- Reduce excessive workload
- Introduce employee recognition programs
- Identify high-risk employees using predictive analytics

## Results

The machine learning analysis successfully identified employee
turnover patterns.

Random Forest was selected as the best-performing model with a
5-Fold Cross Validation accuracy of approximately 97.72%.

The model achieved approximately 98.3% accuracy on the test dataset.

## Conclusion

This project demonstrates how data analytics and machine learning
can help organizations understand employee turnover.

By identifying employees who are more likely to leave, HR teams can
take preventive actions such as improving job satisfaction, managing
workload, providing career opportunities, and improving employee
recognition.

## Project Structure

```text
Employee-Turnover-Analytics/
│
├── HR_comma_sep.csv
├── employee_turnover.py
└── README.md
# Employee Turnover Analytics using Machine Learning

**Author:** Digvijaysinh Thorve

## Project Overview
