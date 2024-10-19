# Customer-Subscription-Prediction-for-Long-Term-Deposits
# Project Overview
This project is a final exam assignment for the Model Deployment course in the 4th semester. It focuses on predicting customer subscription for long-term deposits and deploying the prediction model using FastAPI. The goal of the project is to build a machine learning model that can predict whether a customer will subscribe to a long-term deposit plan based on various customer attributes such as age, job type, marital status, education, and previous interactions with the bank. The model is then deployed using FastAPI to allow external applications to make predictions via an API.

# Case Description
Banks often face challenges in attracting customers to long-term deposit plans. This project aims to assist marketing efforts by building a predictive model to identify which customers are most likely to subscribe to these deposit plans. By focusing on the customers with a higher probability of subscribing, the bank can optimize its marketing resources. The classification target is a binary label: "yes" for customers likely to subscribe and "no" for those who are not.
The data provided includes several customer features such as:
- Demographics: Age, job, marital status, education.
- Economic indicators: Housing, loan status, and balance.
- Previous marketing campaign data: Contact type, number of contacts, days since last contact.
- Outcome of previous campaigns.

# Objectives
- Build a Machine Learning Model to predict customer subscription to long-term deposits.
- Optimize the model performance using hyperparameter tuning.
- Deploy the model using FastAPI, allowing external applications to make predictions via an API.
- Evaluate model performance using accuracy, precision, recall, and F1-score metrics.

# Project Steps and Features
1. Data Preprocessing
- Handling missing values.
- Encoding categorical variables using One-Hot Encoding and Label Encoding.
- Scaling numerical features using StandardScaler.

2. Model Building
- Several machine learning models were trained and compared, including:
  - Random Forest
  - XGBoost
  - Gradient Boosting
  - Logistic Regression
- Grid Search was used to find the best hyperparameters for each model.

4. Model Evaluation
- The models were evaluated using accuracy, precision, recall, F1-score, and ROC AUC metrics.
- Cross-validation was applied to ensure the model generalizes well.
5. Model Deployment with FastAPI
- The best-performing model, Random Forest, was serialized using pickle and deployed using FastAPI.
- The API accepts customer data in JSON format and returns a prediction of whether the customer is likely to subscribe to a long-term deposit.
- A simple user interface was created using Streamlit to allow users to input customer details and receive predictions.

# Tools
- Python: For model development.
- Libraries:
  - Pandas: For data manipulation.
  - NumPy: For numerical computations.
  - Scikit-learn: For building and evaluating machine learning models (includes modules like train_test_split, GridSearchCV, StandardScaler, RandomForestClassifier, LogisticRegression, and evaluation metrics such as classification_report).
  - XGBoost: For implementing the XGBoost classification model.
  - Pickle: For saving and loading the trained models.
  - Seaborn: For data visualization.
  - Matplotlib: For plotting visualizations.
  - Shutil: For file operations (e.g., copying and deleting files).
  - IPython.display: For creating file links.

# Challenges
- Imbalanced Dataset: The dataset had more customers who did not subscribe compared to those who did. This imbalance was handled by tuning the classification threshold and using metrics like precision and recall.
- Hyperparameter Tuning: Finding the best hyperparameters for models like Random Forest and XGBoost required multiple iterations and long training times.
- Model Deployment: Deploying the model using FastAPI required careful handling of input data formats and ensuring the model was properly serialized and loaded.

# Conclusion
This project successfully developed and deployed a machine learning model to predict customer subscriptions to long-term deposits. The model was built using various classification algorithms and deployed using FastAPI to provide real-time predictions via an API. The best-performing model was XGBoost Tuned, with strong performance metrics.

