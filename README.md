🏦 Loan Approval Prediction Using Machine Learning
This project takes on the real-world challenge of predicting whether a loan application will be approved or not, based on applicant details like income, credit history, education, and more. It's a classic supervised learning problem, and here we apply several ML models to find the best fit!

📁 Dataset Overview
The dataset loan_data.csv includes various features related to loan applicants:

Gender, Married, Dependents, Education, Self_Employed

ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term

Credit_History, Property_Area

Loan_Status (Target)

🔍 What This Project Covers
This project walks through the entire data science pipeline — from messy raw data to a tuned ML model ready to make predictions.

🧹 1. Data Cleaning & Preprocessing
Filled missing numerical values using mean

Filled missing categorical values using mode

Applied log transformations on skewed features (like ApplicantIncome, LoanAmount)

Created a new feature: TotalIncome = ApplicantIncome + CoapplicantIncome

Dropped irrelevant columns (e.g., Loan_ID and ApplicantIncome after creating TotalIncome)

Encoded categorical features using LabelEncoder

📊 2. Exploratory Data Analysis (EDA)
Histograms and distribution plots for understanding variable spread

Countplots to inspect class distributions

Correlation matrix visualized using a heatmap to detect strong/weak relationships

🧠 3. Model Building & Evaluation
Trained and compared multiple ML models:

Logistic Regression

Decision Tree

Random Forest

Extra Trees Classifier

Each model was evaluated with:

Accuracy score

Cross-validation score

Confusion Matrix and heatmap

Additionally, a Random Forest model was fine-tuned using hyperparameters (n_estimators, max_depth, min_samples_split).

🧪 Sample Output
matlab
Copy
Edit
the accuracy is: 82.19%
the cross validation is: 80.34%
Confusion matrix heatmaps helped visualize where the model misclassified approvals vs rejections.

🛠️ Tech Stack
Python

Pandas, NumPy for data handling

Matplotlib, Seaborn for visualization

Scikit-learn for ML models and evaluation

▶️ How to Run
Make sure you have the loan_data.csv file in the correct path (/content/loan_data.csv for Colab).

Open and run loan_data_prediction.py in a Python IDE or Colab.

Explore the EDA visuals, model performance, and confusion matrices.
