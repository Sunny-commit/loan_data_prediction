# 💰 Loan Data Prediction - Credit Risk Assessment

A **machine learning system for loan approval prediction** that assesses credit risk by analyzing borrower information and determining loan eligibility.

## 🎯 Overview

This project covers:
- ✅ Credit risk classification
- ✅ Missing value handling strategies
- ✅ Credit feature engineering
- ✅ Logistic regression for probability
- ✅ Risk scoring
- ✅ ROC/AUC analysis

## 📊 Loan Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LoanDataAnalysis:
    """Analyze loan dataset"""
    
    def __init__(self, filepath='loan_data.csv'):
        self.df = pd.read_csv(filepath)
    
    def explore_data(self):
        """Dataset exploration"""
        print(f"Shape: {self.df.shape}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nTarget distribution:")
        print(self.df['Loan_Status'].value_counts())
        
        # Loan statistics
        print(f"\nLoan amount statistics:\n{self.df['LoanAmount'].describe()}")
        print(f"Annual income statistics:\n{self.df['ApplicantIncome'].describe()}")
    
    def risk_analysis(self):
        """Analyze risk factors"""
        # Approve rate by employment
        print("Approval rate by employment:")
        print(self.df.groupby('Self_Employed')['Loan_Status'].value_counts(normalize=True))
        
        # Approve rate by credit history
        print("\nApproval rate by credit history:")
        print(self.df.groupby('Credit_History')['Loan_Status'].value_counts(normalize=True))
```

## 🔧 Feature Engineering

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class LoanFeatureEngineer:
    """Engineer credit features"""
    
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.encoders = {}
    
    def fill_missing_values(self, df):
        """Handle missing data"""
        df_copy = df.copy()
        
        # Numerical: median
        df_copy['LoanAmount'].fillna(df_copy['LoanAmount'].median(), inplace=True)
        df_copy['Loan_Amount_Term'].fillna(df_copy['Loan_Amount_Term'].mode()[0], inplace=True)
        
        # Categorical: mode
        df_copy['Gender'].fillna(df_copy['Gender'].mode()[0], inplace=True)
        df_copy['Married'].fillna(df_copy['Married'].mode()[0], inplace=True)
        df_copy['Self_Employed'].fillna('No', inplace=True)
        df_copy['Credit_History'].fillna(1, inplace=True)
        
        return df_copy
    
    def create_credit_features(self, df):
        """Engineer credit assessment features"""
        df_copy = df.copy()
        
        # Debt to income ratio
        df_copy['DebtToIncomeRatio'] = (df_copy['LoanAmount'] / df_copy['ApplicantIncome']).fillna(0)
        
        # Total annual income
        df_copy['TotalIncome'] = df_copy['ApplicantIncome'] + df_copy['CoapplicantIncome']
        
        # Loan to income ratio
        df_copy['LoanToIncomeRatio'] = df_copy['LoanAmount'] / df_copy['TotalIncome']
        
        # EMI per month
        df_copy['EMI'] = df_copy['LoanAmount'] / df_copy['Loan_Amount_Term']
        df_copy['EMI_to_Income'] = (df_copy['EMI'] * 12) / df_copy['TotalIncome']
        
        # Dependents risk
        df_copy['DependentRisk'] = df_copy['Dependents'].astype(int)
        
        # Employment stability (married + stable job)
        df_copy['EmploymentStability'] = ((df_copy['Married'] == 'Yes') & 
                                         (df_copy['Self_Employed'] == 'No')).astype(int)
        
        return df_copy
    
    def encode_categorical(self, df):
        """Encode categorical features"""
        df_copy = df.copy()
        
        categorical_features = ['Gender', 'Married', 'Education', 'Self_Employed', 'Loan_Status']
        
        for feature in categorical_features:
            if feature not in self.encoders:
                self.encoders[feature] = LabelEncoder()
            
            df_copy[feature] = self.encoders[feature].fit_transform(df[feature])
        
        return df_copy
```

## 🎯 Risk Scoring

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

class LoanRiskScorer:
    """Score loan risk"""
    
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.risk_threshold = 0.5
    
    def fit(self, X_train, y_train):
        """Train risk model"""
        self.model.fit(X_train, y_train)
    
    def calculate_risk_score(self, X_test):
        """Calculate probability of default"""
        probabilities = self.model.predict_proba(X_test)[:, 1]
        
        return probabilities
    
    def classify_risk(self, risk_scores, low=0.3, medium=0.7):
        """Classify borrowers"""
        classifications = []
        
        for score in risk_scores:
            if score < low:
                classifications.append('Low Risk')
            elif score < medium:
                classifications.append('Medium Risk')
            else:
                classifications.append('High Risk')
        
        return classifications
    
    def generate_decision(self, risk_scores, feature_scores):
        """Generate loan decision"""
        decisions = []
        reasons = []
        
        for score in risk_scores:
            if score < 0.3:
                decisions.append('APPROVED')
                reasons.append('Low default risk')
            elif score < 0.7:
                decisions.append('MANUAL REVIEW')
                reasons.append('Moderate risk - requires human review')
            else:
                decisions.append('REJECTED')
                reasons.append('High default risk')
        
        return pd.DataFrame({
            'Decision': decisions,
            'Risk_Score': risk_scores,
            'Reason': reasons
        })
```

## 📊 Model Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

class LoanModelEvaluator:
    """Evaluate loan prediction"""
    
    @staticmethod
    def evaluate(y_true, y_pred, y_pred_proba):
        """Comprehensive evaluation"""
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
        print(f"\nROC-AUC Score: {roc_auc_score(y_true, y_pred_proba):.4f}")
    
    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC={auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Loan Approval')
        plt.legend()
        plt.grid()
        plt.show()
    
    @staticmethod
    def feature_importance(model, feature_names):
        """Extract coefficients"""
        coefficients = abs(model.coef_[0])
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        }).sort_values('Coefficient', ascending=False)
        
        print("\nFeature Importance (Coefficients):")
        print(importance_df.head(10))
```

## 💡 Interview Talking Points

**Q: Handle class imbalance?**
```
Answer:
- Class weights (penalize minority more)
- Stratified cross-validation
- Adjust decision threshold
- F1 score metric
```

**Q: Regulatory compliance?**
```
Answer:
- Fair lending regulations
- Feature importance for explanations
- Bias detection across demographics
- Automated decision review
```

## 🌟 Portfolio Value

✅ Credit risk assessment
✅ Missing value strategies
✅ Feature engineering (financial domain)
✅ Logistic regression interpretation
✅ Risk scoring
✅ Classification metrics
✅ Regulatory considerations

---

**Technologies**: Scikit-learn, Pandas, NumPy

