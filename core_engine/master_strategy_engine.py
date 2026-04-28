# =====================================================================
# LENDING CLUB CAPSTONE: FULL DATASET (2.2M ROWS) + SMOTE
# PLATFORM: KAGGLE (FREE 30GB RAM OPTIMIZED)
# =====================================================================

# 1. Install required libraries
!pip install lifelines imbalanced-learn xgboost

import os
import pandas as pd
import numpy as np
import warnings
import re
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lifelines import CoxPHFitter

warnings.filterwarnings('ignore')

# =====================================================================
# PHASE 1 & 2: DATA LOADING & FULL DATASET OPTIMIZATION
# =====================================================================
def load_and_clean_full_data(file_path):
    print(f"--- STARTING FULL DATASET INGESTION (2.2M ROWS) ---")
    
    # Load all rows using chunksize/low_memory=False for efficiency
    df = pd.read_csv(file_path, low_memory=False)
    
    # 2.1 Target Variable & Filtering
    print("Filtering definitive loan statuses...")
    definitive_statuses = ['Fully Paid', 'Charged Off', 'Default']
    df = df[df['loan_status'].isin(definitive_statuses)]
    df['is_bad'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1, 'Default': 1}).astype('int8')
    
    # 2.2 MEMORY DOWNCASTING (VITAL FOR FULL DATASET)
    # Reduces 64-bit floats/ints to 32/8-bit to fit in 30GB RAM
    print("Downcasting memory to fit 30GB limit...")
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')

    # Date Handling
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'], format='%b-%Y', errors='coerce')
    df['time_on_books_months'] = (df['last_pymnt_d'] - df['issue_d']) / np.timedelta64(1, 'D')
    df['time_on_books_months'] = (df['time_on_books_months'] / 30).fillna(1).astype('float32')
    df['time_on_books_months'] = df['time_on_books_months'].apply(lambda x: max(1, round(x)))

    # 2.3 Drop Leaky & High-Cardinality Columns
    # These columns cause memory explosions during one-hot encoding
    drop_cols = [
        'loan_status', 'funded_amnt', 'funded_amnt_inv', 'out_prncp', 'out_prncp_inv',
        'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 
        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 
        'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
        'emp_title', 'title', 'zip_code', 'url', 'desc', 'id', 'member_id', 'policy_code'
    ]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    # Missing Values
    missing_pct = df.isnull().mean()
    df.drop(columns=missing_pct[missing_pct > 0.40].index, inplace=True)
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in num_cols: df[col].fillna(df[col].median(), inplace=True)
    for col in cat_cols: 
        if col != 'issue_d': df[col].fillna('Missing', inplace=True)
            
    print(f"Data ingested and cleaned. Total Rows: {len(df):,}")
    return df

# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    # Deep file search to find the CSV in any directory
    base_path = "/kaggle/input"
    file_path = None
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(('.csv', '.csv.gz')) and 'accepted' in file.lower():
                file_path = os.path.join(root, file)
                break
        if file_path: break

    if file_path:
        try:
            # 1. Load COMPLETE dataset
            df = load_and_clean_full_data(file_path)
            
            # 2. Prepare ML data
            df_grades = df[['grade', 'loan_amnt', 'int_rate']].copy()
            ml_df = df.drop(columns=['issue_d'])
            
            # 3. One-Hot Encoding
            print("Performing One-Hot Encoding on categorical features...")
            ml_df = pd.get_dummies(ml_df, drop_first=True)
            
            X = ml_df.drop(columns=['is_bad', 'time_on_books_months'])
            y = ml_df['is_bad']
            
            # 4. XGBoost Name Sanitization
            X.columns = [re.sub(r"[\[\]<]", "_", str(col)) for col in X.columns]
            
            # 5. Train/Test Split (80/20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # 6. SMOTE (OVER-SAMPLING)
            # This is where the 30GB RAM is put to the test. 
            print("\n--- PHASE 4: SMOTE OVER-SAMPLING (BALANCING 2.2M RECORDS) ---")
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            print(f"Balanced Dataset Size: {len(X_train_res):,} rows")
            
            # 7. XGBOOST TRAINING
            print("Training XGBoost Classifier on full portfolio (may take ~15-20 mins)...")
            xgb_model = XGBClassifier(
                n_estimators=100, 
                max_depth=4, 
                learning_rate=0.1, 
                eval_metric='logloss',
                tree_method='hist' # Uses optimized histogram method for large datasets
            )
            xgb_model.fit(X_train_res, y_train_res)
            
            # 8. PERFORMANCE EVALUATION
            y_pred = xgb_model.predict(X_test)
            print("\n--- ML PERFORMANCE REPORT ---")
            print(classification_report(y_test, y_pred))
            
            # 9. BUSINESS CASE: UNLOCKED REVENUE
            print("\n--- PHASE 5: BUSINESS CASE & REVENUE EXTRACTION ---")
            df_grades['risk_score'] = xgb_model.predict_proba(X)[:, 1]
            avg_risk_b = df_grades[df_grades['grade'] == 'B']['risk_score'].mean()
            
            # Identify Hidden Gems: Grade D/E borrowers with Grade B risk levels
            hidden_gems = df_grades[
                (df_grades['grade'].isin(['D', 'E'])) & 
                (df_grades['risk_score'] < avg_risk_b)
            ]
            
            total_vol = hidden_gems['loan_amnt'].sum()
            avg_rate = hidden_gems['int_rate'].mean() / 100
            revenue = total_vol * avg_rate
            
            print(f"======================================================")
            print(f"ENTERPRISE-WIDE ANALYSIS (FULL 2.2M PORTFOLIO):")
            print(f"Total 'False Negatives' (Hidden Gems) Found: {len(hidden_gems):,}")
            print(f"Total Loan Volume in Hidden Segment: ${total_vol:,.2f}")
            print(f"TOTAL UNLOCKED ANNUAL REVENUE: ${revenue:,.2f}")
            print(f"======================================================")
            
        except Exception as e:
            print(f"Critical Error during execution: {str(e)}")
    else:
        print("Error: LendingClub dataset not found in /kaggle/input.")
