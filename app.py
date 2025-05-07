from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# -------------------- تنظیمات کلی --------------------
MODEL_PATHS = {
    '1': 'breast_cancer.csv',
    '2': 'diabetes.csv',
    '3': 'Heart_disease_cleveland_new.csv'
}

MODEL_PREFIXES = {
    '1': 'breast_model_',
    '2': 'diabetes_model_',
    '3': 'heart_model_'
}

# -------------------- توابع عمومی --------------------
@app.route('/')
def home():
    return render_template('index.html')

# -------------------- ماژول سرطان پستان --------------------
def breast_cancer_load_data():
    try:
        df = pd.read_csv(MODEL_PATHS['1'])
        df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
        df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})
        numeric_cols = df.select_dtypes(include=np.number).columns
        df = df.dropna(subset=numeric_cols)
        return df, [col for col in df.columns if col != 'diagnosis']
    except Exception as e:
        return None, str(e)

def breast_cancer_train():
    df, feature_names = breast_cancer_load_data()
    if df is None:
        return {'error': feature_names}  # feature_names contains error message here
    
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': feature_names,
        'performance': {
            'accuracy': accuracy
        }
    }
    joblib.dump(model_data, f"{MODEL_PREFIXES['1']}{timestamp}.pkl")
    return model_data

# -------------------- ماژول دیابت --------------------
def diabetes_load_data():
    try:
        data = pd.read_csv(MODEL_PATHS['2'])
        cols_to_replace_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        data[cols_to_replace_zero] = data[cols_to_replace_zero].replace(0, np.nan)
        data.fillna(data.median(), inplace=True)
        return data, None
    except Exception as e:
        return None, str(e)

def diabetes_train():
    data, error = diabetes_load_data()
    if data is None:
        return {'error': error}
    
    FEATURE_NAMES = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    
    X = data[FEATURE_NAMES]
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_estimators=150,
        learning_rate=0.1
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': FEATURE_NAMES,
        'performance': {
            'accuracy': accuracy
        }
    }
    joblib.dump(model_data, f"{MODEL_PREFIXES['2']}{timestamp}.pkl")
    return model_data

# -------------------- ماژول بیماری قلبی --------------------
def heart_load_data():
    try:
        df = pd.read_csv(MODEL_PATHS['3'], na_values=['?', ' ', 'NA', ''])
    except:
        try:
            df = pd.read_csv(MODEL_PATHS['3'], names=[
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
            ], na_values=['?', ' ', 'NA', ''])
        except Exception as e:
            return None, str(e)
    
    df = df.dropna()
    df['target'] = pd.to_numeric(df['target'], errors='coerce').fillna(0)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    return df, None

def heart_train():
    df, error = heart_load_data()
    if df is None:
        return {'error': error}
    
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': list(X.columns),
        'performance': {
            'accuracy': accuracy
        }
    }
    joblib.dump(model_data, f"{MODEL_PREFIXES['3']}{timestamp}.pkl")
    return model_data

# -------------------- توابع پیش‌بینی --------------------
def predict(model_prefix, input_data, feature_names):
    try:
        model_files = [f for f in os.listdir() if f.startswith(model_prefix)]
        if not model_files:
            return {'error': 'مدلی یافت نشد. ابتدا مدل را آموزش دهید'}
        
        latest_model = sorted(model_files)[-1]
        model_data = joblib.load(latest_model)
        
        input_df = pd.DataFrame([input_data], columns=feature_names)
        scaled_input = model_data['scaler'].transform(input_df)
        proba = model_data['model'].predict_proba(scaled_input)[0][1]
        
        return {'probability': proba}
    except Exception as e:
        return {'error': str(e)}

# -------------------- مسیرهای API --------------------
@app.route('/train/<model_type>', methods=['POST'])
def train_model(model_type):
    if model_type == 'breast':
        result = breast_cancer_train()
    elif model_type == 'diabetes':
        result = diabetes_train()
    elif model_type == 'heart':
        result = heart_train()
    else:
        return jsonify({'error': 'نوع مدل نامعتبر'}), 400
    
    if 'error' in result:
        return jsonify(result), 500
    
    return jsonify({
        'message': 'مدل با موفقیت آموزش داده شد',
        'accuracy': result['performance']['accuracy']
    })

@app.route('/predict/breast', methods=['POST'])
def predict_breast():
    try:
        df, feature_names = breast_cancer_load_data()
        if df is None:
            return jsonify({'error': feature_names}), 500
        
        input_data = request.json
        result = predict(MODEL_PREFIXES['1'], input_data, feature_names)
        
        if 'error' in result:
            return jsonify(result), 500
        
        diagnosis = 'خوش‌خیم' if result['probability'] >= 0.5 else 'بدخیم'
        return jsonify({
            'diagnosis': diagnosis,
            'probability': result['probability'],
            'threshold': 0.5
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        _, error = diabetes_load_data()
        if error:
            return jsonify({'error': error}), 500
        
        input_data = request.json
        model_files = [f for f in os.listdir() if f.startswith(MODEL_PREFIXES['2'])]
        if not model_files:
            return jsonify({'error': 'مدلی یافت نشد. ابتدا مدل را آموزش دهید'}), 400
        
        latest_model = sorted(model_files)[-1]
        model_data = joblib.load(latest_model)
        
        result = predict(MODEL_PREFIXES['2'], input_data, model_data['features'])
        
        if 'error' in result:
            return jsonify(result), 500
        
        diagnosis = 'مثبت' if result['probability'] >= 0.5 else 'منفی'
        return jsonify({
            'diagnosis': diagnosis,
            'probability': result['probability'],
            'threshold': 0.5
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    try:
        _, error = heart_load_data()
        if error:
            return jsonify({'error': error}), 500
        
        input_data = request.json
        model_files = [f for f in os.listdir() if f.startswith(MODEL_PREFIXES['3'])]
        if not model_files:
            return jsonify({'error': 'مدلی یافت نشد. ابتدا مدل را آموزش دهید'}), 400
        
        latest_model = sorted(model_files)[-1]
        model_data = joblib.load(latest_model)
        
        result = predict(MODEL_PREFIXES['3'], input_data, model_data['features'])
        
        if 'error' in result:
            return jsonify(result), 500
        
        diagnosis = 'در معرض خطر' if result['probability'] >= 0.5 else 'سالم'
        return jsonify({
            'diagnosis': diagnosis,
            'probability': result['probability'],
            'threshold': 0.5
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)