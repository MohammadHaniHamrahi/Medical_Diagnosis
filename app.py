from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# -------------------- تنظیمات کلی --------------------
MODEL_PATHS = {
    'breast': 'breast_cancer.csv',
    'diabetes': 'diabetes.csv',
    'heart': 'Heart_disease_cleveland_new.csv'
}

MODEL_FILES = {
    'breast': 'models/breast_model.pkl',
    'diabetes': 'models/diabetes_model.pkl',
    'heart': 'models/heart_model.pkl'
}

# ایجاد پوشه models اگر وجود ندارد
os.makedirs('models', exist_ok=True)

# -------------------- توابع عمومی --------------------
def load_model(model_type):
    """بارگذاری مدل از فایل ذخیره شده"""
    model_file = MODEL_FILES[model_type]
    if os.path.exists(model_file):
        return joblib.load(model_file)
    return None

# -------------------- ماژول سرطان پستان --------------------
def breast_cancer_load_data():
    try:
        df = pd.read_csv(MODEL_PATHS['breast'])
        df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
        df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})
        numeric_cols = df.select_dtypes(include=np.number).columns
        df = df.dropna(subset=numeric_cols)
        return df, [col for col in df.columns if col != 'diagnosis']
    except Exception as e:
        print(f"Error loading breast cancer data: {str(e)}")
        return None, str(e)

def breast_cancer_train():
    df, feature_names = breast_cancer_load_data()
    if df is None:
        return {'error': feature_names}
    
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
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': feature_names,
        'performance': {
            'accuracy': accuracy
        }
    }
    
    # ذخیره مدل
    joblib.dump(model_data, MODEL_FILES['breast'])
    return model_data

# -------------------- ماژول دیابت --------------------
def diabetes_load_data():
    try:
        data = pd.read_csv(MODEL_PATHS['diabetes'])
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
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': FEATURE_NAMES,
        'performance': {
            'accuracy': accuracy
        }
    }
    
    # ذخیره مدل
    joblib.dump(model_data, MODEL_FILES['diabetes'])
    return model_data

# -------------------- ماژول بیماری قلبی --------------------
def heart_load_data():
    try:
        df = pd.read_csv(MODEL_PATHS['heart'], na_values=['?', ' ', 'NA', ''])
    except:
        try:
            df = pd.read_csv(MODEL_PATHS['heart'], names=[
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
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': list(X.columns),
        'performance': {
            'accuracy': accuracy
        }
    }
    
    # ذخیره مدل
    joblib.dump(model_data, MODEL_FILES['heart'])
    return model_data

# -------------------- توابع پیش‌بینی --------------------
def make_prediction(model_type, input_data):
    model_data = load_model(model_type)
    if not model_data:
        return {'error': 'مدل آموزش داده نشده است. لطفاً ابتدا مدل را آموزش دهید.'}
    
    try:
        # ایجاد DataFrame از داده‌های ورودی
        input_df = pd.DataFrame([input_data], columns=model_data['features'])
        
        # استانداردسازی داده‌های ورودی
        scaled_input = model_data['scaler'].transform(input_df)
        
        # پیش‌بینی احتمال
        proba = model_data['model'].predict_proba(scaled_input)[0][1]
        
        return {'probability': proba}
    except Exception as e:
        return {'error': str(e)}

# -------------------- مسیرهای اصلی --------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train')
def train_page():
    return render_template('train.html')

# -------------------- API برای پیش‌بینی --------------------
@app.route('/predict/<model_type>', methods=['POST'])
def predict_model(model_type):
    if model_type not in MODEL_FILES:
        return jsonify({'error': 'نوع مدل نامعتبر است'}), 400
    
    input_data = request.json
    result = make_prediction(model_type, input_data)
    
    if 'error' in result:
        return jsonify(result), 400
    
    # تعیین تشخیص بر اساس نوع مدل
    proba = result['probability']
    if model_type == 'breast':
        diagnosis = 'خوش‌خیم' if proba >= 0.5 else 'بدخیم'
    elif model_type == 'diabetes':
        diagnosis = 'مثبت' if proba >= 0.5 else 'منفی'
    elif model_type == 'heart':
        diagnosis = 'در معرض خطر' if proba >= 0.5 else 'سالم'
    
    return jsonify({
        'diagnosis': diagnosis,
        'probability': proba,
        'threshold': 0.5
    })

# -------------------- API برای آموزش مدل --------------------
@app.route('/api/train/<model_type>', methods=['POST'])
def train_model(model_type):
    if model_type not in MODEL_FILES:
        return jsonify({'error': 'نوع مدل نامعتبر است'}), 400
    
    try:
        if model_type == 'breast':
            model_data = breast_cancer_train()
        elif model_type == 'diabetes':
            model_data = diabetes_train()
        elif model_type == 'heart':
            model_data = heart_train()
        
        if 'error' in model_data:
            return jsonify({'error': model_data['error']}), 500
        
        return jsonify({
            'message': 'مدل با موفقیت آموزش داده شد',
            'accuracy': model_data['performance']['accuracy']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)