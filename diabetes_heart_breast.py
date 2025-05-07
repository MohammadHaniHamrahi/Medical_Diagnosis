import pandas as pd
import numpy as np
import joblib
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------- تنظیمات کلی --------------------
MODEL_PATHS = {
    '1': 'datasets/breast_cancer.csv',
    '2': 'datasets/diabetes.csv',
    '3': 'datasets/Heart_disease_cleveland_new.csv'
}

MODEL_PREFIXES = {
    '1': 'breast_model_',
    '2': 'diabetes_model_',
    '3': 'heart_model_'
}

def delete_old_models(prefix):
    model_files = [f for f in os.listdir() if f.startswith(prefix)]
    for f in model_files:
        os.remove(f)

# -------------------- توابع عمومی --------------------
def show_menu():
    print("\n" + "="*40)
    print("سیستم تشخیص پزشکی یکپارچه")
    print("="*40)
    print("1. تشخیص سرطان پستان")
    print("2. تشخیص دیابت")
    print("3. تشخیص بیماری قلبی")
    print("4. خروج")
    return input("لطفاً گزینه مورد نظر را انتخاب کنید: ")

# -------------------- ماژول سرطان پستان --------------------
def breast_cancer_load_data():
    try:
        df = pd.read_csv(MODEL_PATHS['1'])
        df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
        df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})
        
        # اطمینان از وجود تمام ستون‌ها
        required_features = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 
            'area_mean', 'smoothness_mean', 'compactness_mean',
            'concavity_mean', 'concave points_mean', 'symmetry_mean', 
            'fractal_dimension_mean', 'radius_se', 'texture_se', 
            'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
            'concavity_se', 'concave points_se', 'symmetry_se', 
            'fractal_dimension_se', 'radius_worst', 'texture_worst',
            'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave points_worst',
            'symmetry_worst', 'fractal_dimension_worst'
        ]
        
        # حذف سطرهای دارای مقادیر NaN
        df = df.dropna(subset=required_features)
        
        return df, required_features
    except Exception as e:
        print(f"Error: {str(e)}")
        exit()

def breast_cancer_train():
    df, feature_names = breast_cancer_load_data()
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
    print("\nارزیابی سرطان پستان:")
    print("دقت:", accuracy_score(y_test, y_pred))
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': feature_names,
        'performance': {
            'accuracy': accuracy_score(y_test, y_pred)
        }
    }
    delete_old_models(MODEL_PREFIXES['1'])
    joblib.dump(model_data, f"{MODEL_PREFIXES['1']}{timestamp}.pkl")
    return model_data

# -------------------- ماژول دیابت --------------------
def diabetes_load_data():
    data = pd.read_csv(MODEL_PATHS['2'])
    cols_to_replace_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[cols_to_replace_zero] = data[cols_to_replace_zero].replace(0, np.nan)
    data.fillna(data.median(), inplace=True)
    return data

def diabetes_train():
    data = diabetes_load_data()
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
    print("\nارزیابی دیابت:")
    print("دقت:", accuracy_score(y_test, y_pred))
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': FEATURE_NAMES,
        'performance': {
            'accuracy': accuracy_score(y_test, y_pred)
        }
    }
    delete_old_models(MODEL_PREFIXES['2'])
    joblib.dump(model_data, f"{MODEL_PREFIXES['2']}{timestamp}.pkl")
    return model_data

# -------------------- ماژول بیماری قلبی --------------------
def heart_load_data():
    try:
        df = pd.read_csv(MODEL_PATHS['3'], na_values=['?', ' ', 'NA', ''])
    except:
        df = pd.read_csv(MODEL_PATHS['3'], names=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ], na_values=['?', ' ', 'NA', ''])
    
    df = df.dropna()
    df['target'] = pd.to_numeric(df['target'], errors='coerce').fillna(0)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    return df

def heart_train():
    df = heart_load_data()
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
    print("\nارزیابی بیماری قلبی:")
    print("دقت:", accuracy_score(y_test, y_pred))
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': list(X.columns),
        'performance': {
            'accuracy': accuracy_score(y_test, y_pred)
        }
    }
    delete_old_models(MODEL_PREFIXES['3'])
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
        
        return proba
    except Exception as e:
        return {'error': str(e)}

# -------------------- توابع دریافت ورودی --------------------
def get_breast_input(feature_names):
    print("\nلطفاً مقادیر ویژگی‌های سرطان پستان را وارد کنید:")
    return {feature: float(input(f"{feature}: ")) for feature in feature_names if feature not in ['id', 'diagnosis']}

def get_diabetes_input():
    print("\nلطفاً اطلاعات دیابت را وارد کنید:")
    return {
        'Pregnancies': float(input("تعداد بارداری‌ها: ")),
        'Glucose': float(input("گلوکز (mg/dl): ")),
        'BloodPressure': float(input("فشار خون (mmHg): ")),
        'SkinThickness': float(input("ضخامت پوست (mm): ")),
        'Insulin': float(input("انسولین (muU/ml): ")),
        'BMI': float(input("شاخص توده بدنی: ")),
        'DiabetesPedigreeFunction': float(input("سابقه خانوادگی: ")),
        'Age': float(input("سن (سال): "))
    }

def get_heart_input():
    print("\nلطفاً اطلاعات قلبی را وارد کنید:")
    return {
        'age': float(input("سن (سال): ")),
        'sex': int(input("جنسیت (0: زن، 1: مرد): ")),
        'cp': int(input("نوع درد قفسه سینه (0-3): ")),
        'trestbps': float(input("فشار خون (mmHg): ")),
        'chol': float(input("کلسترول (mg/dl): ")),
        'fbs': int(input("قند خون ناشتا (0: ≤120, 1: >120): ")),
        'restecg': int(input("نتیجه ECG استراحت (0-2): ")),
        'thalach': float(input("حداکثر ضربان قلب: ")),
        'exang': int(input("آنژین القا شده (0: خیر، 1: بله): ")),
        'oldpeak': float(input("افسردگی ST: ")),
        'slope': int(input("شیب ST (0-2): ")),
        'ca': int(input("تعداد عروق اصلی (0-3): ")),
        'thal': int(input("نتیجه تالاسیمی (0-3): "))
    }

# -------------------- اجرای اصلی --------------------
if __name__ == "__main__":
    while True:
        choice = show_menu()
        
        if choice == '4':
            print("خروج از برنامه...")
            break
            
        if choice not in ['1', '2', '3']:
            print("⚠️ انتخاب نامعتبر!")
            continue
            
        # آموزش یا بارگذاری مدل
        model_prefix = MODEL_PREFIXES[choice]
        model_files = [f for f in os.listdir() if f.startswith(model_prefix)]
        
        if not model_files:
            print("آموزش مدل جدید...")
            if choice == '1':
                model_data = breast_cancer_train()
            elif choice == '2':
                model_data = diabetes_train()
            elif choice == '3':
                model_data = heart_train()
        else:
            print("استفاده از مدل موجود")
            latest_model = sorted(model_files)[-1]
            model_data = joblib.load(latest_model)
        
        # دریافت ورودی و پیش‌بینی
        try:
            if choice == '1':
                df, features = breast_cancer_load_data()
                user_input = get_breast_input(features)
            elif choice == '2':
                user_input = get_diabetes_input()
                features = model_data['features']
            elif choice == '3':
                user_input = get_heart_input()
                features = model_data['features']
            
            result = predict(model_prefix, user_input, features)
            
            if isinstance(result, dict) and 'error' in result:
                print(f"\n❌ خطا: {result['error']}")
            else:
                proba = result
                if choice == '1':
                    diagnosis = 'خوش‌خیم' if proba >= 0.5 else 'بدخیم'
                elif choice == '2':
                    diagnosis = '✅ مثبت' if proba >= 0.5 else '❌ منفی'
                elif choice == '3':
                    diagnosis = 'در معرض خطر' if proba >= 0.5 else 'سالم'
                
                print("\n✅ نتایج تشخیص:")
                print(f"- تشخیص: {diagnosis}")
                print(f"- احتمال: {proba*100:.1f}%")
                print(f"- آستانه تشخیص: 0.5")
                
        except Exception as e:
            print(f"\n❌ خطا در پردازش: {str(e)}")