from flask import Flask, render_template, request, redirect, url_for
from diabetes_heart_breast import *
import os
import joblib

app = Flask(__name__)
app.secret_key = 'supersecretkey'

MODEL_PREFIXES = {
    '1': 'breast_model_',
    '2': 'diabetes_model_',
    '3': 'heart_model_'
}

MODEL_NAMES = {
    '1': 'سرطان پستان',
    '2': 'دیابت',
    '3': 'بیماری قلبی'
}

FEATURE_TRANSLATIONS = {
    # سرطان پستان
    'radius_mean': {'fa_name': 'میانگین شعاع سلول', 'range': '6-28 میکرومتر', 'example': 12.34},
    'texture_mean': {'fa_name': 'میانگین بافت سلول', 'range': '9-39', 'example': 18.5},
    'perimeter_mean': {'fa_name': 'میانگین محیط سلول', 'range': '43-188 میکرومتر', 'example': 85.2},
    'area_mean': {'fa_name': 'میانگین مساحت سلول', 'range': '143-2501 میکرومتر مربع', 'example': 550.1},
    'smoothness_mean': {'fa_name': 'میانگین صافی سطح', 'range': '0.05-0.16', 'example': 0.095},
    'compactness_mean': {'fa_name': 'میانگین تراکم سلول', 'range': '0.02-0.35', 'example': 0.12},
    'concavity_mean': {'fa_name': 'میانگین تقعر سطح', 'range': '0.0-0.45', 'example': 0.15},
    'concave points_mean': {'fa_name': 'میانگین نقاط مقعر', 'range': '0.0-0.21', 'example': 0.08},
    'symmetry_mean': {'fa_name': 'میانگین تقارن', 'range': '0.11-0.31', 'example': 0.18},
    'fractal_dimension_mean': {'fa_name': 'بعد فراکتالی', 'range': '0.05-0.1', 'example': 0.065},

    # دیابت
    'Pregnancies': {'fa_name': 'تعداد بارداری', 'range': '0-17', 'example': 2},
    'Glucose': {'fa_name': 'سطح گلوکز خون', 'range': '0-199 mg/dL', 'example': 120},
    'BloodPressure': {'fa_name': 'فشار خون', 'range': '0-122 mmHg', 'example': 72},
    'SkinThickness': {'fa_name': 'ضخامت پوست', 'range': '0-99 mm', 'example': 23},
    'Insulin': {'fa_name': 'سطح انسولین', 'range': '0-846 μU/mL', 'example': 80},
    'BMI': {'fa_name': 'شاخص توده بدنی', 'range': '0-67 kg/m²', 'example': 26.5},
    'DiabetesPedigreeFunction': {'fa_name': 'سابقه خانوادگی دیابت', 'range': '0.08-2.42', 'example': 0.45},
    'Age': {'fa_name': 'سن', 'range': '21-81 سال', 'example': 35},

    # بیماری قلبی
    'age': {'fa_name': 'سن', 'range': '29-77 سال', 'example': 45},
    'sex': {'fa_name': 'جنسیت', 'range': '0:زن، 1:مرد', 'example': 1},
    'cp': {'fa_name': 'نوع درد قفسه سینه', 'range': '0-3', 'example': 2},
    'trestbps': {'fa_name': 'فشار خون استراحت', 'range': '94-200 mmHg', 'example': 120},
    'chol': {'fa_name': 'کلسترول خون', 'range': '126-564 mg/dL', 'example': 240},
    'fbs': {'fa_name': 'قند خون ناشتا', 'range': '0:≤120، 1:>120', 'example': 0},
    'restecg': {'fa_name': 'نتیجه ECG استراحت', 'range': '0-2', 'example': 1},
    'thalach': {'fa_name': 'حداکثر ضربان قلب', 'range': '71-202 bpm', 'example': 150},
    'exang': {'fa_name': 'آنژین القا شده', 'range': '0:خیر، 1:بله', 'example': 0},
    'oldpeak': {'fa_name': 'افت ST', 'range': '0-6.2', 'example': 1.2},
    'slope': {'fa_name': 'شیب ST', 'range': '0-2', 'example': 1},
    'ca': {'fa_name': 'تعداد عروق اصلی', 'range': '0-3', 'example': 0},
    'thal': {'fa_name': 'نتیجه تالاسیمی', 'range': '0-3', 'example': 2},

    # ویژگی‌های خطای استاندارد (SE)
    'radius_se': {'fa_name': 'خطای استاندارد شعاع سلول', 'range': '0.1-2.8 میکرومتر', 'example': 0.5},
    'texture_se': {'fa_name': 'خطای استاندارد بافت سلول', 'range': '0.4-4.8', 'example': 1.2},
    'perimeter_se': {'fa_name': 'خطای استاندارد محیط سلول', 'range': '0.8-21.9 میکرومتر', 'example': 2.3},
    'area_se': {'fa_name': 'خطای استاندارد مساحت سلول', 'range': '6-542 μm²', 'example': 45.6},
    'smoothness_se': {'fa_name': 'خطای استاندارد صافی سطح', 'range': '0.001-0.031', 'example': 0.007},
    'compactness_se': {'fa_name': 'خطای استاندارد تراکم سلول', 'range': '0.002-0.135', 'example': 0.025},
    'concavity_se': {'fa_name': 'خطای استاندارد تقعر سطح', 'range': '0.0-0.396', 'example': 0.045},
    'concave points_se': {'fa_name': 'خطای استاندارد نقاط مقعر', 'range': '0.0-0.053', 'example': 0.015},
    'symmetry_se': {'fa_name': 'خطای استاندارد تقارن', 'range': '0.008-0.079', 'example': 0.022},
    'fractal_dimension_se': {'fa_name': 'خطای استاندارد بعد فراکتالی', 'range': '0.001-0.03', 'example': 0.005},
    
    # ویژگی‌های اضافی سرطان پستان
    'radius_worst': {'fa_name': 'بدترین شعاع سلول', 'range': '7-36 میکرومتر', 'example': 14.5},
    'texture_worst': {'fa_name': 'بدترین بافت سلول', 'range': '12-49', 'example': 22.3},
    'perimeter_worst': {'fa_name': 'بدترین محیط سلول', 'range': '50-251 میکرومتر', 'example': 95.7},
    'area_worst': {'fa_name': 'بدترین مساحت سلول', 'range': '185-4250 μm²', 'example': 680.4},
    'smoothness_worst': {'fa_name': 'بدترین صافی سطح', 'range': '0.07-0.22', 'example': 0.13},
    'compactness_worst': {'fa_name': 'بدترین تراکم سلول', 'range': '0.03-1.06', 'example': 0.25},
    'concavity_worst': {'fa_name': 'بدترین تقعر سطح', 'range': '0.0-1.25', 'example': 0.35},
    'concave points_worst': {'fa_name': 'بدترین نقاط مقعر', 'range': '0.0-0.29', 'example': 0.12},
    'symmetry_worst': {'fa_name': 'بدترین تقارن', 'range': '0.16-0.66', 'example': 0.28},
    'fractal_dimension_worst': {'fa_name': 'بدترین بعد فراکتالی', 'range': '0.06-0.21', 'example': 0.09}
}

@app.context_processor
def inject_globals():
    return {
        'MODEL_NAMES': MODEL_NAMES,
        'FEATURE_TRANSLATIONS': FEATURE_TRANSLATIONS
    }

def get_model_status():
    models = {}
    for model_id, prefix in MODEL_PREFIXES.items():
        model_files = sorted([f for f in os.listdir() if f.startswith(prefix)])
        if model_files:
            latest = model_files[-1]
            timestamp = latest[len(prefix):-4]
            models[model_id] = {
                'name': MODEL_NAMES[model_id],
                'timestamp': timestamp,
                'path': latest
            }
        else:
            models[model_id] = None
    return models

@app.route('/')
def home():
    return render_template('train.html', models=get_model_status())

@app.route('/train', methods=['POST'])
def train_model():
    model_id = request.form['model_id']
    try:
        if model_id == '1':
            breast_cancer_train()
        elif model_id == '2':
            diabetes_train()
        elif model_id == '3':
            heart_train()
        return redirect(url_for('home'))
    except FileNotFoundError as e:
        return f"خطا: فایل دیتاست یافت نشد!<br>{str(e)}", 404
    except Exception as e:
        return f"خطای سرور: {str(e)}", 500

@app.route('/get-fields')
def get_fields():
    model_id = request.args.get('model_id')
    model_prefix = MODEL_PREFIXES[model_id]
    
    try:
        model_files = [f for f in os.listdir() if f.startswith(model_prefix)]
        if not model_files:
            return "مدلی برای این بیماری آموزش داده نشده است", 404
            
        latest_model = sorted(model_files)[-1]
        model_data = joblib.load(latest_model)
        
        # بررسی وجود کلید features در مدل
        if 'features' not in model_data:
            return "خطا در ساختار مدل ذخیره شده", 500
            
        return render_template(
            'fields.html', 
            features=model_data['features'],
            model_id=model_id
        )
        
    except Exception as e:
        return f"خطای سرور: {str(e)}", 500

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        model_id = request.form['model_type']
        model_prefix = MODEL_PREFIXES[model_id]
        
        try:
            model_files = [f for f in os.listdir() if f.startswith(model_prefix)]
            if not model_files:
                return "مدل آموزش داده نشده است", 400
                
            latest_model = sorted(model_files)[-1]
            model_data = joblib.load(latest_model)
            
            input_data = {}
            for feature in model_data['features']:
                input_data[feature] = float(request.form[feature])
            
            scaled_input = model_data['scaler'].transform(pd.DataFrame([input_data]))
            proba = model_data['model'].predict_proba(scaled_input)[0][1]
            
            diagnosis = ''
            if model_id == '1':
                diagnosis = 'خوش‌خیم' if proba >= 0.5 else 'بدخیم'
            elif model_id == '2':
                diagnosis = 'مثبت' if proba >= 0.5 else 'منفی'
            elif model_id == '3':
                diagnosis = 'در معرض خطر' if proba >= 0.5 else 'سالم'
            
            return render_template('predict.html', 
                            result={
                                'diagnosis': f"تشخیص: {diagnosis} (احتمال: {proba*100:.1f}%)",
                                'inputs': input_data
                            },
                            models=get_model_status())
        
        except Exception as e:
            return str(e), 500
    
    return render_template('predict.html', models=get_model_status())

if __name__ == '__main__':
    app.run(debug=True)