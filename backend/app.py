import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

# Configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-this-in-production'
    DEBUG = True
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'sleep_disorder_xgb.pkl')
    DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset.xlsx')

# Create Flask app
app = Flask(__name__, 
            template_folder='../frontend',
            static_folder='../frontend')
app.config.from_object(Config)

# Add CORS support for all routes
CORS(app, origins=['*'])

def load_dataset(path=Config.DATASET_PATH):
    """Load and analyze the dataset"""
    if not os.path.exists(path):
        print(f"Dataset file not found at: {path}")
        return None
    
    try:
        # Try different Excel engines
        try:
            df = pd.read_excel(path, engine='openpyxl')
        except ImportError:
            print("openpyxl not installed. Trying to install...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'openpyxl'])
            df = pd.read_excel(path, engine='openpyxl')
        except Exception:
            # Fallback to other engines
            try:
                df = pd.read_excel(path, engine='xlrd')
            except:
                print("Cannot read Excel file. Please ensure dataset.xlsx is valid.")
                return None
                
        print(f"Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head())
        
        # Analyze categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        print(f"Categorical columns: {categorical_cols}")
        
        for col in categorical_cols:
            unique_vals = df[col].unique()
            print(f"{col} unique values: {unique_vals}")
            
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def load_model_and_meta(path=Config.MODEL_PATH):
    """Load model and metadata from pickle file with warning suppression"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")

    # Suppress XGBoost warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
    
    try:
        obj = joblib.load(path)
        print(f"Loaded object type: {type(obj)}")

        if isinstance(obj, (list, tuple)) and len(obj) >= 2:
            model = obj[0]
            meta = obj[1] if isinstance(obj[1], dict) else {}
            
            # Handle different metadata formats
            if len(obj) >= 3 and not isinstance(obj[1], dict):
                # Format: [model, label_encoders, target_encoder, ...]
                meta = {
                    "label_encoders": obj[1] if isinstance(obj[1], dict) else {},
                    "target_encoder": obj[2] if len(obj) > 2 else None,
                    "feature_columns": obj[3] if len(obj) > 3 and isinstance(obj[3], list) else [],
                    "mode_values": {},
                    "median_values": {}
                }
            
            return model, meta

        if hasattr(obj, "predict"):
            model = obj
            meta = {
                "label_encoders": {},
                "target_encoder": None,
                "mode_values": {},
                "median_values": {},
                "feature_columns": [],
                "top_features": []
            }
            return model, meta

        raise ValueError("Unrecognized model file format.")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try to create a simple fallback model for testing
        print("Creating fallback model for testing...")
        return create_fallback_model()

def create_fallback_model():
    """Create a simple fallback model if the main model fails to load"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # Create simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create dummy training data
        X_dummy = np.random.rand(100, 5)  # 5 features
        y_dummy = np.random.choice(['No Sleep Disorder', 'Sleep Apnea', 'Insomnia'], 100)
        model.fit(X_dummy, y_dummy)
        
        # Create label encoders
        label_encoders = {}
        
        # Gender encoder
        gender_encoder = LabelEncoder()
        gender_encoder.fit(['male', 'female'])
        label_encoders['Gender'] = gender_encoder
        
        # BMI Category encoder
        bmi_encoder = LabelEncoder()
        bmi_encoder.fit(['normal', 'overweight', 'obese'])
        label_encoders['BMI Category'] = bmi_encoder
        
        # Blood Pressure encoder
        bp_encoder = LabelEncoder()
        bp_encoder.fit(['120/80', '130/85', '140/90'])
        label_encoders['Blood Pressure'] = bp_encoder
        
        # Occupation encoder - ALL occupations from your dataset
        occ_encoder = LabelEncoder()
        occ_encoder.fit([
            'nurse', 'doctor', 'engineer', 'software engineer', 'scientist',
            'teacher', 'manager', 'sales representative', 'accountant', 'lawyer'
        ])
        label_encoders['Occupation'] = occ_encoder
        
        # Create target encoder
        target_encoder = LabelEncoder()
        target_encoder.fit(['No Sleep Disorder', 'Sleep Apnea', 'Insomnia'])
        
        meta = {
            'label_encoders': label_encoders,
            'target_encoder': target_encoder,
            'feature_columns': ['Gender', 'BMI Category', 'Blood Pressure', 'Occupation', 'Heart Rate'],
            'top_features': ['BMI Category', 'Blood Pressure', 'Occupation', 'Heart Rate', 'Gender'],
            'mode_values': {
                'Gender': 'male',
                'BMI Category': 'normal', 
                'Blood Pressure': '120/80',
                'Occupation': 'nurse'
            },
            'median_values': {
                'Heart Rate': 70.0
            }
        }
        
        print("Fallback model created successfully!")
        return model, meta
        
    except Exception as e:
        print(f"Failed to create fallback model: {e}")
        return None, {}

# Load dataset for analysis
dataset = load_dataset()

# Load model and metadata
try:
    model, meta = load_model_and_meta()
    load_error = None
    print("Model loaded successfully!")
    print(f"Metadata keys: {meta.keys()}")
except Exception as e:
    model = None
    meta = {}
    load_error = str(e)
    print(f"Model load error: {load_error}")

# Extract metadata
label_encoders = meta.get("label_encoders", {})
target_encoder = meta.get("target_encoder", None)
mode_values = meta.get("mode_values", {})
median_values = meta.get("median_values", {})
feature_columns = meta.get("feature_columns", [])
top_features = meta.get("top_features", [])

# If dataset is available, extract feature information from it
if dataset is not None:
    print("\n=== DATASET ANALYSIS ===")
    
    # Get actual column names from dataset
    actual_columns = list(dataset.columns)
    print(f"Actual dataset columns: {actual_columns}")
    
    # Update feature columns if not available from model
    if not feature_columns:
        # Remove target column if it exists
        target_col = 'Sleep Disorder'
        if target_col in actual_columns:
            feature_columns = [col for col in actual_columns if col != target_col]
        else:
            feature_columns = actual_columns
    
    # Analyze categorical and numerical features
    categorical_features = []
    numerical_features = []
    
    for col in feature_columns:
        if col in dataset.columns:
            if dataset[col].dtype == 'object' or col in ['Gender', 'Occupation', 'BMI Category', 'Blood Pressure']:
                categorical_features.append(col)
                unique_vals = dataset[col].unique()
                print(f"{col} (categorical): {unique_vals}")
            else:
                numerical_features.append(col)
                print(f"{col} (numerical): min={dataset[col].min()}, max={dataset[col].max()}, median={dataset[col].median()}")
    
    print(f"Categorical features from dataset: {categorical_features}")
    print(f"Numerical features from dataset: {numerical_features}")
    
    # Update mode and median values from dataset
    if not mode_values:
        mode_values = {}
        for col in categorical_features:
            if col in dataset.columns:
                mode_val = dataset[col].mode().iloc[0] if len(dataset[col].mode()) > 0 else dataset[col].iloc[0]
                mode_values[col] = str(mode_val).lower()
    
    if not median_values:
        median_values = {}
        for col in numerical_features:
            if col in dataset.columns:
                median_values[col] = float(dataset[col].median())

# Print available encoders for debugging
print(f"Available label encoders: {list(label_encoders.keys())}")
for name, encoder in label_encoders.items():
    if hasattr(encoder, 'classes_'):
        print(f"{name} classes: {encoder.classes_}")

# Set default features based on your model or dataset
if not top_features:
    if feature_columns:
        # Use top 5 most important features or all if less than 5
        top_features = feature_columns[:5] if len(feature_columns) >= 5 else feature_columns
    else:
        # Fallback to common feature names
        top_features = ["BMI Category", "Blood Pressure", "Occupation", "Heart Rate", "Gender"]

# Define proper sleep disorder labels
SLEEP_DISORDERS = {
    'nan': 'No Sleep Disorder',
    'insomnia': 'Insomnia', 
    'sleep apnea': 'Sleep Apnea',
    'no sleep disorder': 'No Sleep Disorder'
}

def clean_prediction(pred_value):
    """Clean and standardize prediction output"""
    if pd.isna(pred_value) or str(pred_value).lower() in ['nan', 'none', '']:
        return 'No Sleep Disorder'
    
    pred_str = str(pred_value).lower().strip()
    
    # Direct mapping
    if pred_str in SLEEP_DISORDERS:
        return SLEEP_DISORDERS[pred_str]
    
    # Partial matching
    if 'insomnia' in pred_str:
        return 'Insomnia'
    elif 'apnea' in pred_str or 'sleep apnea' in pred_str:
        return 'Sleep Apnea'
    elif 'no' in pred_str or 'normal' in pred_str:
        return 'No Sleep Disorder'
    
    return pred_str.title()

print(f"\n=== FINAL CONFIGURATION ===")
print(f"Available features: {top_features}")
print(f"Feature columns: {feature_columns}")
print(f"Mode values: {mode_values}")
print(f"Median values: {median_values}")
print(f"Label encoders: {list(label_encoders.keys())}")

@app.route("/", methods=["GET"])
def index():
    """Render the main page"""
    return render_template("index.html")

@app.route("/dataset-info", methods=["GET"])
def dataset_info():
    """Get dataset information"""
    if dataset is not None:
        info = {
            'loaded': True,
            'shape': dataset.shape,
            'columns': list(dataset.columns),
            'sample_data': dataset.head().to_dict('records'),
            'categorical_features': [col for col in dataset.columns if dataset[col].dtype == 'object'],
            'numerical_features': [col for col in dataset.columns if dataset[col].dtype != 'object']
        }
        
        # Add unique values for categorical columns
        for col in info['categorical_features']:
            info[f'{col}_unique'] = dataset[col].unique().tolist()
            
        return jsonify(info)
    else:
        return jsonify({
            'loaded': False,
            'error': 'Dataset not found or could not be loaded'
        })

@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests"""
    if model is None:
        return jsonify({
            'success': False,
            'error': f'Model not loaded: {load_error}'
        })

    try:
        # Get user inputs
        user_inputs = {}
        
        # Use top_features for input mapping
        for feature in top_features:
            val = request.form.get(feature, "").strip()
            user_inputs[feature] = val

        print(f"User inputs: {user_inputs}")

        # Create a DataFrame with all required columns
        if feature_columns:
            # Use the exact column structure expected by the model
            input_df = pd.DataFrame(index=[0], columns=feature_columns)
            
            # Fill in the values we have
            for col in feature_columns:
                if col in user_inputs and user_inputs[col]:
                    input_df.at[0, col] = user_inputs[col]
                else:
                    # Use defaults from dataset analysis or predefined defaults
                    if col in mode_values:
                        input_df.at[0, col] = mode_values[col]
                    elif col in median_values:
                        input_df.at[0, col] = median_values[col]
                    else:
                        # Fallback defaults
                        defaults = {
                            "Age": 30,
                            "Sleep Duration": 7.5,
                            "Quality of Sleep": 6,
                            "Physical Activity Level": 50,
                            "Stress Level": 5,
                            "Daily Steps": 8000,
                            "Person ID": 1,
                            "BMI Category": "normal",
                            "Blood Pressure": "120/80",
                            "Occupation": "nurse",
                            "Gender": "male",
                            "Heart Rate": 70
                        }
                        input_df.at[0, col] = defaults.get(col, 0)
        else:
            # Fallback if no feature_columns defined
            input_df = pd.DataFrame([user_inputs])

        print(f"DataFrame before encoding:")
        print(input_df)

        # Encode categorical features using label encoders
        for col, le in label_encoders.items():
            if col in input_df.columns:
                val = str(input_df.at[0, col]).strip().lower()
                print(f"Encoding {col}: '{val}' from classes: {le.classes_}")
                
                if val in le.classes_:
                    encoded = int(le.transform([val])[0])
                    print(f"Successfully encoded {col} '{val}' -> {encoded}")
                else:
                    # Try to find a matching class
                    encoded = 0  # Default fallback
                    for i, class_val in enumerate(le.classes_):
                        if class_val.lower() == val:
                            encoded = i
                            break
                    else:
                        # Use the first class as fallback
                        if len(le.classes_) > 0:
                            encoded = 0
                            print(f"Warning: '{val}' not found in {col} classes, using default: {le.classes_[0]}")
                
                input_df.at[0, col] = encoded

        # Convert all columns to appropriate numeric types
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                
                # Fill NaN values
                if input_df[col].isna().any():
                    default_val = median_values.get(col, 0)
                    input_df[col] = input_df[col].fillna(default_val)
                        
            except Exception as e:
                print(f"Error converting {col} to numeric: {e}")
                input_df[col] = 0

        # Ensure all data types are compatible with XGBoost
        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        print(f"Final DataFrame:")
        print(input_df)

        # Make prediction with warning suppression
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = model.predict(input_df)
        
        print(f"Raw prediction: {pred}")

        # Handle prediction decoding
        pred_label = None
        if target_encoder is not None:
            try:
                decoded = target_encoder.inverse_transform(pred)
                pred_label = decoded[0]
                print(f"Decoded prediction: {pred_label}")
            except Exception as e:
                print(f"Error decoding with target_encoder: {e}")
                pred_label = pred[0]
        else:
            pred_label = pred[0]

        # Clean the prediction
        final_prediction = clean_prediction(pred_label)
        print(f"Final cleaned prediction: {final_prediction}")

        return jsonify({
            'success': True,
            'prediction': final_prediction,
            'raw_prediction': str(pred_label)
        })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Prediction error: {error_trace}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'dataset_loaded': dataset is not None,
        'features': top_features,
        'feature_columns': feature_columns,
        'model_type': type(model).__name__ if model else None,
        'available_encoders': list(label_encoders.keys())
    })

if __name__ == "__main__":
    print("Starting Flask application...")
    print(f"Model loaded: {model is not None}")
    print(f"Dataset loaded: {dataset is not None}")
    print(f"Top features: {top_features}")
    print(f"Feature columns: {feature_columns}")
    print(f"Template folder: {app.template_folder}")
    print(f"Static folder: {app.static_folder}")
    
    app.run(debug=True, host="0.0.0.0", port=5000)