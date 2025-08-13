import fitz  # PyMuPDF
import os
import re
# Disable oneDNN optimizations to prevent startup hangs on some CPUs
# This is now handled in run.py, but leaving it commented here for history.
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import joblib
import shap
import numpy as np
from tensorflow.keras.models import load_model

# Paths relative to this file to avoid CWD issues
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, 'models')

# Optional decision threshold for classifying as Malicious (0-1).
# If DECISION_THRESHOLD is not set (or set to 'auto'), we use the model's own decision (argmax/threshold).
_DT_ENV = os.environ.get('DECISION_THRESHOLD', '').strip()
DECISION_THRESHOLD = None if _DT_ENV in ('', 'auto', None) else float(_DT_ENV)

# Scenario-specific decision thresholds to reduce false positives
SCENARIO_THRESHOLDS = {
    'A': 0.5,  # Default threshold for scenario A
    'B': 0.5,  # Default threshold for scenario B  
    'C': 0.99, # EXTREME CONSERVATIVE threshold - hanya 99%+ yang malicious
    'D': 0.5,  # Default threshold for scenario D
    'E': 0.5,  # Default threshold for scenario E
}

def _resolve_path(candidate_path: str, default_dir: str) -> str:
    """Resolve a model/pipeline path robustly regardless of current working directory.

    Tries multiple fallbacks:
    - Absolute path as-is
    - Relative path from current CWD
    - Join with the given default_dir using the basename
    - If the path starts with 'app/', strip it and join with APP_DIR
    """
    # If already absolute and exists
    if os.path.isabs(candidate_path) and os.path.exists(candidate_path):
        return candidate_path

    # Try as given relative to current CWD
    if os.path.exists(candidate_path):
        return candidate_path

    # Try by stripping a leading 'app/' and joining with APP_DIR
    normalized = candidate_path.replace('\\', '/')
    if normalized.startswith('app/'):
        stripped = normalized[len('app/'):]
        joined = os.path.join(APP_DIR, stripped)
        if os.path.exists(joined):
            return joined

    # Finally, try inside default_dir with basename
    fallback = os.path.join(default_dir, os.path.basename(candidate_path))
    return fallback

# --- Model & Pipeline Configuration ---
# All paths point to the 'app/models/' directory.
MODEL_PATHS = {
    # If you prefer a unified scikit-learn pipeline (preprocessing + model),
    # save it to the 'pipeline' path and omit the 'model' file for that scenario.
    # The loader will detect it automatically (must provide predict_proba/predict).
    'A': {'model': 'app/models/model_A_MLP_static.h5', 'pipeline': 'app/models/pipeline_A_static.pkl'},
    # Scenario B: shares the same artifacts as A but uses hybrid features (static + dynamic)
    'B': {'model': 'app/models/model_A_MLP_static.h5', 'pipeline': 'app/models/pipeline_A_static.pkl'},
    'C': {'model': 'app/models/model_C_MLP_static_dynamic.h5', 'pipeline': 'app/models/pipeline_C_static_dynamic.pkl'},
    'D': {'model': 'app/models/model_D_Stacking_static.pkl', 'pipeline': 'app/models/pipeline_D_static.pkl'},
    'E': {'model': 'app/models/model_E_Stacking_static_dynamic.pkl', 'pipeline': 'app/models/pipeline_E_static_dynamic.pkl'},
}

# Lazy-loading cache for models and pipelines
LOADED_MODELS = {}

# Cache for SHAP explainers to avoid recomputation
SHAP_EXPLAINERS = {}

# --- Feature Definitions ---
STATIC_FEATURES = [
    'PdfSize', 'MetadataSize', 'Pages', 'XrefLength', 'TitleCharacters', 
    'isEncrypted', 'EmbeddedFiles', 'Images', 'Text', 'Header', 'Obj', 
    'Endobj', 'Stream', 'Endstream', 'Xref', 'Trailer', 'StartXref', 'PageNo', 
    'Encrypt', 'ObjStm', 'JS', 'Javascript', 'AA', 'OpenAction', 'Acroform', 
    'JBIG2Decode', 'RichMedia', 'Launch', 'EmbeddedFile', 'XFA', 'Colors'
]
DYNAMIC_FEATURES = [
    'JSExec', 'AutoExec', 'FileDrop', 'ShellExec', 'ExploitAttempt'
]

def get_model_and_pipeline(scenario):
    """
    Lazily loads and caches the model and pipeline for a given scenario.
    Handles both .pkl and .h5 model file formats.
    If a previous attempt cached (None, None), it will retry loading.
    """
    # If not in cache, or previously failed, attempt to (re)load
    if scenario not in LOADED_MODELS or LOADED_MODELS.get(scenario) == (None, None):
        paths = MODEL_PATHS.get(scenario)
        if not paths:
            raise ValueError(f"Invalid scenario: {scenario}. No model paths defined.")
        
        try:
            model_path = _resolve_path(paths['model'], MODELS_DIR) if 'model' in paths else None
            pipeline_path = _resolve_path(paths['pipeline'], MODELS_DIR) if 'pipeline' in paths else None

            # Load preprocessing pipeline first (required in all current setups)
            if not pipeline_path or not os.path.exists(pipeline_path):
                raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
            try:
                pipeline = joblib.load(pipeline_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load pipeline from {pipeline_path}: {e}")

            model = None

            # Load model if present
            if model_path and os.path.exists(model_path):
                if model_path.endswith('.h5'):
                    try:
                        model = load_model(model_path)
                    except Exception as e:
                        raise RuntimeError(f"Failed to load Keras model from {model_path}: {e}")
                elif model_path.endswith('.pkl'):
                    try:
                        model = joblib.load(model_path)
                    except Exception as e:
                        raise RuntimeError(f"Failed to load scikit-learn model from {model_path}: {e}")
                else:
                    raise ValueError(f"Unsupported model format for {model_path}. Only .h5 and .pkl are supported.")
            else:
                # No separate model file. If the pipeline can predict, treat it as a unified pipeline.
                if hasattr(pipeline, 'predict_proba') or hasattr(pipeline, 'predict'):
                    print(f"Detected unified pipeline for scenario {scenario} (no separate model file).")
                    model = None  # unified: model encapsulated in pipeline
                else:
                    raise FileNotFoundError(
                        f"Model file not found: {model_path}. The pipeline does not provide prediction methods; cannot use unified pipeline.")

            LOADED_MODELS[scenario] = (model, pipeline)
            print(f"Successfully loaded resources for scenario {scenario}")

        except (FileNotFoundError, RuntimeError, ValueError) as e:
            print(f"Error loading model/pipeline for scenario {scenario}: {e}")
            LOADED_MODELS[scenario] = (None, None)
        except Exception as e:
            print(f"Unexpected error loading model/pipeline for scenario {scenario}: {e}")
            LOADED_MODELS[scenario] = (None, None)
            
    return LOADED_MODELS[scenario]

def extract_features_from_pdf(file_path):
    doc = fitz.open(file_path)
    metadata = doc.metadata

    # Basic info
    pdf_size = os.path.getsize(file_path)
    num_pages = doc.page_count
    xref_length = doc.xref_length()
    is_encrypted = int(doc.is_encrypted)
    page_no = num_pages

    metadata_str = str(metadata)
    metadata_size = len(metadata_str.encode('utf-8'))

    title = metadata.get('title', '') or ''
    title_characters = len(title)

    text_found = 0
    for page in doc:
        if page.get_text().strip():
            text_found = 1
            break

    with open(file_path, 'rb') as f:
        header_line = f.readline().decode(errors='ignore').strip()
    header_version = header_line.replace('%PDF-', '') if '%PDF-' in header_line else ''

    keywords = [
        "Obj", "Endobj", "Stream", "Endstream", "Xref", "Trailer", "StartXref",
        "ObjStm", "JS", "Javascript", "AA", "OpenAction", "Acroform", "JBIG2Decode",
        "RichMedia", "Launch", "EmbeddedFile", "XFA", "Colors"
    ]
    keyword_counts = {key:0 for key in keywords}

    embedded_files_count = 0
    images_count = 0

    # Fitur dinamis
    js_exec = 0
    auto_exec = 0
    file_drop = 0
    shell_exec = 0
    exploit_attempt = 0

    # Pola indikatif untuk fitur dinamis
    dynamic_patterns = {
        'JSExec': [r'app\.launchURL', r'app\.execMenuItem', r'app\.execCommand'],
        'AutoExec': [r'/AA', r'/OpenAction', r'/Launch'],
        'FileDrop': [r'EmbeddedFile', r'launch'],
        'ShellExec': [r'cmd\.exe', r'PowerShell', r'cmd /c', r'cmd /k'],
        'ExploitAttempt': [r'JBIG2Decode', r'/RichMedia', r'/XFA', r'/Launch']
    }

    for xref in range(1, xref_length):
        try:
            obj = doc.xref_object(xref, compressed=False)
            obj_lower = obj.lower()

            # Static keyword count
            for key in keywords:
                pattern = re.compile(r'\b' + key.lower() + r'\b')
                matches = pattern.findall(obj_lower)
                keyword_counts[key] += len(matches)

            if "/javascript" in obj_lower or "/js" in obj_lower:
                keyword_counts["JS"] += 1
                keyword_counts["Javascript"] += 1
            if "/embeddedfile" in obj_lower:
                embedded_files_count += 1

            # Dynamic feature pattern matching
            for feat, patterns in dynamic_patterns.items():
                for pat in patterns:
                    if re.search(pat, obj, re.IGNORECASE):
                        if feat == 'JSExec': js_exec = 1
                        elif feat == 'AutoExec': auto_exec = 1
                        elif feat == 'FileDrop': file_drop = 1
                        elif feat == 'ShellExec': shell_exec = 1
                        elif feat == 'ExploitAttempt': exploit_attempt = 1
        except Exception:
            continue

    for page in doc:
        images_count += len(page.get_images())

    features = {
        'PdfSize': pdf_size,
        'MetadataSize': metadata_size,
        'Pages': num_pages,
        'XrefLength': xref_length,
        'TitleCharacters': title_characters,
        'isEncrypted': is_encrypted,
        'EmbeddedFiles': embedded_files_count,
        'Images': images_count,
        'Text': text_found,
        'Header': header_version,
        'Obj': keyword_counts["Obj"],
        'Endobj': keyword_counts["Endobj"],
        'Stream': keyword_counts["Stream"],
        'Endstream': keyword_counts["Endstream"],
        'Xref': keyword_counts["Xref"],
        'Trailer': keyword_counts["Trailer"],
        'StartXref': keyword_counts["StartXref"],
        'PageNo': page_no,
        'ObjStm': keyword_counts["ObjStm"],
        'JS': keyword_counts["JS"],
        'Javascript': keyword_counts["Javascript"],
        'AA': keyword_counts["AA"],
        'OpenAction': keyword_counts["OpenAction"],
        'Acroform': keyword_counts["Acroform"],
        'JBIG2Decode': keyword_counts["JBIG2Decode"],
        'RichMedia': keyword_counts["RichMedia"],
        'Launch': keyword_counts["Launch"],
        'EmbeddedFile': keyword_counts["EmbeddedFile"],
        'XFA': keyword_counts["XFA"],
        'Colors': keyword_counts["Colors"],
        # Tambahan fitur dinamis
        'JSExec': js_exec,
        'AutoExec': auto_exec,
        'FileDrop': file_drop,
        'ShellExec': shell_exec,
        'ExploitAttempt': exploit_attempt
    }

    return pd.DataFrame([features])


def analyze_pdf(features_df, scenario, top_n_features=5):
    """
    Main analysis function to handle different scenarios.
    """
    if scenario not in MODEL_PATHS:
        return "Invalid Scenario", 0.0, None, "N/A"

    model, pipeline = get_model_and_pipeline(scenario)
    if model is None or pipeline is None:
        error_msg = f"Model or pipeline for scenario '{scenario}' not found. Please upload the required files."
        return error_msg, 0.0, None, "N/A"

    # --- Feature Selection based on Scenario ---
    if scenario in ['A', 'D']:
        feature_cols = STATIC_FEATURES
    elif scenario in ['B', 'C', 'E']:
        feature_cols = STATIC_FEATURES + DYNAMIC_FEATURES
    else: # Should not happen if checked above, but as a safeguard
        return "Invalid Scenario", 0.0, None, "N/A"
        
    # Create a new DataFrame with columns in the correct order for the pipeline
    features_subset = pd.DataFrame(columns=feature_cols)
    for col in feature_cols:
        if col in features_df.columns:
            features_subset[col] = features_df[col]
        else:
            features_subset[col] = 0 # Or some other default

    # Decide whether we are using a unified pipeline (preprocess + model) or separate model
    using_unified_pipeline = model is None and (hasattr(pipeline, 'predict_proba') or hasattr(pipeline, 'predict'))

    if using_unified_pipeline:
        # Let the unified pipeline handle both transformation and prediction
        # Probabilities
        if hasattr(pipeline, 'predict_proba'):
            proba = pipeline.predict_proba(features_subset)[0]
            # Determine predicted class from probabilities if pipeline.predict is missing
            prediction_result = int(np.argmax(proba)) if not hasattr(pipeline, 'predict') else pipeline.predict(features_subset)[0]
            # Ensure proba ordering matches classes_
            if hasattr(pipeline, 'classes_'):
                prob_map = dict(zip(pipeline.classes_, proba))
                prob_benign = prob_map.get(0, 0.0)
                prob_malicious = prob_map.get(1, 0.0)
                prediction_proba = [prob_benign, prob_malicious]
            else:
                # Assume binary and proba is [benign, malicious]
                prediction_proba = proba
        else:
            # No predict_proba: approximate using decision function or predict
            prediction_result = pipeline.predict(features_subset)[0]
            # Fallback fixed confidence
            prediction_proba = [1.0, 0.0] if prediction_result == 0 else [0.0, 1.0]

        # For SHAP, try to separate preprocessor and final estimator if it's a sklearn Pipeline
        data_transformed_for_shap = None
        shap_model = None
        try:
            from sklearn.pipeline import Pipeline as SkPipeline  # local import to avoid hard dep messages
            if isinstance(pipeline, SkPipeline) and len(pipeline.steps) >= 1:
                if len(pipeline.steps) >= 2:
                    preprocessor = SkPipeline(pipeline.steps[:-1])
                    data_transformed_for_shap = preprocessor.transform(features_subset)
                    shap_model = pipeline.steps[-1][1]
                else:
                    # Single-step pipeline: use raw features with the step as model
                    data_transformed_for_shap = features_subset
                    shap_model = pipeline.steps[-1][1]
            else:
                # Unknown type: use pipeline directly on raw features
                data_transformed_for_shap = features_subset
                shap_model = pipeline
        except Exception as e:
            print(f"Error separating pipeline for SHAP: {e}")
            data_transformed_for_shap = features_subset
            shap_model = pipeline

        # Debug: print data info for SHAP
        print(f"Data for SHAP shape: {data_transformed_for_shap.shape if hasattr(data_transformed_for_shap, 'shape') else len(data_transformed_for_shap)}")
        print(f"SHAP model type: {type(shap_model)}")

        top_features = get_top_shap_features(
            data_transformed_for_shap,
            shap_model,
            pipeline,
            features_subset.columns,
            top_n=top_n_features,
            is_keras=False,
        )
    else:
        # 1. Apply the preprocessing pipeline
        processed_features = pipeline.transform(features_subset)

        # --- 2. Make Prediction (handles both Keras and Scikit-learn models) ---
        is_keras_model = hasattr(model, 'predict_on_batch')

        # CONVENTION: 0 = Benign, 1 = Malicious

        if is_keras_model:
            # Keras model prediction. Support both sigmoid (1-unit) and softmax (2-units) outputs.
            pred = model.predict(processed_features, verbose=0)
            if pred.shape[-1] == 1:
                # Sigmoid: single probability for class 1 (Malicious)
                prob_malicious = float(pred[0][0])
                prob_benign = 1.0 - prob_malicious
            elif pred.shape[-1] == 2:
                # Softmax: index 0 = class 0 (Benign), index 1 = class 1 (Malicious)
                prob_benign = float(pred[0][0])
                prob_malicious = float(pred[0][1])
            else:
                raise RuntimeError("Unsupported Keras output shape for binary classification.")

            prediction_proba = [prob_benign, prob_malicious]
            prediction_result = 1 if prob_malicious >= prob_benign else 0
        else:
            # Scikit-learn model prediction
            # We assume the trained model follows the convention {0: 'Benign', 1: 'Malicious'}
            prediction_result = model.predict(processed_features)[0]

            # Ensure correct order of probabilities
            probs = model.predict_proba(processed_features)[0]
            classes_order = model.classes_  # e.g., [0, 1]

            # Create a dictionary to map class label to its probability
            prob_map = dict(zip(classes_order, probs))

            # Ensure we handle cases where a class might not be in the map (should not happen)
            prob_benign = prob_map.get(0, 0.0)
            prob_malicious = prob_map.get(1, 0.0)

            prediction_proba = [prob_benign, prob_malicious]

        # --- 3. Get SHAP values for feature importance ---
        # SHAP explainer needs to be adapted for Keras models
        print(f"Keras model type: {type(model)}")
        print(f"Processed features shape: {processed_features.shape}")
        
        top_features = get_top_shap_features(
            processed_features,
            model,
            pipeline,
            features_subset.columns,
            top_n=top_n_features,
            is_keras=is_keras_model,
        )

    # Final decision with scenario-specific thresholds
    scenario_threshold = SCENARIO_THRESHOLDS.get(scenario, 0.5)
    
    if DECISION_THRESHOLD is not None:
        # Environment variable threshold takes precedence
        threshold_to_use = DECISION_THRESHOLD
    else:
        # Use scenario-specific threshold
        threshold_to_use = scenario_threshold
    
    prob_malicious_final = float(prediction_proba[1])
    if prob_malicious_final >= threshold_to_use:
        status = "Malicious"
        confidence = prob_malicious_final
    else:
        status = "Benign"
        confidence = 1.0 - prob_malicious_final
    
    # Log the threshold used for debugging
    print(f"Scenario {scenario}: Using threshold {threshold_to_use:.2f}, prob_malicious={prob_malicious_final:.4f}, result={status}")
    dominant_feature = top_features[0]['feature'] if top_features else "N/A"

    return status, confidence, top_features, dominant_feature


def get_top_shap_features(data_transformed, model, pipeline, feature_names, top_n=5, is_keras=False):
    """
    Calculates SHAP values and returns the top N most influential features.
    Handles both Keras and Scikit-learn models with caching for better performance.
    """
    # Create a more unique cache key based on model type, model id, and data shape
    model_id = id(model) if model is not None else id(pipeline)
    data_shape = data_transformed.shape if hasattr(data_transformed, 'shape') else len(data_transformed)
    cache_key = f"{'keras' if is_keras else 'sklearn'}_{model_id}_{data_shape}"
    
    explainer = None
    
    # Check if explainer is already cached
    if cache_key not in SHAP_EXPLAINERS:
        print(f"Creating new SHAP explainer for {cache_key}")
        try:
            if is_keras:
                # For Keras models, use a simplified prediction function
                def f(x):
                    return model.predict(x, verbose=0)  # Turn off verbose to reduce output
                # Use a small background dataset for better SHAP calculation
                background_data = data_transformed[:min(10, len(data_transformed))]
                explainer = shap.KernelExplainer(f, background_data)
            else:
                # For scikit-learn models (or unified sklearn pipeline estimator)
                predict_fn = getattr(model, 'predict_proba', None)
                if predict_fn is None:
                    # Fallback to decision_function or predict; SHAP will still approximate
                    predict_fn = getattr(model, 'decision_function', None)
                if predict_fn is None:
                    predict_fn = model.predict
                
                # Use a small background dataset for better SHAP calculation
                background_data = data_transformed[:min(10, len(data_transformed))]
                explainer = shap.KernelExplainer(predict_fn, background_data)
            
            # Cache the explainer for future use
            SHAP_EXPLAINERS[cache_key] = explainer
        except Exception as e:
            print(f"Error creating SHAP explainer: {e}")
            # Return empty features if SHAP fails
            return []
    else:
        print(f"Using cached SHAP explainer for {cache_key}")
        explainer = SHAP_EXPLAINERS[cache_key]
    
    try:
        # Calculate SHAP values with reduced nsamples for better performance
        # For production, you might want to make this configurable
        nsamples = 50 if is_keras else 100  # Keras models typically need fewer samples
        shap_values = explainer.shap_values(data_transformed, nsamples=nsamples)
        
        # The output structure of shap_values can differ. For binary classification,
        # it's often a list of two arrays [shap_for_class_0, shap_for_class_1].
        # We'll focus on what drives the "Malicious" prediction (class 1).
        # For Keras models with sigmoid output, SHAP typically returns values for class 1 (Malicious)
        if is_keras:
            # For Keras models, SHAP values are typically for the positive class (Malicious = 1)
            if isinstance(shap_values, list):
                # If it's a list, take the second element (class 1) or first if only one
                shap_values_for_malicious = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                # If it's a single array, use it directly
                shap_values_for_malicious = shap_values
        else:
            # For scikit-learn models, we want the SHAP values for class 1 (Malicious)
            if isinstance(shap_values, list):
                shap_values_for_malicious = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_values_for_malicious = shap_values
        
        # Ensure we're working with a 1D array for a single prediction
        if len(shap_values_for_malicious.shape) > 1:
            shap_values_for_malicious = shap_values_for_malicious[0]
        
        # Debug: print SHAP values to see what we're getting
        print(f"SHAP values shape: {shap_values_for_malicious.shape}")
        print(f"SHAP values sample: {shap_values_for_malicious[:5]}")
        
        # Get absolute SHAP values to gauge importance regardless of direction
        abs_shap = np.abs(shap_values_for_malicious)
        
        # Get column names from the pipeline if possible
        try:
            # Try direct get_feature_names_out (ColumnTransformer or custom transformers)
            feature_names_out = pipeline.get_feature_names_out()
        except Exception:
            # If it's a sklearn Pipeline, try to get names from the preprocessor step
            try:
                from sklearn.pipeline import Pipeline as SkPipeline
                if isinstance(pipeline, SkPipeline):
                    preprocessor = None
                    # Prefer a named 'preprocessor' step if present
                    if hasattr(pipeline, 'named_steps') and 'preprocessor' in pipeline.named_steps:
                        preprocessor = pipeline.named_steps['preprocessor']
                    elif len(pipeline.steps) >= 2:
                        preprocessor = SkPipeline(pipeline.steps[:-1])
                    if preprocessor is not None and hasattr(preprocessor, 'get_feature_names_out'):
                        feature_names_out = preprocessor.get_feature_names_out()
                    else:
                        feature_names_out = feature_names
                else:
                    feature_names_out = feature_names
            except Exception:
                feature_names_out = feature_names  # Fallback to original names
        
        # Ensure we have the right number of feature names
        if len(feature_names_out) != len(shap_values_for_malicious):
            print(f"Warning: Feature names count ({len(feature_names_out)}) doesn't match SHAP values count ({len(shap_values_for_malicious)})")
            # Use original feature names if there's a mismatch
            feature_names_out = feature_names[:len(shap_values_for_malicious)]
        
        # Get indices of top N features
        top_indices = np.argsort(abs_shap)[-top_n:][::-1]
        
        top_features = []
        for i in top_indices:
            if i < len(feature_names_out) and i < len(shap_values_for_malicious):
                top_features.append({
                    'feature': feature_names_out[i],
                    'value': float(shap_values_for_malicious[i])  # Ensure it's a Python float
                })
        
        print(f"Top features: {top_features}")
        return top_features
        
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        # Return empty features if SHAP calculation fails
        return []


def summarize_extracted_features(features_df: pd.DataFrame, scenario: str):
    """
    Build a human-readable summary of extracted features used by the given scenario.

    Returns a list of dicts: { name, value, kind, unit }
    - kind: one of ['bytes', 'count', 'binary', 'string']
    - unit: 'bytes' for size features, otherwise ''
    """
    if features_df is None or features_df.empty:
        return []

    if scenario in ['A', 'D']:
        feature_cols = STATIC_FEATURES
    elif scenario in ['C', 'E']:
        feature_cols = STATIC_FEATURES + DYNAMIC_FEATURES
    else:
        feature_cols = list(features_df.columns)

    first_row = features_df.iloc[0]

    bytes_fields = {'PdfSize', 'MetadataSize'}
    binary_fields = {'isEncrypted', 'Text', 'JSExec', 'AutoExec', 'FileDrop', 'ShellExec', 'ExploitAttempt'}
    string_fields = {'Header'}

    summary = []
    for name in feature_cols:
        if name not in features_df.columns:
            continue
        value = first_row[name]

        if name in bytes_fields:
            kind = 'bytes'
            unit = 'bytes'
        elif name in binary_fields:
            kind = 'binary'
            unit = ''
        elif name in string_fields:
            kind = 'string'
            unit = ''
        else:
            kind = 'count' if isinstance(value, (int, float, np.integer, np.floating)) else 'string'
            unit = ''

        # Convert numpy scalars to python scalars for JSON-ability
        if isinstance(value, (np.generic,)):
            value = value.item()

        summary.append({
            'name': name,
            'value': value,
            'kind': kind,
            'unit': unit,
        })

    return summary


def generate_short_explanation(top_features, status: str, features_df: pd.DataFrame = None) -> str:
    """
    Compose a short natural-language reason based on SHAP top features and key indicators.
    - If status is 'Malicious', prioritize SHAP features with positive values (push toward Malicious).
    - If status is 'Benign', prioritize SHAP features with negative values (push toward Benign).
    Optionally mentions risky indicators present in the document (JS, Launch, EmbeddedFile, RichMedia, XFA).
    """
    if not top_features:
        return "Penjelasan tidak tersedia."

    # Separate by sign aligned with the predicted status
    aligned = []
    for feat in top_features:
        val = feat.get('value', 0)
        if status == 'Malicious' and val > 0:
            aligned.append(feat['feature'])
        elif status == 'Benign' and val < 0:
            aligned.append(feat['feature'])

    aligned = aligned[:3]

    reason_core = None
    if aligned:
        if status == 'Malicious':
            reason_core = f"Model menilai fitur {', '.join(aligned)} paling mendorong ke Malicious."
        else:
            reason_core = f"Model menilai fitur {', '.join(aligned)} paling mendorong ke Benign."
    else:
        reason_core = "Model tidak menemukan fitur dominan yang jelas."

    # Append indicator hints when available
    indicator_msg = None
    try:
        if features_df is not None and not features_df.empty:
            row = features_df.iloc[0]
            indicators = []
            for name in ['JS', 'Javascript', 'Launch', 'EmbeddedFile', 'RichMedia', 'XFA']:
                try:
                    if name in features_df.columns and int(row[name]) > 0:
                        indicators.append(name)
                except Exception:
                    continue
            if indicators:
                indicator_msg = f" Indikator risiko terdeteksi: {', '.join(indicators)}."
    except Exception:
        pass

    return (reason_core or "") + (indicator_msg or "")
