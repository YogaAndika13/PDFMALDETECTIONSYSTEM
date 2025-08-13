import os
import zipfile
import shutil
from flask import render_template, request, redirect, url_for, flash, jsonify
from app import app
from werkzeug.utils import secure_filename
import pandas as pd
from app.utils import extract_features_from_pdf, analyze_pdf

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def home():
    # Ensure upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        scenario = request.form.get('scenario', 'A') # Default to scenario 'A'

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Pass scenario to the result page
            return redirect(url_for('result', filename=filename, scenario=scenario))

    return render_template('home.html')

@app.route('/result/<filename>')
def result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    results = []
    is_batch = False
    scenario = request.args.get('scenario', 'A') # Get scenario from URL parameter

    if filename.endswith('.zip'):
        is_batch = True
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_zip_extract')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        for item in os.listdir(temp_dir):
            if item.endswith('.pdf'):
                pdf_path = os.path.join(temp_dir, item)
                try:
                    features = extract_features_from_pdf(pdf_path)
                    status, confidence, top_features, dominant_feature = analyze_pdf(features, scenario)
                    
                    if "Model or pipeline for scenario" in status:
                        flash(status) # Show the error message to the user
                        return redirect(url_for('home'))

                    results.append({
                        'filename': item,
                        'status': status,
                        'confidence': confidence,
                        'dominant_feature': dominant_feature,
                        'top_features': top_features
                    })
                except Exception as e:
                    flash(f"An error occurred while processing {item}: {e}")
                    continue
        
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")
    else: # Single PDF
        try:
            features = extract_features_from_pdf(filepath)
            status, confidence, top_features, dominant_feature = analyze_pdf(features, scenario)

            if "Model or pipeline for scenario" in status:
                flash(status)
                return redirect(url_for('home'))

            results.append({
                'filename': filename,
                'status': status,
                'confidence': confidence,
                'dominant_feature': dominant_feature,
                'top_features': top_features
            })
        except Exception as e:
            flash(f"An error occurred while processing {filename}: {e}")
            return redirect(url_for('home'))

    if not results:
        flash('No PDF files found or an error occurred during processing.')
        return redirect(url_for('home'))

    df = pd.DataFrame(results)
    csv_path = os.path.join('app', 'static', 'results.csv')
    df.to_csv(csv_path, index=False)
    
    chart_data = None
    if is_batch:
        status_counts = df['status'].value_counts().to_dict()
        chart_data = {
            'pie_chart': {
                'labels': list(status_counts.keys()),
                'data': list(status_counts.values())
            },
            'bar_chart': {
                'labels': df['filename'].tolist(),
                'data': df['confidence'].tolist()
            }
        }

    return render_template('result.html', results=results, is_batch=is_batch, chart_data=chart_data, csv_path='static/results.csv', scenario=scenario)

@app.route('/about')
def about():
    return render_template('about.html')
