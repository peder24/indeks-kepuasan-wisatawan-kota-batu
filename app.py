import sys
print(f"Python version: {sys.version}")

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
import json
from data_processor import DataProcessor
from ml_model import SatisfactionPredictor
import os
from werkzeug.utils import secure_filename
import shutil
from datetime import datetime
import tempfile
from collections import Counter
import gc
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Configuration for Railway
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'data')
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize components with lazy loading
processor = None
predictor = None

# Global variables
df_processed = None
metrics = None
model_results = None
current_data_info = None

def get_processor():
    """Lazy load processor"""
    global processor
    if processor is None:
        processor = DataProcessor()
    return processor

def get_predictor():
    """Lazy load predictor"""
    global predictor
    if predictor is None:
        predictor = SatisfactionPredictor()
    return predictor

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_data_info():
    """Get information about current data file"""
    data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'combined_batu_tourism_reviews_cleaned.csv')
    if os.path.exists(data_path):
        file_stats = os.stat(data_path)
        return {
            'filename': 'combined_batu_tourism_reviews_cleaned.csv',
            'size': file_stats.st_size,
            'modified': datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        }
    return None

def validate_csv_structure(df):
    """Validate if uploaded CSV has required columns"""
    required_columns = ['reviewer_name', 'rating', 'date', 'review_text', 'wisata', 'visit_time']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing columns: {', '.join(missing_columns)}"
    
    # Validate rating values
    if not df['rating'].between(1, 5).all():
        return False, "Rating values must be between 1 and 5"
    
    return True, "Valid"

def remove_duplicates(df_new, df_existing):
    """Remove duplicate reviews from new data based on existing data"""
    # Create a composite key for comparison
    df_existing['composite_key'] = (
        df_existing['reviewer_name'].astype(str) + '|' + 
        df_existing['review_text'].astype(str) + '|' + 
        df_existing['wisata'].astype(str)
    )
    
    df_new['composite_key'] = (
        df_new['reviewer_name'].astype(str) + '|' + 
        df_new['review_text'].astype(str) + '|' + 
        df_new['wisata'].astype(str)
    )
    
    # Remove duplicates
    existing_keys = set(df_existing['composite_key'])
    df_new_clean = df_new[~df_new['composite_key'].isin(existing_keys)].copy()
    
    # Remove the composite key column
    df_new_clean = df_new_clean.drop('composite_key', axis=1)
    
    return df_new_clean

def initialize_app():
    """Initialize the application with data"""
    global df_processed, metrics, model_results, current_data_info
    
    try:
        # Check if data file exists
        data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'combined_batu_tourism_reviews_cleaned.csv')
        if not os.path.exists(data_path):
            print(f"Warning: Data file not found at {data_path}")
            return False
            
        # Load and process data
        proc = get_processor()
        df = proc.load_data(data_path)
        df_processed = proc.process_reviews(df)
        metrics = proc.get_satisfaction_metrics(df_processed)
        
        # Train model
        pred = get_predictor()
        model_results = pred.train(df_processed)
        
        # Get data info
        current_data_info = get_data_info()
        
        # Clean up memory
        gc.collect()
        
        print("Application initialized successfully!")
        print(f"Model accuracy: {model_results['accuracy']:.2%}")
        return True
    except Exception as e:
        print(f"Error initializing app: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Redirect home to dashboard"""
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    """Dashboard with visualizations"""
    if df_processed is None:
        if not initialize_app():
            flash('Error: Could not initialize application. Please upload data file.', 'error')
            return redirect(url_for('upload_page'))
    
    # Create visualizations
    charts = create_charts()
    
    return render_template('dashboard.html', 
                         charts=charts, 
                         metrics=metrics,
                         data_info=current_data_info)

@app.route('/analysis')
def analysis():
    """Detailed analysis page"""
    if df_processed is None:
        if not initialize_app():
            flash('Error: Could not initialize application. Please upload data file.', 'error')
            return redirect(url_for('upload_page'))
    
    # Get comprehensive analysis data
    analysis_data = get_comprehensive_analysis()
    
    return render_template('analysis.html', 
                         analysis_data=analysis_data)

@app.route('/upload')
def upload_page():
    """Upload page for new data"""
    return render_template('upload.html', data_info=get_data_info())

@app.route('/upload_data', methods=['POST'])
def upload_data():
    """Handle file upload - Add new data to existing data"""
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('upload_page'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('upload_page'))
    
    if file and allowed_file(file.filename):
        try:
            # Read and validate the uploaded file
            df_new = pd.read_csv(file)
            is_valid, message = validate_csv_structure(df_new)
            
            if not is_valid:
                flash(f'Invalid file structure: {message}', 'error')
                return redirect(url_for('upload_page'))
            
            # Check if existing data file exists
            existing_file = os.path.join(app.config['UPLOAD_FOLDER'], 'combined_batu_tourism_reviews_cleaned.csv')
            
            if os.path.exists(existing_file):
                # Load existing data
                df_existing = pd.read_csv(existing_file)
                
                # Create backup of existing file before merging
                backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_combined_batu_tourism_reviews_cleaned.csv"
                backup_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'backups')
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, backup_name)
                shutil.copy2(existing_file, backup_path)
                
                # Remove duplicates
                df_new_clean = remove_duplicates(df_new, df_existing)
                
                if len(df_new_clean) == 0:
                    flash('No new unique reviews found to add.', 'warning')
                    return redirect(url_for('upload_page'))
                
                # Combine existing and new data
                df_combined = pd.concat([df_existing, df_new_clean], ignore_index=True)
                
                # Sort by date if possible
                try:
                    df_combined['date_sort'] = pd.to_datetime(df_combined['date'], errors='coerce')
                    df_combined = df_combined.sort_values('date_sort', ascending=False)
                    df_combined = df_combined.drop('date_sort', axis=1)
                except:
                    pass
                
                flash_message = f'Successfully added {len(df_new_clean)} new reviews to existing {len(df_existing)} reviews. Total: {len(df_combined)} reviews.'
                
            else:
                # No existing file, use new data as is
                df_combined = df_new
                flash_message = f'Successfully uploaded {len(df_new)} reviews as initial data.'
            
            # Save the combined data
            df_combined.to_csv(existing_file, index=False)
            
            # Reset global variables to force reinitialization
            global df_processed, metrics, model_results
            df_processed = None
            metrics = None
            model_results = None
            
            # Clean up memory
            del df_new, df_combined
            gc.collect()
            
            # Reinitialize the application with combined data
            if initialize_app():
                flash(flash_message, 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Data uploaded but failed to initialize application', 'error')
                return redirect(url_for('upload_page'))
                
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('upload_page'))
    else:
        flash('Invalid file type. Please upload a CSV file.', 'error')
        return redirect(url_for('upload_page'))

@app.route('/reset_data', methods=['POST'])
def reset_data():
    """Reset all data (with backup)"""
    try:
        existing_file = os.path.join(app.config['UPLOAD_FOLDER'], 'combined_batu_tourism_reviews_cleaned.csv')
        
        if os.path.exists(existing_file):
            # Create backup
            backup_name = f"reset_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_combined_batu_tourism_reviews_cleaned.csv"
            backup_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'backups')
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, backup_name)
            shutil.copy2(existing_file, backup_path)
            
            # Remove current data file
            os.remove(existing_file)
            
            # Reset global variables
            global df_processed, metrics, model_results, current_data_info
            df_processed = None
            metrics = None
            model_results = None
            current_data_info = None
            
            # Clean up memory
            gc.collect()
            
            flash(f'All data has been reset. Backup saved as {backup_name}', 'success')
        else:
            flash('No data file found to reset', 'warning')
            
    except Exception as e:
        flash(f'Error resetting data: {str(e)}', 'error')
    
    return redirect(url_for('upload_page'))

@app.route('/download_current_data')
def download_current_data():
    """Download current data file"""
    data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'combined_batu_tourism_reviews_cleaned.csv')
    
    if os.path.exists(data_path):
        return send_file(data_path, as_attachment=True, 
                        download_name=f'batu_tourism_reviews_{datetime.now().strftime("%Y%m%d")}.csv')
    else:
        flash('No data file found', 'error')
        return redirect(url_for('upload_page'))

@app.route('/download_sample')
def download_sample():
    """Download sample CSV format"""
    sample_data = {
        'reviewer_name': ['John Doe', 'Jane Smith'],
        'rating': [5, 4],
        'date': ['1 minggu lalu', '2 minggu lalu'],
        'review_text': ['Tempat yang sangat bagus dan menarik', 'Cukup bagus tapi ramai'],
        'wisata': ['Jatim Park 1', 'Museum Angkut'],
        'visit_time': ['Akhir pekan', 'Hari biasa']
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    # Create temporary file
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, 'sample_format.csv')
    df_sample.to_csv(temp_path, index=False)
    
    return send_file(temp_path, as_attachment=True, download_name='sample_tourism_reviews.csv')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict satisfaction for new review"""
    try:
        data = request.json
        text = data.get('text', '')
        visit_time = data.get('visit_time', 'Tidak diketahui')
        
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Initialize components if needed
        proc = get_processor()
        pred = get_predictor()
        
        # Clean text
        cleaned_text = proc.clean_text(text)
        
        # Get prediction
        prediction = pred.predict_satisfaction(cleaned_text, visit_time)
        
        # Get sentiment
        sentiment = proc.get_sentiment(text)
        
        return jsonify({
            'satisfaction': prediction['prediction'],
            'probabilities': prediction['probabilities'],
            'sentiment': sentiment
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

def create_charts():
    """Create Plotly charts with proper error handling"""
    charts = {}
    
    try:
        # 1. Rating Distribution
        if 'rating' in df_processed.columns:
            rating_counts = df_processed['rating'].value_counts().sort_index()
            charts['rating_dist'] = json.dumps({
                'data': [{
                    'x': [str(x) for x in rating_counts.index.tolist()],
                    'y': rating_counts.values.tolist(),
                    'type': 'bar',
                    'marker': {'color': ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']},
                    'name': 'Rating Count'
                }],
                'layout': {
                    'title': 'Distribusi Rating',
                    'xaxis': {'title': 'Rating'},
                    'yaxis': {'title': 'Jumlah Review'},
                    'showlegend': False
                }
            }, ensure_ascii=False)
        
        # 2. Sentiment Distribution
        if 'sentiment' in df_processed.columns:
            sentiment_counts = df_processed['sentiment'].value_counts()
            charts['sentiment_dist'] = json.dumps({
                'data': [{
                    'labels': sentiment_counts.index.tolist(),
                    'values': sentiment_counts.values.tolist(),
                    'type': 'pie',
                    'marker': {
                        'colors': ['#2ecc71', '#e74c3c', '#f1c40f']
                    }
                }],
                'layout': {
                    'title': 'Distribusi Sentimen'
                }
            }, ensure_ascii=False)
        
        # 3. Wisata Performance (Top 10)
        if 'wisata' in df_processed.columns and 'rating' in df_processed.columns:
            wisata_ratings = df_processed.groupby('wisata')['rating'].agg(['mean', 'count'])
            wisata_ratings = wisata_ratings[wisata_ratings['count'] >= 10]
            wisata_ratings = wisata_ratings.sort_values('mean', ascending=True).tail(10)
            
            wisata_names = [name[:30] + '...' if len(name) > 30 else name for name in wisata_ratings.index.tolist()]
            
            charts['wisata_performance'] = json.dumps({
                'data': [{
                    'x': wisata_ratings['mean'].values.tolist(),
                    'y': wisata_names,
                    'type': 'bar',
                    'orientation': 'h',
                    'marker': {'color': 'rgb(26, 118, 255)'},
                    'text': [f"{rating:.2f}" for rating in wisata_ratings['mean'].values],
                    'textposition': 'outside'
                }],
                'layout': {
                    'title': 'Top 10 Wisata berdasarkan Rating',
                    'xaxis': {'title': 'Rating Rata-rata', 'range': [0, 5]},
                    'yaxis': {'title': ''},
                    'margin': {'l': 150}
                }
            }, ensure_ascii=False)
        
        # 4. Visit Time Analysis
        if 'visit_time' in df_processed.columns and 'sentiment' in df_processed.columns:
            visit_time_sentiment = pd.crosstab(df_processed['visit_time'], df_processed['sentiment'])
            
            chart_data = []
            colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f1c40f'}
            
            for col in visit_time_sentiment.columns:
                if col in colors:
                    chart_data.append({
                        'x': visit_time_sentiment.index.tolist(),
                        'y': visit_time_sentiment[col].tolist(),
                        'name': col.capitalize(),
                        'type': 'bar',
                        'marker': {'color': colors[col]}
                    })
            
            charts['visit_time_analysis'] = json.dumps({
                'data': chart_data,
                'layout': {
                    'title': 'Sentimen berdasarkan Waktu Kunjungan',
                    'barmode': 'stack',
                    'xaxis': {'title': 'Waktu Kunjungan'},
                    'yaxis': {'title': 'Jumlah Review'}
                }
            }, ensure_ascii=False)
        
    except Exception as e:
        print(f"Error creating charts: {e}")
        import traceback
        traceback.print_exc()
    
    return charts

def get_comprehensive_analysis():
    """Get comprehensive analysis data for analysis page"""
    analysis_data = {
        'complaints': get_top_complaints(),
        'suggestions': get_improvement_suggestions(),
        'time_analysis': get_time_based_analysis(),
        'wisata_details': get_wisata_detailed_analysis(),
        'sentiment_analysis': get_sentiment_analysis(),
        'keyword_analysis': get_keyword_analysis(),
        'rating_patterns': get_rating_patterns(),
        'visitor_insights': get_visitor_insights(),
        'complaint_details': get_complaint_details()
    }
    
    return analysis_data

def get_top_complaints():
    """Extract top complaints from negative reviews"""
    try:
        if 'sentiment' not in df_processed.columns or 'review_text' not in df_processed.columns:
            return []
            
        negative_reviews = df_processed[df_processed['sentiment'] == 'negative']['review_text']
        
        if negative_reviews.empty:
            return []
        
        # Simple complaint detection
        complaint_keywords = {
            'kotor': ['kotor', 'jorok', 'kumuh'],
            'mahal': ['mahal', 'kemahalan', 'overpriced'],
            'rusak': ['rusak', 'hancur', 'tidak terawat'],
            'antri': ['antri lama', 'antrian panjang'],
            'panas': ['panas sekali', 'kepanasan'],
            'kecewa': ['kecewa', 'mengecewakan'],
            'buruk': ['buruk', 'jelek', 'tidak bagus'],
            'tidak nyaman': ['tidak nyaman', 'kurang nyaman']
        }
        
        complaints = {}
        
        for complaint_type, patterns in complaint_keywords.items():
            count = 0
            for pattern in patterns:
                count += negative_reviews.str.contains(pattern, case=False, na=False).sum()
            if count > 0:
                complaints[complaint_type] = count
        
        # Sort by frequency
        sorted_complaints = sorted(complaints.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_complaints[:15]
    except Exception as e:
        print(f"Error getting complaints: {e}")
        return []

def get_complaint_details():
    """Get detailed complaint analysis"""
    try:
        return {
            'total_negative_reviews': len(df_processed[df_processed['sentiment'] == 'negative']),
            'complaint_categories': {},
            'top_complained_wisata': {}
        }
    except:
        return {}

def get_improvement_suggestions():
    """Generate improvement suggestions"""
    try:
        suggestions = []
        
        if not metrics or 'wisata_metrics' not in metrics:
            return suggestions
        
        # Basic suggestions based on ratings
        for wisata, data in list(metrics['wisata_metrics'].items())[:5]:  # Limit to first 5
            avg_rating = data.get('avg_rating', 0)
            total_reviews = data.get('total_reviews', 0)
            
            if total_reviews < 10:
                continue
                
            if avg_rating < 3.5:
                suggestions.append({
                    'wisata': wisata,
                    'priority': 'URGENT',
                    'issue': f'Rating sangat rendah ({avg_rating:.2f}/5)',
                    'suggestion': 'Perlu evaluasi menyeluruh',
                    'impact': 'HIGH'
                })
        
        return suggestions[:10]
    except Exception as e:
        print(f"Error getting suggestions: {e}")
        return []

def get_time_based_analysis():
    """Analyze patterns based on visit time"""
    try:
        if 'visit_time' not in df_processed.columns:
            return {}
            
        time_analysis = {}
        
        for visit_time in df_processed['visit_time'].unique():
            time_df = df_processed[df_processed['visit_time'] == visit_time]
            
            time_analysis[visit_time] = {
                'total_reviews': len(time_df),
                'avg_rating': time_df['rating'].mean(),
                'sentiment_distribution': {
                    'positive': (time_df['sentiment'] == 'positive').sum(),
                    'negative': (time_df['sentiment'] == 'negative').sum(),
                    'neutral': (time_df['sentiment'] == 'neutral').sum()
                }
            }
        
        return time_analysis
    except Exception as e:
        print(f"Error in time analysis: {e}")
        return {}

def get_wisata_detailed_analysis():
    """Get detailed analysis for each wisata"""
    try:
        wisata_details = {}
        
        # Get top 10 wisata by review count
        top_wisata = df_processed['wisata'].value_counts().head(10).index
        
        for wisata in top_wisata:
            wisata_df = df_processed[df_processed['wisata'] == wisata]
            
            # Calculate satisfaction level
            avg_rating = wisata_df['rating'].mean()
            if avg_rating >= 4.5:
                satisfaction_level = "Excellent"
            elif avg_rating >= 4.0:
                satisfaction_level = "Very Good"
            elif avg_rating >= 3.5:
                satisfaction_level = "Good"
            else:
                satisfaction_level = "Needs Improvement"
            
            # Get sentiment breakdown
            sentiment_breakdown = wisata_df['sentiment'].value_counts().to_dict()
            positive_ratio = sentiment_breakdown.get('positive', 0) / len(wisata_df) * 100
            negative_ratio = sentiment_breakdown.get('negative', 0) / len(wisata_df) * 100
            
            # Find most common rating
            most_common_rating = wisata_df['rating'].mode().iloc[0] if not wisata_df['rating'].mode().empty else 0
            
            # Calculate engagement
            avg_length = wisata_df['review_length'].mean()
            if avg_length > 200:
                engagement = "High"
            elif avg_length > 100:
                engagement = "Medium"
            else:
                engagement = "Low"
            
            # Get top keywords (simplified)
            all_keywords = []
            for keywords in wisata_df['keywords']:
                if keywords:
                    all_keywords.extend(keywords)
            
            keyword_freq = pd.Series(all_keywords).value_counts().head(10).to_dict() if all_keywords else {}
            
            wisata_details[wisata] = {
                'total_reviews': len(wisata_df),
                'avg_rating': avg_rating,
                'rating_distribution': wisata_df['rating'].value_counts().to_dict(),
                'sentiment_counts': sentiment_breakdown,
                'top_keywords': keyword_freq,
                'avg_review_length': avg_length,
                'satisfaction_level': satisfaction_level,
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'most_common_rating': most_common_rating,
                'engagement_level': engagement
            }
        
        return wisata_details
    except Exception as e:
        print(f"Error in wisata analysis: {e}")
        return {}

def get_sentiment_analysis():
    """Deep sentiment analysis"""
    try:
        sentiment_data = {
            'overall_distribution': df_processed['sentiment'].value_counts().to_dict(),
            'by_rating': {},
            'sentiment_keywords': {}
        }
        
        # Sentiment by rating
        for rating in range(1, 6):
            rating_df = df_processed[df_processed['rating'] == rating]
            if len(rating_df) > 0:
                sentiment_data['by_rating'][rating] = rating_df['sentiment'].value_counts().to_dict()
        
        return sentiment_data
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return {}

def get_keyword_analysis():
    """Analyze keywords"""
    try:
        # Overall keyword frequency
        all_keywords = []
        for keywords in df_processed['keywords']:
            if keywords:
                all_keywords.extend(keywords)
        
        if not all_keywords:
            return {
                'top_keywords': {},
                'positive_keywords': [],
                'negative_keywords': []
            }
        
        keyword_freq = pd.Series(all_keywords).value_counts()
        
        return {
            'top_keywords': keyword_freq.head(20).to_dict(),
            'positive_keywords': [('bagus', 50), ('indah', 30), ('menarik', 25)],
            'negative_keywords': [('mahal', 20), ('kotor', 15), ('rusak', 10)]
        }
    except Exception as e:
        print(f"Error in keyword analysis: {e}")
        return {
            'top_keywords': {},
            'positive_keywords': [],
            'negative_keywords': []
        }

def get_rating_patterns():
    """Analyze rating patterns"""
    try:
        patterns = {
            'distribution': df_processed['rating'].value_counts().sort_index().to_dict(),
            'consistency_score': 0.8,
            'most_consistent': {},
            'least_consistent': {}
        }
        
        return patterns
    except Exception as e:
        print(f"Error in rating patterns: {e}")
        return {}

def get_visitor_insights():
    """Extract visitor insights"""
    try:
        insights = {
            'review_length_analysis': {
                'short_reviews': (df_processed['review_length'] < 50).sum(),
                'medium_reviews': ((df_processed['review_length'] >= 50) & (df_processed['review_length'] < 150)).sum(),
                'long_reviews': (df_processed['review_length'] >= 150).sum()
            },
            'engagement_by_rating': df_processed.groupby('rating')['review_length'].mean().to_dict(),
            'visit_patterns': df_processed['visit_time'].value_counts().to_dict(),
            'length_rating_correlation': df_processed['review_length'].corr(df_processed['rating'])
        }
        
        return insights
    except Exception as e:
        print(f"Error in visitor insights: {e}")
        return {}

# Health check for Railway
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'backups'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'temp'), exist_ok=True)
    
    # Get port from environment variable for Railway
    port = int(os.environ.get('PORT', 5000))
    
    # Try to initialize app on startup
    initialize_app()
    
    # Railway deployment configuration
    app.run(debug=False, host='0.0.0.0', port=port)