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
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-for-development')

# Configuration for Vercel
UPLOAD_FOLDER = '/tmp/data' if os.environ.get('VERCEL') else 'data'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize components
processor = DataProcessor()
predictor = SatisfactionPredictor()

# Global variables
df_processed = None
metrics = None
model_results = None
current_data_info = None

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

def create_sample_data():
    """Create sample data for initial deployment"""
    sample_data = {
        'reviewer_name': [
            'Ahmad Santoso', 'Siti Nurhaliza', 'Budi Prasetyo', 'Rina Kusuma', 'Dedi Wijaya',
            'Maya Sari', 'Andi Pratama', 'Lina Wati', 'Rudi Hartono', 'Dewi Anggraini',
            'Fajar Nugroho', 'Indah Permata', 'Joko Susilo', 'Eka Putri', 'Hendra Gunawan',
            'Novi Rahayu', 'Agus Setiawan', 'Ratna Dewi', 'Bambang Sutrisno', 'Ani Widodo'
        ],
        'rating': [5, 4, 5, 3, 4, 5, 4, 3, 5, 4, 3, 4, 5, 2, 4, 5, 3, 4, 5, 4],
        'date': [
            '1 minggu lalu', '2 minggu lalu', '3 minggu lalu', '1 bulan lalu', '2 bulan lalu',
            '1 minggu lalu', '2 minggu lalu', '3 minggu lalu', '1 bulan lalu', '2 bulan lalu',
            '1 minggu lalu', '2 minggu lalu', '3 minggu lalu', '1 bulan lalu', '2 bulan lalu',
            '1 minggu lalu', '2 minggu lalu', '3 minggu lalu', '1 bulan lalu', '2 bulan lalu'
        ],
        'review_text': [
            'Tempat wisata yang sangat bagus dan menarik. Anak-anak sangat senang bermain di sini. Fasilitas lengkap dan bersih.',
            'Cukup bagus tapi agak ramai. Harga tiket masih terjangkau. Parkir luas dan mudah.',
            'Wahana seru dan menantang. Staff ramah dan membantu. Recommended untuk keluarga.',
            'Biasa saja, tidak terlalu istimewa. Agak kotor di beberapa area. Perlu perawatan lebih baik.',
            'Bagus untuk liburan keluarga. Banyak spot foto yang menarik. Makanan di food court enak.',
            'Luar biasa! Pengalaman yang tak terlupakan. Wahana modern dan aman. Pasti akan kembali lagi.',
            'Lumayan bagus tapi antrian panjang. Sebaiknya datang pagi hari. Overall memuaskan.',
            'Kurang terawat dan agak kusam. Harga tidak sebanding dengan fasilitas. Kecewa.',
            'Spektakuler! Pemandangan indah dan udara sejuk. Perfect untuk refreshing dari rutinitas.',
            'Bagus dan edukatif. Anak-anak bisa belajar sambil bermain. Konsep yang menarik.',
            'Standar saja, tidak ada yang spesial. Mungkin cocok untuk anak kecil. Harga lumayan.',
            'Menyenangkan dan seru. Banyak wahana yang bisa dicoba. Cocok untuk semua umur.',
            'Amazing experience! Teknologi canggih dan interaktif. Must visit destination di Batu.',
            'Mengecewakan. Banyak wahana rusak dan tidak terawat. Pelayanan kurang memuaskan.',
            'Bagus untuk foto-foto. Instagram-able banget. Tapi agak panas siang hari.',
            'Fantastis! Koleksi lengkap dan unik. Anak-anak takjub melihat semua koleksi.',
            'Biasa aja sih. Mungkin karena ekspektasi terlalu tinggi. Lumayan lah untuk sekali kunjung.',
            'Recommended! Wahana lengkap dan terawat. Harga tiket worth it dengan fasilitasnya.',
            'Keren abis! Suasana malam yang romantis. Perfect untuk date atau family time.',
            'Bagus dan menyenangkan. Pelayanan ramah. Fasilitas bersih dan nyaman.'
        ],
        'wisata': [
            'Jatim Park 1', 'Museum Angkut', 'Jatim Park 2', 'Eco Green Park', 'Jatim Park 3',
            'Batu Night Spectacular', 'Museum Satwa', 'Selecta', 'Coban Rondo', 'Alun-alun Batu',
            'Jatim Park 1', 'Museum Angkut', 'Jatim Park 2', 'Eco Green Park', 'Jatim Park 3',
            'Batu Night Spectacular', 'Museum Satwa', 'Selecta', 'Coban Rondo', 'Alun-alun Batu'
        ],
        'visit_time': [
            'Akhir pekan', 'Hari biasa', 'Akhir pekan', 'Hari libur nasional', 'Hari biasa',
            'Akhir pekan', 'Hari biasa', 'Akhir pekan', 'Hari libur nasional', 'Hari biasa',
            'Akhir pekan', 'Hari biasa', 'Akhir pekan', 'Hari libur nasional', 'Hari biasa',
            'Akhir pekan', 'Hari biasa', 'Akhir pekan', 'Hari libur nasional', 'Hari biasa'
        ]
    }
    
    return pd.DataFrame(sample_data)

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
        # Create necessary directories
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'backups'), exist_ok=True)
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'temp'), exist_ok=True)
        
        # Check if data file exists
        data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'combined_batu_tourism_reviews_cleaned.csv')
        
        if not os.path.exists(data_path):
            logger.info("No existing data found, creating sample data...")
            # Create sample data for initial deployment
            df_sample = create_sample_data()
            df_sample.to_csv(data_path, index=False)
            logger.info(f"Sample data created with {len(df_sample)} records")
            
        # Load and process data
        df = processor.load_data(data_path)
        df_processed = processor.process_reviews(df)
        metrics = processor.get_satisfaction_metrics(df_processed)
        
        # Train model
        model_results = predictor.train(df_processed)
        
        # Get data info
        current_data_info = get_data_info()
        
        logger.info("Application initialized successfully!")
        logger.info(f"Model accuracy: {model_results['accuracy']:.2%}")
        return True
    except Exception as e:
        logger.error(f"Error initializing app: {e}")
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
                backup_path = os.path.join(app.config['UPLOAD_FOLDER'], 'backups', backup_name)
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                shutil.copy2(existing_file, backup_path)
                
                # Remove duplicates based on reviewer_name, review_text, and wisata
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
            backup_path = os.path.join(app.config['UPLOAD_FOLDER'], 'backups', backup_name)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            shutil.copy2(existing_file, backup_path)
            
            # Remove current data file
            os.remove(existing_file)
            
            # Reset global variables
            global df_processed, metrics, model_results, current_data_info
            df_processed = None
            metrics = None
            model_results = None
            current_data_info = None
            
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
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp', 'sample_format.csv')
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
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
        
        # Clean text
        cleaned_text = processor.clean_text(text)
        
        # Get prediction
        prediction = predictor.predict_satisfaction(cleaned_text, visit_time)
        
        # Get sentiment
        sentiment = processor.get_sentiment(text)
        
        return jsonify({
            'satisfaction': prediction['prediction'],
            'probabilities': prediction['probabilities'],
            'sentiment': sentiment
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
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
            wisata_ratings = wisata_ratings[wisata_ratings['count'] >= 2]  # Lowered threshold for sample data
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
                    'title': 'Top Wisata berdasarkan Rating',
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
        logger.error(f"Error creating charts: {e}")
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
    """Extract top complaints from negative reviews with context analysis"""
    try:
        if 'sentiment' not in df_processed.columns or 'review_text' not in df_processed.columns:
            return []
            
        negative_reviews = df_processed[df_processed['sentiment'] == 'negative']['review_text']
        
        if negative_reviews.empty:
            return []
        
        # Define complaint keywords with their negative contexts
        complaint_patterns = {
            'kotor': ['kotor', 'jorok', 'kumuh', 'tidak bersih', 'kurang bersih'],
            'mahal': ['mahal', 'kemahalan', 'overpriced', 'tidak worth', 'harga tinggi'],
            'rusak': ['rusak', 'hancur', 'tidak terawat', 'bobrok'],
            'antri': ['antri lama', 'antrian panjang', 'mengantri', 'antre'],
            'panas': ['panas sekali', 'kepanasan', 'terik', 'gerah'],
            'kecewa': ['kecewa', 'mengecewakan', 'tidak puas', 'disappointed'],
            'buruk': ['buruk', 'jelek', 'tidak bagus', 'payah'],
            'lama': ['lama sekali', 'kelamaan', 'menunggu lama', 'lambat'],
            'tidak nyaman': ['tidak nyaman', 'kurang nyaman', 'uncomfortable'],
            'susah': ['susah', 'sulit', 'ribet', 'repot'],
            'macet': ['macet', 'kemacetan', 'terjebak macet'],
            'penuh': ['penuh sesak', 'terlalu penuh', 'overcrowded'],
            'bau': ['bau', 'berbau', 'tidak sedap', 'pesing'],
            'sempit': ['sempit', 'sumpek', 'tidak luas'],
            'pengap': ['pengap', 'tidak ada ventilasi', 'sesak'],
            'kasar': ['kasar', 'tidak ramah', 'jutek', 'galak'],
            'sepi': ['sepi', 'tidak ramai pengunjung', 'sunyi'],
            'membosankan': ['membosankan', 'boring', 'tidak menarik', 'biasa saja']
        }
        
        complaints = {}
        
        # Process definite complaints
        for complaint_type, patterns in complaint_patterns.items():
            count = 0
            for pattern in patterns:
                count += negative_reviews.str.contains(pattern, case=False, na=False).sum()
            if count > 0:
                complaints[complaint_type] = count
        
        # Sort by frequency
        sorted_complaints = sorted(complaints.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_complaints[:15]
    except Exception as e:
        logger.error(f"Error getting complaints: {e}")
        return []

def get_improvement_suggestions():
    """Generate improvement suggestions based on analysis"""
    try:
        suggestions = []
        
        if not metrics or 'wisata_metrics' not in metrics:
            return suggestions
        
        # Analyze each wisata
        for wisata, data in metrics['wisata_metrics'].items():
            avg_rating = data.get('avg_rating', 0)
            total_reviews = data.get('total_reviews', 0)
            
            if total_reviews < 2:  # Lowered threshold for sample data
                continue
                
            if avg_rating < 3.5:
                suggestions.append({
                    'wisata': wisata,
                    'priority': 'URGENT',
                    'issue': f'Rating sangat rendah ({avg_rating:.2f}/5)',
                    'suggestion': 'Perlu evaluasi menyeluruh dan perbaikan segera pada semua aspek layanan',
                    'impact': 'HIGH'
                })
            elif avg_rating < 4.0:
                suggestions.append({
                    'wisata': wisata,
                    'priority': 'HIGH',
                    'issue': f'Rating di bawah standar ({avg_rating:.2f}/5)',
                    'suggestion': 'Fokus perbaikan pada keluhan utama pengunjung',
                    'impact': 'MEDIUM'
                })
        
        # Overall suggestions
        if metrics.get('negative_percentage', 0) > 20:
            suggestions.append({
                'wisata': 'SEMUA DESTINASI',
                'priority': 'URGENT',
                'issue': f'Sentimen negatif tinggi ({metrics["negative_percentage"]:.1f}%)',
                'suggestion': 'Implementasi sistem penanganan keluhan terpadu dan responsif',
                'impact': 'HIGH'
            })
        
        # Sort by priority
        priority_order = {'URGENT': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        suggestions.sort(key=lambda x: priority_order.get(x['priority'], 4))
        
        return suggestions[:10]
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
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
                },
                'top_wisata': time_df['wisata'].value_counts().head(3).to_dict()
            }
        
        return time_analysis
    except Exception as e:
        logger.error(f"Error in time analysis: {e}")
        return {}

def get_wisata_detailed_analysis():
    """Get detailed analysis for each wisata"""
    try:
        wisata_details = {}
        
        # Get top wisata by review count
        top_wisata = df_processed['wisata'].value_counts().head(10).index
        
        for wisata in top_wisata:
            wisata_df = df_processed[df_processed['wisata'] == wisata]
            
            # Extract common keywords from this wisata's reviews
            all_keywords = []
            for keywords in wisata_df['keywords']:
                if keywords:
                    all_keywords.extend(keywords)
            
            keyword_freq = pd.Series(all_keywords).value_counts().head(10).to_dict() if all_keywords else {}
            
            # Calculate satisfaction level based on rating
            avg_rating = wisata_df['rating'].mean()
            if avg_rating >= 4.5:
                satisfaction_level = "Excellent"
                level_color = "success"
            elif avg_rating >= 4.0:
                satisfaction_level = "Very Good"
                level_color = "info"
            elif avg_rating >= 3.5:
                satisfaction_level = "Good"
                level_color = "warning"
            else:
                satisfaction_level = "Needs Improvement"
                level_color = "danger"
            
            # Get sentiment breakdown
            sentiment_breakdown = wisata_df['sentiment'].value_counts().to_dict()
            positive_ratio = sentiment_breakdown.get('positive', 0) / len(wisata_df) * 100
            negative_ratio = sentiment_breakdown.get('negative', 0) / len(wisata_df) * 100
            
            # Find most common rating
            most_common_rating = wisata_df['rating'].mode().iloc[0] if not wisata_df['rating'].mode().empty else 0
            
            # Calculate review engagement (based on review length)
            avg_length = wisata_df['review_length'].mean()
            if avg_length > 200:
                engagement = "High"
            elif avg_length > 100:
                engagement = "Medium"
            else:
                engagement = "Low"
            
            wisata_details[wisata] = {
                'total_reviews': len(wisata_df),
                'avg_rating': avg_rating,
                'rating_distribution': wisata_df['rating'].value_counts().to_dict(),
                'sentiment_counts': sentiment_breakdown,
                'top_keywords': keyword_freq,
                'avg_review_length': avg_length,
                'satisfaction_level': satisfaction_level,
                'level_color': level_color,
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'most_common_rating': most_common_rating,
                'engagement_level': engagement
            }
        
        return wisata_details
    except Exception as e:
        logger.error(f"Error in wisata analysis: {e}")
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
        
        # Extract keywords by sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_df = df_processed[df_processed['sentiment'] == sentiment]
            all_keywords = []
            for keywords in sentiment_df['keywords']:
                if keywords:
                    all_keywords.extend(keywords)
            
            if all_keywords:
                sentiment_data['sentiment_keywords'][sentiment] = pd.Series(all_keywords).value_counts().head(15).to_dict()
            else:
                sentiment_data['sentiment_keywords'][sentiment] = {}
        
        return sentiment_data
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return {}

def get_keyword_analysis():
    """Analyze most common keywords and phrases"""
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
        
        # Categorize keywords
        positive_keywords = []
        negative_keywords = []
        
        # Ambil keywords dari review positif dan negatif
        positive_reviews = df_processed[df_processed['sentiment'] == 'positive']
        negative_reviews = df_processed[df_processed['sentiment'] == 'negative']
        
        # Keywords dari review positif
        pos_keywords = []
        for keywords in positive_reviews['keywords']:
            if keywords:
                pos_keywords.extend(keywords)
        
        # Keywords dari review negatif  
        neg_keywords = []
        for keywords in negative_reviews['keywords']:
            if keywords:
                neg_keywords.extend(keywords)
        
        # Hitung frekuensi di masing-masing sentiment
        pos_counter = Counter(pos_keywords)
        neg_counter = Counter(neg_keywords)
        
        # Kategorikan berdasarkan dominasi di sentiment tertentu
        all_unique_keywords = set(pos_keywords + neg_keywords)
        
        for keyword in all_unique_keywords:
            pos_count = pos_counter.get(keyword, 0)
            neg_count = neg_counter.get(keyword, 0)
            total_count = pos_count + neg_count
            
            if total_count >= 2:  # Minimal muncul 2 kali untuk sample data
                # Jika lebih dominan di positive
                if pos_count > neg_count and pos_count / total_count >= 0.6:
                    positive_keywords.append((keyword, pos_count))
                # Jika lebih dominan di negative
                elif neg_count > pos_count and neg_count / total_count >= 0.6:
                    negative_keywords.append((keyword, neg_count))
        
        # Sort
        positive_keywords.sort(key=lambda x: x[1], reverse=True)
        negative_keywords.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'top_keywords': keyword_freq.head(20).to_dict(),
            'positive_keywords': positive_keywords[:15],
            'negative_keywords': negative_keywords[:15]
        }
    except Exception as e:
        logger.error(f"Error in keyword analysis: {e}")
        return {
            'top_keywords': {},
            'positive_keywords': [],
            'negative_keywords': []
        }

def get_rating_patterns():
    """Analyze rating patterns and distributions"""
    try:
        patterns = {
            'distribution': df_processed['rating'].value_counts().sort_index().to_dict(),
            'by_wisata_type': {},
            'consistency_score': 0
        }
        
        # Calculate consistency score (lower std = more consistent)
        wisata_ratings_std = df_processed.groupby('wisata')['rating'].std()
        patterns['consistency_score'] = 1 - (wisata_ratings_std.mean() / 5)  # Normalize to 0-1
        
        # Identify wisata with most consistent ratings
        patterns['most_consistent'] = wisata_ratings_std.nsmallest(5).to_dict()
        patterns['least_consistent'] = wisata_ratings_std.nlargest(5).to_dict()
        
        return patterns
    except Exception as e:
        logger.error(f"Error in rating patterns: {e}")
        return {}

def get_visitor_insights():
    """Extract insights about visitors"""
    try:
        insights = {
            'review_length_analysis': {
                'short_reviews': (df_processed['review_length'] < 50).sum(),
                'medium_reviews': ((df_processed['review_length'] >= 50) & (df_processed['review_length'] < 150)).sum(),
                'long_reviews': (df_processed['review_length'] >= 150).sum()
            },
            'engagement_by_rating': df_processed.groupby('rating')['review_length'].mean().to_dict(),
            'visit_patterns': df_processed['visit_time'].value_counts().to_dict()
        }
        
        # Correlation between review length and rating
        insights['length_rating_correlation'] = df_processed['review_length'].corr(df_processed['rating'])
        
        return insights
    except Exception as e:
        logger.error(f"Error in visitor insights: {e}")
        return {}

def get_complaint_details():
    """Get detailed complaint analysis with context"""
    try:
        if 'sentiment' not in df_processed.columns:
            return {}
            
        negative_reviews = df_processed[df_processed['sentiment'] == 'negative']
        
        complaint_details = {
            'total_negative_reviews': len(negative_reviews),
            'complaint_categories': {},
            'top_complained_wisata': {}
        }
        
        # Categorize complaints
        categories = {
            'kebersihan': ['kotor', 'jorok', 'kumuh', 'bau', 'tidak bersih', 'kurang bersih'],
            'harga': ['mahal', 'kemahalan', 'overpriced', 'tidak worth'],
            'fasilitas': ['rusak', 'tidak terawat', 'kurang fasilitas', 'tidak ada'],
            'pelayanan': ['tidak ramah', 'kasar', 'jutek', 'pelayanan buruk'],
            'kenyamanan': ['tidak nyaman', 'panas', 'pengap', 'sempit', 'ramai sekali'],
            'akses': ['macet', 'susah', 'jauh', 'sulit dijangkau'],
            'antrian': ['antri lama', 'antrian panjang', 'menunggu lama']
        }
        
        for category, keywords in categories.items():
            count = 0
            for keyword in keywords:
                count += negative_reviews['review_text'].str.contains(keyword, case=False, na=False).sum()
            if count > 0:
                complaint_details['complaint_categories'][category] = count
        
        # Find which wisata get most complaints
        if len(negative_reviews) > 0:
            wisata_complaints = negative_reviews['wisata'].value_counts().head(5)
            for wisata, count in wisata_complaints.items():
                wisata_df = negative_reviews[negative_reviews['wisata'] == wisata]
                # Extract main complaints for this wisata
                main_complaints = []
                for category, keywords in categories.items():
                    cat_count = 0
                    for keyword in keywords:
                        cat_count += wisata_df['review_text'].str.contains(keyword, case=False, na=False).sum()
                    if cat_count > 0:
                        main_complaints.append((category, cat_count))
                
                complaint_details['top_complained_wisata'][wisata] = {
                    'total_complaints': count,
                    'main_issues': sorted(main_complaints, key=lambda x: x[1], reverse=True)[:3]
                }
        
        return complaint_details
        
    except Exception as e:
        logger.error(f"Error in complaint details: {e}")
        return {}

# Initialize app on startup for Vercel
try:
    initialize_app()
except Exception as e:
    logger.error(f"Failed to initialize on startup: {e}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)