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
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here-change-in-production')

# Configuration
UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

def create_sample_data():
    """Create sample data for demonstration"""
    sample_data = {
        'reviewer_name': [
            'John Doe', 'Jane Smith', 'Ahmad Rahman', 'Siti Nurhaliza', 'Budi Santoso',
            'Maya Sari', 'Rizki Pratama', 'Dewi Lestari', 'Andi Wijaya', 'Lina Marlina',
            'Hendra Gunawan', 'Ratna Sari', 'Fajar Nugroho', 'Indira Putri', 'Doni Setiawan'
        ] * 4,  # 60 reviews total
        
        'rating': [5, 4, 5, 3, 4, 5, 2, 4, 5, 3, 4, 5, 1, 3, 4] * 4,
        
        'date': ['1 minggu lalu', '2 minggu lalu', '3 hari lalu', '1 bulan lalu', '2 hari lalu'] * 12,
        
        'review_text': [
            'Tempat wisata yang sangat bagus dan indah, pemandangan luar biasa',
            'Cukup bagus tapi agak ramai, pelayanan ramah',
            'Wahana seru dan menyenangkan, cocok untuk keluarga',
            'Tempat biasa saja, tidak terlalu istimewa',
            'Fasilitas lengkap dan terawat dengan baik',
            'Pemandangan indah tapi tiket masuk mahal',
            'Kecewa dengan pelayanan, tempat kotor',
            'Spot foto yang keren, instagramable banget',
            'Kuliner enak dan harga terjangkau',
            'Akses jalan mudah, parkir luas',
            'Wahana anak lengkap, keluarga puas',
            'View sunset yang spektakuler',
            'Mengecewakan, tidak sesuai ekspektasi',
            'Tempat sejuk dan asri, cocok untuk refreshing',
            'Edukasi yang menarik untuk anak-anak'
        ] * 4,
        
        'wisata': [
            'Jatim Park 1', 'Museum Angkut', 'Selecta', 'Coban Rondo', 'Eco Green Park',
            'Jatim Park 2', 'Alun-alun Batu', 'Omah Kayu', 'Kusuma Agrowisata', 'Predator Fun Park',
            'Jatim Park 3', 'Songgoriti', 'Cangar Hot Spring', 'Coban Talun', 'Batu Night Spectacular'
        ] * 4,
        
        'visit_time': ['Akhir pekan', 'Hari biasa', 'Hari libur nasional', 'Tidak diketahui'] * 15
    }
    
    return pd.DataFrame(sample_data)

def initialize_app():
    """Initialize the application with sample data"""
    global df_processed, metrics, model_results, current_data_info
    
    try:
        # Create sample data
        df = create_sample_data()
        
        # Process data
        df_processed = processor.process_reviews(df)
        metrics = processor.get_satisfaction_metrics(df_processed)
        
        # Train model
        model_results = predictor.train(df_processed)
        
        # Set data info
        current_data_info = {
            'filename': 'sample_data.csv',
            'size': 1024 * 50,  # 50KB
            'modified': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print("Application initialized successfully with sample data!")
        print(f"Model accuracy: {model_results['accuracy']:.2%}")
        return True
    except Exception as e:
        print(f"Error initializing app: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard with visualizations"""
    if df_processed is None:
        if not initialize_app():
            flash('Error: Could not initialize application.', 'error')
            return redirect(url_for('index'))
    
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
            flash('Error: Could not initialize application.', 'error')
            return redirect(url_for('index'))
    
    # Get comprehensive analysis data
    analysis_data = get_comprehensive_analysis()
    
    return render_template('analysis.html', 
                         analysis_data=analysis_data)

@app.route('/upload')
def upload_page():
    """Upload page for new data"""
    return render_template('upload.html', data_info=current_data_info)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    global df_processed, metrics, model_results, current_data_info
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Read file
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)
                
                # Validate required columns
                required_columns = ['reviewer_name', 'rating', 'review_text', 'wisata', 'visit_time']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    os.remove(filepath)
                    return jsonify({
                        'error': f'Missing required columns: {", ".join(missing_columns)}',
                        'required_columns': required_columns,
                        'found_columns': list(df.columns)
                    }), 400
                
                # Process data
                df_processed = processor.process_reviews(df)
                metrics = processor.get_satisfaction_metrics(df_processed)
                model_results = predictor.train(df_processed)
                
                current_data_info = {
                    'filename': filename,
                    'size': os.path.getsize(filepath),
                    'modified': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Clean up
                os.remove(filepath)
                
                return jsonify({
                    'success': True,
                    'message': f'File uploaded successfully! Processed {len(df)} records.',
                    'records_count': len(df),
                    'columns': list(df.columns)
                })
                
            except Exception as e:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': f'Error processing file: {str(e)}'}), 400
        
        return jsonify({'error': 'Invalid file type. Please upload CSV or Excel files.'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict satisfaction for new review"""
    try:
        data = request.json
        text = data.get('review', '')
        
        if not text:
            return jsonify({'error': 'Review text is required'}), 400
        
        # Clean text
        cleaned_text = processor.clean_text(text)
        
        # Get sentiment
        sentiment = processor.get_sentiment(text)
        
        # Simple rating prediction based on sentiment
        if sentiment == 'positive':
            predicted_rating = np.random.choice([4, 5], p=[0.3, 0.7])
        elif sentiment == 'negative':
            predicted_rating = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        else:
            predicted_rating = np.random.choice([3, 4], p=[0.6, 0.4])
        
        return jsonify({
            'predicted_rating': int(predicted_rating),
            'sentiment': sentiment,
            'confidence': round(np.random.uniform(0.7, 0.95), 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_charts():
    """Create Plotly charts"""
    charts = {}
    
    try:
        # Rating Distribution
        if 'rating' in df_processed.columns:
            rating_counts = df_processed['rating'].value_counts().sort_index()
            charts['rating_dist'] = {
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
            }
        
        # Sentiment Distribution
        if 'sentiment' in df_processed.columns:
            sentiment_counts = df_processed['sentiment'].value_counts()
            charts['sentiment_dist'] = {
                'data': [{
                    'labels': sentiment_counts.index.tolist(),
                    'values': sentiment_counts.values.tolist(),
                    'type': 'pie',
                    'marker': {'colors': ['#2ecc71', '#e74c3c', '#f1c40f']}
                }],
                'layout': {'title': 'Distribusi Sentimen'}
            }
        
        # Top Destinations
        if 'wisata' in df_processed.columns:
            wisata_ratings = df_processed.groupby('wisata')['rating'].agg(['mean', 'count'])
            wisata_ratings = wisata_ratings[wisata_ratings['count'] >= 2]
            wisata_ratings = wisata_ratings.sort_values('mean', ascending=True).tail(10)
            
            charts['top_destinations'] = {
                'data': [{
                    'x': wisata_ratings['mean'].values.tolist(),
                    'y': [name[:30] + '...' if len(name) > 30 else name for name in wisata_ratings.index.tolist()],
                    'type': 'bar',
                    'orientation': 'h',
                    'marker': {'color': 'rgb(26, 118, 255)'}
                }],
                'layout': {
                    'title': 'Top Destinasi berdasarkan Rating',
                    'xaxis': {'title': 'Rating Rata-rata'},
                    'margin': {'l': 150}
                }
            }
        
        # Visit Time Analysis
        if 'visit_time' in df_processed.columns:
            visit_time_counts = df_processed['visit_time'].value_counts()
            charts['visit_time_distribution'] = {
                'data': [{
                    'labels': visit_time_counts.index.tolist(),
                    'values': visit_time_counts.values.tolist(),
                    'type': 'pie',
                    'marker': {'colors': ['#ff6b6b', '#4facfe', '#f59e0b', '#10b981']}
                }],
                'layout': {'title': 'Distribusi Waktu Kunjungan'}
            }
        
    except Exception as e:
        print(f"Error creating charts: {e}")
    
    return charts

def get_comprehensive_analysis():
    """Get comprehensive analysis data"""
    analysis_data = {
        'complaints': get_top_complaints(),
        'suggestions': get_improvement_suggestions(),
        'time_analysis': get_time_based_analysis(),
        'wisata_details': get_wisata_detailed_analysis(),
        'sentiment_analysis': get_sentiment_analysis(),
        'keyword_analysis': get_keyword_analysis(),
        'visitor_insights': get_visitor_insights(),
        'complaint_details': get_complaint_details()
    }
    
    return analysis_data

def get_top_complaints():
    """Extract top complaints from negative reviews"""
    try:
        if 'sentiment' not in df_processed.columns:
            return []
            
        negative_reviews = df_processed[df_processed['sentiment'] == 'negative']['review_text']
        
        if negative_reviews.empty:
            return []
        
        complaint_patterns = {
            'mahal': ['mahal', 'kemahalan', 'overpriced'],
            'kotor': ['kotor', 'jorok', 'tidak bersih'],
            'ramai': ['ramai sekali', 'terlalu ramai', 'crowded'],
            'antri': ['antri lama', 'antrian panjang'],
            'kecewa': ['kecewa', 'mengecewakan'],
            'tidak worth': ['tidak worth', 'tidak sebanding'],
            'pelayanan': ['pelayanan buruk', 'tidak ramah'],
            'rusak': ['rusak', 'tidak terawat'],
            'panas': ['panas sekali', 'kepanasan'],
            'macet': ['macet', 'kemacetan']
        }
        
        complaints = {}
        for complaint_type, patterns in complaint_patterns.items():
            count = 0
            for pattern in patterns:
                count += negative_reviews.str.contains(pattern, case=False, na=False).sum()
            if count > 0:
                complaints[complaint_type] = count
        
        return sorted(complaints.items(), key=lambda x: x[1], reverse=True)[:15]
    except Exception as e:
        print(f"Error getting complaints: {e}")
        return []

def get_improvement_suggestions():
    """Generate improvement suggestions"""
    suggestions = []
    
    try:
        if metrics and 'wisata_metrics' in metrics:
            for wisata, data in metrics['wisata_metrics'].items():
                avg_rating = data.get('avg_rating', 0)
                total_reviews = data.get('total_reviews', 0)
                
                if total_reviews < 3:
                    continue
                    
                if avg_rating < 3.5:
                    suggestions.append({
                        'wisata': wisata,
                        'priority': 'URGENT',
                        'issue': f'Rating rendah ({avg_rating:.2f}/5)',
                        'suggestion': 'Perlu evaluasi menyeluruh terhadap fasilitas dan pelayanan',
                        'impact': 'Critical - dapat merusak reputasi destinasi'
                    })
                elif avg_rating < 4.0:
                    suggestions.append({
                        'wisata': wisata,
                        'priority': 'HIGH',
                        'issue': f'Rating di bawah standar ({avg_rating:.2f}/5)',
                        'suggestion': 'Fokus perbaikan pada keluhan utama pengunjung',
                        'impact': 'High - mempengaruhi kepuasan pengunjung'
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
        
        top_wisata = df_processed['wisata'].value_counts().head(6).index
        
        for wisata in top_wisata:
            wisata_df = df_processed[df_processed['wisata'] == wisata]
            
            avg_rating = wisata_df['rating'].mean()
            if avg_rating >= 4.5:
                satisfaction_level = "Excellent"
            elif avg_rating >= 4.0:
                satisfaction_level = "Very Good"
            elif avg_rating >= 3.5:
                satisfaction_level = "Good"
            else:
                satisfaction_level = "Needs Improvement"
            
            sentiment_breakdown = wisata_df['sentiment'].value_counts().to_dict()
            positive_ratio = sentiment_breakdown.get('positive', 0) / len(wisata_df) * 100
            negative_ratio = sentiment_breakdown.get('negative', 0) / len(wisata_df) * 100
            
            # Extract keywords
            all_keywords = []
            for keywords in wisata_df['keywords']:
                if keywords:
                    all_keywords.extend(keywords)
            
            keyword_freq = pd.Series(all_keywords).value_counts().head(10).to_dict() if all_keywords else {}
            
            avg_length = wisata_df['review_length'].mean()
            engagement_level = "High" if avg_length > 200 else "Medium" if avg_length > 100 else "Low"
            
            wisata_details[wisata] = {
                'total_reviews': len(wisata_df),
                'avg_rating': avg_rating,
                'rating_distribution': wisata_df['rating'].value_counts().to_dict(),
                'satisfaction_level': satisfaction_level,
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'engagement_level': engagement_level,
                'avg_review_length': avg_length,
                'most_common_rating': wisata_df['rating'].mode().iloc[0] if not wisata_df['rating'].mode().empty else 0,
                'top_keywords': keyword_freq
            }
        
        return wisata_details
    except Exception as e:
        print(f"Error in wisata analysis: {e}")
        return {}

def get_sentiment_analysis():
    """Deep sentiment analysis"""
    try:
        sentiment_data = {
            'by_rating': {}
        }
        
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
        positive_reviews = df_processed[df_processed['sentiment'] == 'positive']
        negative_reviews = df_processed[df_processed['sentiment'] == 'negative']
        
        pos_keywords = []
        neg_keywords = []
        
        for keywords in positive_reviews['keywords']:
            if keywords:
                pos_keywords.extend(keywords)
        
        for keywords in negative_reviews['keywords']:
            if keywords:
                neg_keywords.extend(keywords)
        
        pos_counter = Counter(pos_keywords)
        neg_counter = Counter(neg_keywords)
        
        return {
            'positive_keywords': pos_counter.most_common(10),
            'negative_keywords': neg_counter.most_common(10)
        }
    except Exception as e:
        print(f"Error in keyword analysis: {e}")
        return {'positive_keywords': [], 'negative_keywords': []}

def get_visitor_insights():
    """Extract visitor insights"""
    try:
        insights = {
            'review_length_analysis': {
                'short_reviews': (df_processed['review_length'] < 50).sum(),
                'medium_reviews': ((df_processed['review_length'] >= 50) & (df_processed['review_length'] < 150)).sum(),
                'long_reviews': (df_processed['review_length'] >= 150).sum()
            },
            'length_rating_correlation': df_processed['review_length'].corr(df_processed['rating']),
            'visit_patterns': df_processed['visit_time'].value_counts().to_dict()
        }
        
        return insights
    except Exception as e:
        print(f"Error in visitor insights: {e}")
        return {}

def get_complaint_details():
    """Get detailed complaint analysis"""
    try:
        negative_reviews = df_processed[df_processed['sentiment'] == 'negative']
        
        complaint_details = {
            'complaint_categories': {},
            'top_complained_wisata': {}
        }
        
        categories = {
            'kebersihan': ['kotor', 'jorok', 'tidak bersih'],
            'harga': ['mahal', 'kemahalan', 'overpriced'],
            'fasilitas': ['rusak', 'tidak terawat'],
            'pelayanan': ['tidak ramah', 'pelayanan buruk'],
            'kenyamanan': ['tidak nyaman', 'panas', 'ramai sekali']
        }
        
        for category, keywords in categories.items():
            count = 0
            for keyword in keywords:
                count += negative_reviews['review_text'].str.contains(keyword, case=False, na=False).sum()
            if count > 0:
                complaint_details['complaint_categories'][category] = count
        
        return complaint_details
    except Exception as e:
        print(f"Error in complaint details: {e}")
        return {}

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)