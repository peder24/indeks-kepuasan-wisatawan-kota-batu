import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from collections import Counter
import requests
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data with error handling for Python 3.12.6
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            print(f"Could not download punkt: {e}")

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            print(f"Could not download stopwords: {e}")

# Initialize NLTK data download
download_nltk_data()

class DataProcessor:
    def __init__(self):
        # Load datasets with fallback
        self.stop_words = self.load_stopwords()
        self.positive_words, self.negative_words = self.load_sentiment_words()
        self.domain_words = self.load_domain_words()
    
    def load_stopwords(self):
        """Load Indonesian stopwords with fallback"""
        try:
            # Try online first with timeout
            url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-id/master/stopwords-id.txt"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                stopwords_list = response.text.strip().split('\n')
                print(f"Loaded {len(stopwords_list)} stopwords from online")
                return set(stopwords_list)
            else:
                raise Exception("Failed to fetch from URL")
                
        except Exception as e:
            print(f"Using fallback stopwords: {e}")
            
            # Fallback stopwords - comprehensive Indonesian stopwords
            return set([
                # Basic Indonesian stopwords
                'yang', 'untuk', 'pada', 'ke', 'di', 'dari', 'ini', 'itu', 
                'dengan', 'dan', 'atau', 'bisa', 'ada', 'adalah', 'ya', 
                'tidak', 'juga', 'saya', 'kami', 'kita', 'mereka', 'anda',
                'dia', 'kamu', 'kalian', 'aku', 'akan', 'sudah', 'telah',
                'pernah', 'masih', 'sangat', 'banyak', 'sekali', 'lagi',
                'jadi', 'karena', 'oleh', 'sebagai', 'dalam', 'dapat',
                'hanya', 'semua', 'sendiri', 'masing', 'setiap', 'bila',
                'jika', 'agar', 'maka', 'tentang', 'demikian', 'setelah',
                'saat', 'bahwa', 'ketika', 'seperti', 'belum', 'lain',
                
                # Informal and abbreviations
                'nya', 'banget', 'buat', 'gak', 'ga', 'udah', 'aja', 'sih',
                'deh', 'dong', 'nih', 'tuh', 'kan', 'lah', 'kah', 'pun',
                'juga', 'saja', 'kalo', 'kalau', 'gimana', 'gitu', 'begitu',
                'begini', 'kok', 'sih', 'ya', 'yg', 'dgn', 'utk', 'dr',
                'dg', 'ny', 'd', 'k', 'g', 'tp', 'bs', 'sdh', 'krn',
                
                # Time and measurement words
                'tempat', 'waktu', 'hal', 'orang', 'hari', 'kali', 'buah',
                'tahun', 'jam', 'menit', 'detik', 'minggu', 'bulan',
                
                # Prepositions and auxiliary verbs
                'atas', 'bawah', 'depan', 'belakang', 'samping', 'antara',
                'kepada', 'terhadap', 'bagi', 'tentang', 'mengenai', 'sekitar',
                'ialah', 'yaitu', 'yakni', 'merupakan', 'menjadi',
                'memiliki', 'mempunyai', 'terdapat', 'berada', 'terletak'
            ])
    
    def load_sentiment_words(self):
        """Load sentiment words with fallback"""
        try:
            # Try to load from online with timeout
            positive_words = set()
            negative_words = set()
            
            try:
                url = "https://raw.githubusercontent.com/fajri91/InSet/master/positive_negative_words_id.txt"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    lines = response.text.strip().split('\n')
                    for line in lines[:1000]:  # Limit to first 1000 lines for memory
                        if line.strip():
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                word = parts[0].strip()
                                sentiment = parts[1].strip()
                                if sentiment == 'positive':
                                    positive_words.add(word)
                                elif sentiment == 'negative':
                                    negative_words.add(word)
                    
                    if positive_words and negative_words:
                        print(f"Loaded {len(positive_words)} positive and {len(negative_words)} negative words")
                        return positive_words, negative_words
            except Exception as e:
                print(f"Failed to load from InSet: {e}")
                
        except Exception as e:
            print(f"Using fallback sentiment words: {e}")
        
        # Comprehensive fallback sentiment words for Indonesian tourism
        positive_words = set([
            # Quality words
            'bagus', 'baik', 'indah', 'cantik', 'menarik', 'seru', 'asik',
            'keren', 'mantap', 'luar biasa', 'puas', 'senang', 'suka',
            'recommended', 'rekomendasi', 'worth', 'layak', 'bersih',
            'nyaman', 'ramah', 'murah', 'terjangkau', 'strategis',
            'menyenangkan', 'memuaskan', 'spektakuler', 'mengagumkan',
            'fantastis', 'sempurna', 'terbaik', 'favorit', 'hebat',
            'istimewa', 'menawan', 'memukau', 'menakjubkan',
            'oke', 'ok', 'top', 'juara', 'kece', 'ciamik', 'jos',
            
            # English positive words commonly used
            'amazing', 'awesome', 'great', 'excellent', 'perfect', 'nice',
            'beautiful', 'wonderful', 'good', 'best', 'love', 'like',
            'cool', 'fantastic', 'outstanding', 'superb', 'brilliant',
            
            # Tourism specific positive
            'sejuk', 'asri', 'hijau', 'segar', 'alami', 'natural',
            'terawat', 'rapi', 'tertata', 'modern', 'tradisional',
            'unik', 'eksotis', 'bersejarah', 'edukatif', 'interaktif',
            'fotogenic', 'instagramable', 'panorama', 'pemandangan',
            'view', 'sunset', 'sunrise', 'kuliner', 'lezat', 'enak',
            'lengkap', 'komplit', 'variatif', 'beragam', 'bervariasi',
            'gratis', 'free', 'affordable', 'budget', 'ekonomis'
        ])
        
        negative_words = set([
            # Quality issues
            'buruk', 'jelek', 'kotor', 'mahal', 'kecewa', 'rusak',
            'tidak bagus', 'mengecewakan', 'parah', 'hancur', 'busuk',
            'bau', 'panas', 'macet', 'antri', 'lama', 'lambat',
            'tidak nyaman', 'tidak ramah', 'kasar', 'jorok', 'kumuh',
            'sepi', 'membosankan', 'biasa saja', 'tidak worth', 
            'kurang', 'minim', 'terbatas', 'sempit', 'pengap',
            'gak bagus', 'ga bagus', 'jelek banget', 'buruk sekali',
            'zonk', 'bohong', 'tipu', 'rugi', 'kecewa berat',
            
            # English negative words
            'bad', 'terrible', 'awful', 'worst', 'hate', 'sucks', 'boring',
            'expensive', 'overpriced', 'dirty', 'messy', 'broken', 'damaged',
            'disappointing', 'poor', 'horrible', 'disgusting', 'nasty',
            
            # Tourism specific negative
            'overrated', 'overhyped', 'gersang', 'tandus',
            'tidak terawat', 'berantakan', 'berbahaya', 'licin', 'curam',
            'gelap', 'suram', 'menakutkan', 'tidak aman', 'rawan',
            'penipuan', 'tidak sesuai', 'kemahalan', 'kelamaan', 
            'kepanasan', 'kedinginan', 'sesak', 'sumpek', 'bising',
            'ribut', 'gaduh', 'berisik', 'crowded', 'penuh sesak',
            'antrian panjang', 'parkir susah', 'jauh', 'susah dijangkau',
            'tidak ada fasilitas', 'fasilitas kurang', 'pelayanan buruk',
            'tidak profesional', 'asal-asalan', 'tidak higienis',
            'tidak bersih', 'kurang perawatan', 'maintenance buruk'
        ])
        
        return positive_words, negative_words
    
    def load_domain_words(self):
        """Load tourism domain-specific words"""
        return set([
            # Places and attractions
            'wisata', 'destinasi', 'objek', 'lokasi', 'tempat', 'area',
            'wahana', 'atraksi', 'spot', 'taman', 'pantai', 'gunung',
            'museum', 'monumen', 'candi', 'air terjun', 'kolam', 'danau',
            'pemandian', 'resort', 'villa', 'cottage', 'penginapan',
            'kebun', 'hutan', 'goa', 'curug', 'bendungan', 'jembatan',
            
            # Facilities
            'parkir', 'toilet', 'mushola', 'restoran', 'kafe', 'warung',
            'hotel', 'homestay', 'souvenir', 'oleh-oleh', 'cendera mata',
            'tiket', 'loket', 'pintu', 'gerbang', 'jalan', 'akses',
            'fasilitas', 'amenitas', 'layanan', 'service', 'pelayanan',
            'wifi', 'ac', 'kipas', 'gazebo', 'shelter', 'tempat duduk',
            
            # Activities
            'bermain', 'berenang', 'hiking', 'camping', 'foto', 'selfie',
            'makan', 'belanja', 'jalan-jalan', 'piknik', 'rekreasi',
            'liburan', 'tour', 'trip', 'traveling', 'backpacking',
            'adventure', 'petualangan', 'eksplorasi', 'hunting',
            'tracking', 'trekking', 'climbing', 'rafting', 'diving',
            
            # People
            'anak', 'keluarga', 'dewasa', 'lansia', 'rombongan', 'pasangan',
            'teman', 'wisatawan', 'pengunjung', 'turis', 'backpacker',
            'traveler', 'visitor', 'guest', 'customer', 'guide', 'pemandu'
        ])
    
    def load_data(self, filepath):
        """Load dataset with memory optimization for Python 3.12"""
        try:
            # Load with specific dtypes to save memory
            df = pd.read_csv(filepath, dtype={
                'reviewer_name': 'string',
                'rating': 'int8',
                'date': 'string',
                'review_text': 'string',
                'wisata': 'string',
                'visit_time': 'string'
            })
            print(f"Successfully loaded {len(df)} reviews")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def clean_text(self, text):
        """Clean and preprocess text efficiently"""
        if pd.isna(text) or text == '':
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_sentiment(self, text):
        """Get sentiment with optimized processing for Python 3.12"""
        if not text:
            return 'neutral'
        
        text_lower = text.lower()
        
        # Count positive and negative words with better context handling
        positive_count = 0
        negative_count = 0
        
        # Count positive words
        for word in self.positive_words:
            if word in text_lower:
                # Give more weight to longer, more specific phrases
                weight = 2 if len(word.split()) > 1 else 1
                positive_count += text_lower.count(word) * weight
        
        # Count negative words with context
        for word in self.negative_words:
            if word in text_lower:
                # Give more weight to longer, more specific phrases
                weight = 2 if len(word.split()) > 1 else 1
                negative_count += text_lower.count(word) * weight
        
        # Detect additional negative patterns
        negative_patterns = [
            r'tidak\s+\w+', r'gak\s+\w+', r'ga\s+\w+', 
            r'kurang\s+\w+', r'minim\s+\w+', r'jarang\s+\w+'
        ]
        
        for pattern in negative_patterns:
            matches = re.findall(pattern, text_lower)
            negative_count += len(matches)
        
        # Use TextBlob as backup sentiment analysis
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
        except:
            polarity = 0
        
        # Combine keyword-based and TextBlob sentiment with adjusted thresholds
        if positive_count > negative_count and (positive_count > 0 or polarity > 0.1):
            return 'positive'
        elif negative_count > positive_count and (negative_count > 0 or polarity < -0.1):
            return 'negative'
        else:
            return 'neutral'
    
    def is_meaningful_word(self, word):
        """Check if word is meaningful for analysis"""
        # Skip if too short
        if len(word) < 3:
            return False
            
        # Skip if in stopwords
        if word in self.stop_words:
            return False
            
        # Skip if it's a number
        if word.isdigit():
            return False
            
        # Skip common abbreviations
        common_abbrev = {'yg', 'dgn', 'utk', 'dr', 'dg', 'ny', 'tp', 'bs', 'sdh', 'krn'}
        if word in common_abbrev:
            return False
            
        # Accept if it's a domain-specific word
        if word in self.domain_words:
            return True
            
        # Accept if it's a sentiment word
        if word in self.positive_words or word in self.negative_words:
            return True
            
        # For other words, check if they might be meaningful
        # Skip words that are likely typos or informal variations
        informal_patterns = [
            r'^[a-z]{1,2}$',  # Single or double letters
            r'^ng[a-z]+',     # Words starting with 'ng'
        ]
        
        for pattern in informal_patterns:
            if re.match(pattern, word):
                return False
                
        return True
    
    def extract_keywords(self, text, n=5):
        """Extract keywords efficiently using simple tokenization"""
        if not text:
            return []
            
        try:
            # Use simple split instead of NLTK tokenize for better performance
            words = text.lower().split()
            
            # Filter meaningful words
            meaningful_words = [w for w in words if self.is_meaningful_word(w)]
            
            # Prioritize domain and sentiment words
            priority_words = []
            regular_words = []
            
            for word in meaningful_words:
                if word in self.positive_words or word in self.negative_words or word in self.domain_words:
                    priority_words.append(word)
                else:
                    regular_words.append(word)
            
            # Count frequencies with priority weighting
            all_words = priority_words + regular_words
            word_freq = {}
            for word in all_words:
                # Give higher weight to priority words
                weight = 2 if word in priority_words else 1
                word_freq[word] = word_freq.get(word, 0) + weight
            
            # Sort by frequency and return top n
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Return only the words (not frequencies)
            return [word[0] for word in sorted_words[:n]]
            
        except Exception as e:
            print(f"Error in extract_keywords: {e}")
            return []
    
    def process_reviews(self, df):
        """Process reviews with memory optimization and chunking"""
        print("Processing reviews...")
        
        # Process in chunks to save memory
        chunk_size = 1000
        processed_chunks = []
        
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(df), chunk_size):
            chunk_num = (i // chunk_size) + 1
            print(f"Processing chunk {chunk_num}/{total_chunks}...")
            
            chunk = df.iloc[i:i+chunk_size].copy()
            
            # Clean text
            chunk['cleaned_text'] = chunk['review_text'].apply(self.clean_text)
            
            # Get sentiment
            chunk['sentiment'] = chunk['cleaned_text'].apply(self.get_sentiment)
            
            # Calculate review length
            chunk['review_length'] = chunk['review_text'].astype(str).str.len()
            
            # Extract keywords
            chunk['keywords'] = chunk['cleaned_text'].apply(lambda x: self.extract_keywords(x))
            
            processed_chunks.append(chunk)
            
            # Clean up memory
            del chunk
            gc.collect()
        
        # Combine chunks
        df_processed = pd.concat(processed_chunks, ignore_index=True)
        
        # Convert date with error handling
        df_processed['date_parsed'] = pd.to_datetime(df_processed['date'], errors='coerce')
        
        # Handle missing visit_time
        if 'visit_time' not in df_processed.columns:
            df_processed['visit_time'] = 'Tidak diketahui'
        
        print(f"Processed {len(df_processed)} reviews")
        sentiment_dist = df_processed['sentiment'].value_counts().to_dict()
        print(f"Sentiment distribution: {sentiment_dist}")
        print(f"Loaded datasets - Stopwords: {len(self.stop_words)}, Positive: {len(self.positive_words)}, Negative: {len(self.negative_words)}")
        
        return df_processed
    
    def get_satisfaction_metrics(self, df):
        """Calculate satisfaction metrics efficiently"""
        metrics = {
            'overall_satisfaction': float(df['rating'].mean()),
            'total_reviews': int(len(df)),
            'positive_percentage': float((df['sentiment'] == 'positive').sum() / len(df) * 100),
            'negative_percentage': float((df['sentiment'] == 'negative').sum() / len(df) * 100),
            'neutral_percentage': float((df['sentiment'] == 'neutral').sum() / len(df) * 100),
            'avg_review_length': float(df['review_length'].mean()),
            'rating_distribution': df['rating'].value_counts().to_dict(),
            'wisata_metrics': {}
        }
        
        # Calculate metrics per wisata (limit to top 20 to save memory)
        top_wisata = df['wisata'].value_counts().head(20).index
        
        for wisata in top_wisata:
            wisata_df = df[df['wisata'] == wisata]
            if len(wisata_df) > 0:
                metrics['wisata_metrics'][wisata] = {
                    'avg_rating': float(wisata_df['rating'].mean()),
                    'total_reviews': int(len(wisata_df)),
                    'positive_percentage': float((wisata_df['sentiment'] == 'positive').sum() / len(wisata_df) * 100),
                    'visit_time_distribution': wisata_df['visit_time'].value_counts().to_dict()
                }
        
        return metrics