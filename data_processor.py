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

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class DataProcessor:
    def __init__(self):
        # Load datasets from URLs or local files
        self.stop_words = self.load_stopwords()
        self.positive_words, self.negative_words = self.load_sentiment_words()
        self.domain_words = self.load_domain_words()
    
    def load_stopwords(self):
        """Load Indonesian stopwords from dataset"""
        try:
            # Try to load from URL first
            url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-id/master/stopwords-id.txt"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                stopwords_list = response.text.strip().split('\n')
                print(f"Loaded {len(stopwords_list)} stopwords from online dataset")
                return set(stopwords_list)
            else:
                raise Exception("Failed to fetch from URL")
                
        except Exception as e:
            print(f"Failed to load stopwords from URL: {e}")
            print("Using fallback stopwords...")
            
            # Fallback to curated list
            return set([
                # Kata ganti dan kata sambung umum
                'yang', 'untuk', 'pada', 'ke', 'di', 'dari', 'ini', 'itu', 
                'dengan', 'dan', 'atau', 'bisa', 'ada', 'adalah', 'ya', 
                'tidak', 'juga', 'saya', 'kami', 'kita', 'mereka', 'anda',
                'dia', 'kamu', 'kalian', 'aku', 'akan', 'sudah', 'telah',
                'pernah', 'masih', 'sangat', 'banyak', 'sekali', 'lagi',
                'jadi', 'karena', 'oleh', 'sebagai', 'dalam', 'dapat',
                'hanya', 'semua', 'sendiri', 'masing', 'setiap', 'bila',
                'jika', 'agar', 'maka', 'tentang', 'demikian', 'setelah',
                'saat', 'bahwa', 'ketika', 'seperti', 'belum', 'lain',
                
                # Kata informal dan singkatan
                'nya', 'banget', 'buat', 'gak', 'ga', 'udah', 'aja', 'sih',
                'deh', 'dong', 'nih', 'tuh', 'kan', 'lah', 'kah', 'pun',
                'juga', 'saja', 'kalo', 'kalau', 'gimana', 'gitu', 'begitu',
                'begini', 'kok', 'sih', 'ya', 'yg', 'dgn', 'utk', 'dr',
                'dg', 'ny', 'd', 'k', 'g', 'tp', 'bs', 'sdh', 'krn',
                
                # Kata waktu dan ukuran
                'tempat', 'waktu', 'hal', 'orang', 'hari', 'kali', 'buah',
                'tahun', 'jam', 'menit', 'detik', 'minggu', 'bulan',
                
                # Preposisi dan kata kerja bantu
                'atas', 'bawah', 'depan', 'belakang', 'samping', 'antara',
                'kepada', 'terhadap', 'bagi', 'tentang', 'mengenai', 'sekitar',
                'adalah', 'ialah', 'yaitu', 'yakni', 'merupakan', 'menjadi',
                'memiliki', 'mempunyai', 'terdapat', 'berada', 'terletak'
            ])
    
    def load_sentiment_words(self):
        """Load positive and negative words from dataset"""
        try:
            # Try to load from comprehensive Indonesian sentiment lexicon
            positive_words = set()
            negative_words = set()
            
            # Option 1: Try to load from InSet repository
            try:
                url = "https://raw.githubusercontent.com/fajri91/InSet/master/positive_negative_words_id.txt"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    lines = response.text.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                word = parts[0].strip()
                                sentiment = parts[1].strip()
                                if sentiment == 'positive':
                                    positive_words.add(word)
                                elif sentiment == 'negative':
                                    negative_words.add(word)
                    
                    print(f"Loaded {len(positive_words)} positive and {len(negative_words)} negative words from InSet")
                    return positive_words, negative_words
            except:
                pass
            
            # Option 2: Try alternative source
            try:
                # Load positive words
                pos_url = "https://raw.githubusercontent.com/riochr17/Analisis-Sentimen-ID/master/positive_words.txt"
                response = requests.get(pos_url, timeout=10)
                if response.status_code == 200:
                    positive_words = set(response.text.strip().split('\n'))
                
                # Load negative words
                neg_url = "https://raw.githubusercontent.com/riochr17/Analisis-Sentimen-ID/master/negative_words.txt"
                response = requests.get(neg_url, timeout=10)
                if response.status_code == 200:
                    negative_words = set(response.text.strip().split('\n'))
                
                if positive_words and negative_words:
                    print(f"Loaded {len(positive_words)} positive and {len(negative_words)} negative words from alternative source")
                    return positive_words, negative_words
            except:
                pass
                
        except Exception as e:
            print(f"Failed to load sentiment words from online sources: {e}")
        
        print("Using fallback sentiment words...")
        
        # Fallback positive words
        positive_words = set([
            # Kualitas umum
            'bagus', 'baik', 'indah', 'cantik', 'menarik', 'seru', 'asik',
            'keren', 'mantap', 'luar biasa', 'puas', 'senang', 'suka',
            'recommended', 'rekomendasi', 'worth', 'layak', 'bersih',
            'nyaman', 'ramah', 'murah', 'terjangkau', 'strategis',
            'menyenangkan', 'memuaskan', 'spektakuler', 'mengagumkan',
            'fantastis', 'sempurna', 'terbaik', 'favorit', 'hebat',
            'istimewa', 'menawan', 'memukau', 'menakjubkan',
            'oke', 'ok', 'top', 'juara', 'kece', 'ciamik', 'jos',
            'amazing', 'awesome', 'great', 'excellent', 'perfect', 'nice',
            'beautiful', 'wonderful', 'good', 'best', 'love', 'like',
            
            # Spesifik pariwisata
            'sejuk', 'asri', 'hijau', 'segar', 'alami', 'natural',
            'terawat', 'rapi', 'tertata', 'modern', 'tradisional',
            'unik', 'eksotis', 'bersejarah', 'edukatif', 'interaktif',
            'fotogenic', 'instagramable', 'panorama', 'pemandangan',
            'view', 'sunset', 'sunrise', 'kuliner', 'lezat', 'enak',
            'lengkap', 'komplit', 'variatif', 'beragam', 'bervariasi',
            'murah', 'gratis', 'free', 'affordable', 'budget'
        ])
        
        # Fallback negative words
        negative_words = set([
            # Kualitas buruk
            'buruk', 'jelek', 'kotor', 'mahal', 'kecewa', 'rusak',
            'tidak bagus', 'mengecewakan', 'parah', 'hancur', 'busuk',
            'bau', 'panas', 'macet', 'antri', 'lama', 'lambat',
            'tidak nyaman', 'tidak ramah', 'kasar', 'jorok', 'kumuh',
            'sepi', 'membosankan', 'biasa saja', 'tidak worth', 
            'kurang', 'minim', 'terbatas', 'sempit', 'pengap',
            'gak bagus', 'ga bagus', 'jelek banget', 'buruk sekali',
            'zonk', 'bohong', 'tipu', 'rugi', 'kecewa berat',
            'bad', 'terrible', 'awful', 'worst', 'hate', 'sucks', 'boring',
            'expensive', 'overpriced', 'dirty', 'messy', 'broken', 'damaged',
            
            # Spesifik pariwisata
            'overrated', 'overhyped', 'gersang', 'tandus',
            'tidak terawat', 'berantakan', 'berbahaya', 'licin', 'curam',
            'gelap', 'suram', 'menakutkan', 'tidak aman', 'rawan',
            'penipuan', 'tidak sesuai', 'kemahalan', 'kelamaan', 
            'kepanasan', 'kedinginan', 'sesak', 'sumpek', 'bising',
            'ribut', 'gaduh', 'berisik', 'crowded', 'penuh sesak',
            'antrian panjang', 'parkir susah', 'jauh', 'susah dijangkau',
            'tidak ada fasilitas', 'fasilitas kurang', 'pelayanan buruk',
            'tidak profesional', 'asal-asalan', 'tidak higienis',
            'tidak bersih', 'kurang perawatan'
        ])
        
        return positive_words, negative_words
    
    def load_domain_words(self):
        """Load domain-specific words for tourism"""
        # Tourism domain words - these are curated for Indonesian tourism context
        return set([
            # Tempat wisata
            'wisata', 'destinasi', 'objek', 'lokasi', 'tempat', 'area',
            'wahana', 'atraksi', 'spot', 'taman', 'pantai', 'gunung',
            'museum', 'monumen', 'candi', 'air terjun', 'kolam', 'danau',
            'pemandian', 'resort', 'villa', 'cottage', 'penginapan',
            
            # Fasilitas
            'parkir', 'toilet', 'mushola', 'restoran', 'kafe', 'warung',
            'hotel', 'homestay', 'souvenir', 'oleh-oleh', 'cendera mata',
            'tiket', 'loket', 'pintu', 'gerbang', 'jalan', 'akses',
            'fasilitas', 'amenitas', 'layanan', 'service', 'pelayanan',
            
            # Aktivitas
            'bermain', 'berenang', 'hiking', 'camping', 'foto', 'selfie',
            'makan', 'belanja', 'jalan-jalan', 'piknik', 'rekreasi',
            'liburan', 'tour', 'trip', 'traveling', 'backpacking',
            'adventure', 'petualangan', 'eksplorasi', 'hunting',
            
            # Pengunjung
            'anak', 'keluarga', 'dewasa', 'lansia', 'rombongan', 'pasangan',
            'teman', 'wisatawan', 'pengunjung', 'turis', 'backpacker',
            'traveler', 'visitor', 'guest', 'customer'
        ])
    
    def save_datasets_locally(self):
        """Save loaded datasets locally for faster future loading"""
        try:
            os.makedirs('datasets', exist_ok=True)
            
            # Save stopwords
            with open('datasets/stopwords_id.txt', 'w', encoding='utf-8') as f:
                for word in sorted(self.stop_words):
                    f.write(f"{word}\n")
            
            # Save positive words
            with open('datasets/positive_words_id.txt', 'w', encoding='utf-8') as f:
                for word in sorted(self.positive_words):
                    f.write(f"{word}\n")
            
            # Save negative words
            with open('datasets/negative_words_id.txt', 'w', encoding='utf-8') as f:
                for word in sorted(self.negative_words):
                    f.write(f"{word}\n")
            
            # Save domain words
            with open('datasets/domain_words_tourism.txt', 'w', encoding='utf-8') as f:
                for word in sorted(self.domain_words):
                    f.write(f"{word}\n")
            
            print("Datasets saved locally in 'datasets' folder")
            
        except Exception as e:
            print(f"Failed to save datasets locally: {e}")
    
    def load_data(self, filepath):
        """Load and return the dataset"""
        try:
            df = pd.read_csv(filepath)
            print(f"Successfully loaded {len(df)} reviews")
            
            # Save datasets locally after successful initialization
            self.save_datasets_locally()
            
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text) or text == '':
            return ""
        
        # Convert to string
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_sentiment(self, text):
        """Get sentiment score based on Indonesian keywords and context"""
        if not text:
            return 'neutral'
        
        text_lower = text.lower()
        
        # Count positive and negative words dengan konteks yang lebih baik
        positive_count = 0
        negative_count = 0
        
        # Hitung kata positif
        for word in self.positive_words:
            if word in text_lower:
                # Berikan bobot lebih untuk kata yang lebih spesifik
                weight = 2 if len(word.split()) > 1 else 1
                positive_count += text_lower.count(word) * weight
        
        # Hitung kata negatif dengan konteks
        for word in self.negative_words:
            if word in text_lower:
                # Berikan bobot lebih untuk kata yang lebih spesifik
                weight = 2 if len(word.split()) > 1 else 1
                negative_count += text_lower.count(word) * weight
        
        # Deteksi pola negatif tambahan
        negative_patterns = [
            r'tidak\s+\w+', r'gak\s+\w+', r'ga\s+\w+', 
            r'kurang\s+\w+', r'minim\s+\w+', r'jarang\s+\w+'
        ]
        
        for pattern in negative_patterns:
            matches = re.findall(pattern, text_lower)
            negative_count += len(matches)
        
        # Also use TextBlob as backup
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
        except:
            polarity = 0
        
        # Combine keyword-based and TextBlob sentiment dengan threshold yang lebih sensitif
        if positive_count > negative_count and (positive_count > 0 or polarity > 0.05):
            return 'positive'
        elif negative_count > positive_count and (negative_count > 0 or polarity < -0.05):
            return 'negative'
        else:
            return 'neutral'
    
    def is_meaningful_word(self, word):
        """Check if a word is meaningful for analysis"""
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
        """Extract top meaningful keywords from text"""
        if not text:
            return []
            
        try:
            # Tokenize
            words = word_tokenize(text.lower())
            
            # Filter meaningful words only
            meaningful_words = [w for w in words if self.is_meaningful_word(w)]
            
            # If we have domain-specific or sentiment words, prioritize them
            priority_words = []
            regular_words = []
            
            for word in meaningful_words:
                if word in self.positive_words or word in self.negative_words or word in self.domain_words:
                    priority_words.append(word)
                else:
                    regular_words.append(word)
            
            # Get word frequency
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
        """Process all reviews and add features"""
        print("Processing reviews...")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Clean text
        df['cleaned_text'] = df['review_text'].apply(self.clean_text)
        
        # Get sentiment
        df['sentiment'] = df['cleaned_text'].apply(self.get_sentiment)
        
        # Calculate review length
        df['review_length'] = df['review_text'].astype(str).str.len()
        
        # Extract keywords for each review
        df['keywords'] = df['cleaned_text'].apply(lambda x: self.extract_keywords(x))
        
        # Convert date to datetime
        df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Handle missing visit_time
        if 'visit_time' not in df.columns:
            df['visit_time'] = 'Tidak diketahui'
        
        print(f"Processed {len(df)} reviews")
        print(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
        print(f"Loaded datasets - Stopwords: {len(self.stop_words)}, Positive: {len(self.positive_words)}, Negative: {len(self.negative_words)}")
        
        return df
    
    def get_satisfaction_metrics(self, df):
        """Calculate satisfaction metrics"""
        metrics = {
            'overall_satisfaction': df['rating'].mean(),
            'total_reviews': len(df),
            'positive_percentage': (df['sentiment'] == 'positive').sum() / len(df) * 100,
            'negative_percentage': (df['sentiment'] == 'negative').sum() / len(df) * 100,
            'neutral_percentage': (df['sentiment'] == 'neutral').sum() / len(df) * 100,
            'avg_review_length': df['review_length'].mean(),
            'rating_distribution': df['rating'].value_counts().to_dict(),
            'wisata_metrics': {}
        }
        
        # Calculate metrics per wisata
        for wisata in df['wisata'].unique():
            wisata_df = df[df['wisata'] == wisata]
            metrics['wisata_metrics'][wisata] = {
                'avg_rating': wisata_df['rating'].mean(),
                'total_reviews': len(wisata_df),
                'positive_percentage': (wisata_df['sentiment'] == 'positive').sum() / len(wisata_df) * 100 if len(wisata_df) > 0 else 0,
                'visit_time_distribution': wisata_df['visit_time'].value_counts().to_dict()
            }
        
        return metrics