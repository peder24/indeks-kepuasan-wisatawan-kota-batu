import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SatisfactionPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduced for faster training
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def prepare_features(self, df):
        """Prepare features for training"""
        # Create satisfaction label based on rating
        df['satisfaction_label'] = df['rating'].apply(lambda x: 
            'very_satisfied' if x >= 5 else
            'satisfied' if x >= 4 else
            'neutral' if x >= 3 else
            'dissatisfied'
        )
        
        return df
    
    def train(self, df):
        """Train the satisfaction prediction model"""
        try:
            logger.info("Training satisfaction prediction model...")
            
            # Prepare features
            df = self.prepare_features(df)
            
            # Create feature matrix from text
            X_text = self.vectorizer.fit_transform(df['cleaned_text'])
            
            # Add additional features
            additional_features = pd.DataFrame({
                'review_length': df['review_length'],
                'visit_time_encoded': self.label_encoder.fit_transform(df['visit_time'])
            })
            
            # Combine features
            X = np.hstack([X_text.toarray(), additional_features.values])
            y = df['satisfaction_label']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.classifier.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            self.is_trained = True
            
            logger.info(f"Model trained successfully with accuracy: {accuracy:.2%}")
            
            return {
                'accuracy': accuracy,
                'report': report,
                'feature_importance': self.get_feature_importance()
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {
                'accuracy': 0.0,
                'report': 'Training failed',
                'feature_importance': []
            }
    
    def predict_satisfaction(self, text, visit_time='Tidak diketahui'):
        """Predict satisfaction for new review"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        try:
            # Transform text
            X_text = self.vectorizer.transform([text])
            
            # Add additional features
            try:
                visit_time_encoded = self.label_encoder.transform([visit_time])[0]
            except:
                visit_time_encoded = 0
                
            additional_features = np.array([[len(text), visit_time_encoded]])
            
            # Combine features
            X = np.hstack([X_text.toarray(), additional_features])
            
            # Predict
            prediction = self.classifier.predict(X)[0]
            probabilities = self.classifier.predict_proba(X)[0]
            
            return {
                'prediction': prediction,
                'probabilities': dict(zip(self.classifier.classes_, probabilities))
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {
                'prediction': 'neutral',
                'probabilities': {'neutral': 1.0}
            }
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if not self.is_trained:
            return []
            
        try:
            # Get feature names
            text_features = self.vectorizer.get_feature_names_out()
            all_features = list(text_features) + ['review_length', 'visit_time']
            
            # Get importance scores
            importance_scores = self.classifier.feature_importances_
            
            # Create importance dict
            importance_dict = dict(zip(all_features, importance_scores))
            
            # Sort by importance
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            return sorted_importance[:20]  # Top 20 features
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return []
    
    def save_model(self, path='/tmp/models/'):
        """Save trained model"""
        try:
            if not os.path.exists(path):
                os.makedirs(path)
                
            joblib.dump(self.vectorizer, os.path.join(path, 'vectorizer.pkl'))
            joblib.dump(self.classifier, os.path.join(path, 'classifier.pkl'))
            joblib.dump(self.label_encoder, os.path.join(path, 'label_encoder.pkl'))
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, path='/tmp/models/'):
        """Load trained model"""
        try:
            self.vectorizer = joblib.load(os.path.join(path, 'vectorizer.pkl'))
            self.classifier = joblib.load(os.path.join(path, 'classifier.pkl'))
            self.label_encoder = joblib.load(os.path.join(path, 'label_encoder.pkl'))
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")