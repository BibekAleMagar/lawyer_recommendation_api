import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

class SimplifiedLawyerModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3), 
            stop_words='english',
            min_df=3,
            max_df=0.6 
        )
        self.model = XGBClassifier(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=8,
            objective='multi:softprob',
            random_state=42,
            n_jobs=-1
        )
        self.label_encoder = LabelEncoder()
        
        # High-accuracy consolidated categories
        self.category_keywords = {
            'Criminal': ['Criminal', 'Felony', 'Misdemeanor', 'Defense', 'Prosecution', 'Appellate', 'Police', 'Arrest'],
            'Civil_Litigation': ['Civil', 'Litigation', 'Tort', 'Negligence', 'Rights', 'Trial', 'Equity', 'Administrative', 'Injury', 'Damage'],
            'Corporate_Contract': ['Corporate', 'Business', 'Merger', 'Commercial', 'Contract', 'Breach', 'Agreement', 'Employment', 'Regulatory', 'Compliance'],
            'Property_Estate': ['Property', 'Real Estate', 'Landlord', 'Tenant', 'Housing', 'Trust', 'Estate', 'Probate', 'Will'],
            'Family': ['Family', 'Child', 'Custody', 'Divorce', 'Matrimonial', 'Adoption']
        }

    def clean_text(self, text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return " ".join(text.split())

    def simplify_specialization(self, spec):
        if pd.isna(spec): return 'Civil_Litigation'
        spec = str(spec).title()
        for category, keywords in self.category_keywords.items():
            for kw in keywords:
                if kw.lower() in spec.lower(): return category
        return 'Civil_Litigation'

    def prepare_data(self, df):
        df = df.copy()
        df['main_category'] = df['assigned_lawyer_specialization'].apply(self.simplify_specialization)
        
        # Give Nature of Suit 5x the weight of the summary
        df['combined_text'] = (
            df['case_name'].apply(self.clean_text) + " " + 
            (df['nature_of_suit'].apply(self.clean_text) + " ") * 5 + 
            df['summary'].apply(self.clean_text)
        )
        return df

    def train(self, df):
        print("\n--- Starting XGBoost + SMOTE Optimization ---")
        df = self.prepare_data(df)
        
        X_vec = self.vectorizer.fit_transform(df['combined_text'])
        y_encoded = self.label_encoder.fit_transform(df['main_category'])

        # Fix Data Imbalance using SMOTE
        # 
        print("Balancing dataset with SMOTE...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_vec, y_encoded)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.15, random_state=42
        )

        print(f"Training XGBoost on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n🚀 PROJECTED ACCURACY: {accuracy:.1%}")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return {"accuracy": accuracy}

    def predict(self, case_name, nature_of_suit, summary):
        text = self.clean_text(f"{case_name} {(nature_of_suit + ' ') * 5} {summary}")
        vec = self.vectorizer.transform([text])
        
        probs = self.model.predict_proba(vec)[0]
        pred_idx = np.argmax(probs)
        category = self.label_encoder.inverse_transform([pred_idx])[0]
        
        return {
            'predicted_category': category,
            'confidence': round(float(probs[pred_idx]), 4),
            'top_3': {
                self.label_encoder.inverse_transform([i])[0]: round(float(probs[i]), 4)
                for i in np.argsort(probs)[-3:][::-1]
            }
        }

    def save_model(self, path='simplified_model.pkl'):
        joblib.dump({
            'vectorizer': self.vectorizer,
            'model': self.model,
            'label_encoder': self.label_encoder,
            'category_keywords': self.category_keywords
        }, path)
        print(f"✓ Model saved to {path}")

    def load_model(self, path='simplified_model.pkl'):
        data = joblib.load(path)
        self.vectorizer = data['vectorizer']
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.category_keywords = data['category_keywords']

# --- THIS IS THE CRITICAL ADDITION ---
if __name__ == "__main__":
    # 1. Load Data
    print("Initializing training script...")
    df = None
    
    try:
        import data_loader
        df = data_loader.dataset
        print(f"✓ Data loaded from data_loader ({len(df)} rows)")
    except Exception as e:
        print(f"⚠ data_loader failed: {e}")
        if os.path.exists("data.csv"):
            df = pd.read_csv("data.csv")
            print(f"✓ Loaded data.csv directly ({len(df)} rows)")
    
    if df is not None:
        # 2. Train
        trainer = SimplifiedLawyerModel()
        results = trainer.train(df)
        
        # 3. Save
        trainer.save_model('simplified_model.pkl')
        print("\n✅ SUCCESS: Model is ready for the API.")
    else:
        print("❌ ERROR: No dataset found. Please check your data.csv or data_loader.py")