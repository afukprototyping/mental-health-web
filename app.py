# -*- coding: utf-8 -*-
"""app.py - Versi final dengan fitur Chatbot (klasifikasi + retrieval), Journaling (deteksi mental health), dan Speech-to-Text."""

import re
import os
import pickle
import sys
import random
import pandas as pd
import numpy as np
import torch
import joblib
import wave 
import shutil
from io import BytesIO 
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from typing import Dict, Set, Tuple, Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks 
from pydantic import BaseModel
import uvicorn
from sklearn.preprocessing import LabelEncoder

# Import Google Cloud STT
try:
    from google.cloud import speech
except ImportError:
    print("⚠️ Peringatan: Library google-cloud-speech tidak terinstal. Speech-to-Text akan non-fungsional.", file=sys.stderr)
    class DummySpeechClient:
        def recognize(self, config, audio):
            raise ImportError("Google Cloud Speech library is not available.")
    speech = type('DummySpeechModule', (object,), {'SpeechClient': DummySpeechClient, 'RecognitionAudio': object, 'RecognitionConfig': object, 'AudioEncoding': type('DummyAudioEncoding', (object,), {'LINEAR16': 1})})()


# --- KONFIGURASI PATHS & PARAMETER ---
MODEL_DIR = "model"
PKL_MODEL_PATH = os.path.join(MODEL_DIR, "final_gb_model.pkl") 
PKL_COMPONENTS_PATH = os.path.join(MODEL_DIR, "preprocessed_components.pkl")
MENTAL_HEALTH_MODEL_PATH = os.path.join(MODEL_DIR, "mental_health_final.joblib")

DF_PATH = "df (1).csv"

# PATH FILE KREDENSIAL GOOGLE CLOUD
GOOGLE_CREDENTIALS_PATH = "sec-gacor-df69bd5246d2.json"
if os.path.exists(GOOGLE_CREDENTIALS_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH
    print("✅ GOOGLE_APPLICATION_CREDENTIALS telah disetel.")
else:
    print(f"❌ File kredensial Google Cloud '{GOOGLE_CREDENTIALS_PATH}' tidak ditemukan. STT akan gagal.", file=sys.stderr)


MODEL_NAME = "indolem/indobert-base-uncased" # Model untuk Chatbot
# >>>>> INI ADALAH VARIABEL YANG HILANG DAN PERLU DITAMBAHKAN <<<<<
MODEL_NAME_MH = "indobenchmark/indobert-base-p1" # Model untuk Journaling (sesuai pipeline lama)
SIMILARITY_THRESHOLD = 0.60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FASTAPI SETUP ---
app = FastAPI(
    title="Mental Health Chatbot & Journaling API",
    description="Chatbot & journaling detector untuk mendeteksi potensi gangguan kesehatan mental.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- UTILITAS ---
def minimal_preprocessing(text: Union[str, float]) -> str:
    """Membersihkan karakter khusus dan whitespace"""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- FUNGSI SPEECH-TO-TEXT GOOGLE CLOUD ---
def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Mengubah byte audio WAV menjadi teks menggunakan Google Cloud Speech-to-Text."""
    try:
        if 'speech' not in sys.modules or not hasattr(speech, 'SpeechClient'):
             raise ImportError("Google Cloud Speech library tidak dimuat dengan benar.")

        client = speech.SpeechClient()

        with wave.open(BytesIO(audio_bytes), 'rb') as wf:
            sample_rate_hertz = wf.getframerate()
            
        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate_hertz,
            language_code="id-ID",
            model="default"
        )

        response = client.recognize(config=config, audio=audio)

        if response.results:
            return response.results[0].alternatives[0].transcript
        else:
            return ""

    except ImportError as ie:
        raise HTTPException(status_code=503, detail=f"Speech-to-Text Dependency Error: {str(ie)}. Pastikan library google-cloud-speech diinstal dan kredensial disetel.")
    except Exception as e:
        print(f"Google STT API Error: {str(e)}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Gagal menghubungi Google STT API. Error: {str(e)}. Pastikan koneksi internet dan kredensial Google Cloud Anda valid.")


class KeywordNERPredict:
    """KeywordNER untuk digunakan saat prediksi."""
    def __init__(self, label_keywords_stored: Dict[str, Set[str]]):
        self.mental_health_keywords = {
            'anxiety': ['cemas', 'gelisah', 'khawatir', 'panik', 'takut', 'nervous', 'anxiety', 'deg-degan', 'jantung berdebar', 'keringat dingin', 'tegang', 'was-was', 'overthinking', 'kecemasan', 'gangguan kecemasan'],
            'depression': ['depresi', 'sedih', 'murung', 'putus asa', 'hopeless', 'down', 'tidak bersemangat', 'malas', 'lelah', 'capek', 'kosong', 'hampa', 'tidak ada motivasi', 'kehilangan minat', 'bad mood'],
            'stress': ['stres', 'stress', 'tekanan', 'beban', 'pusing', 'overwhelmed', 'kewalahan', 'terbebani', 'lelah mental', 'burnout', 'jenuh', 'frustrasi', 'terpuruk', 'tertekan'],
            'sleep_disorder': ['tidur', 'insomnia', 'susah tidur', 'tidak bisa tidur', 'begadang', 'mimpi buruk', 'nightmare', 'bangun malam', 'gelisah tidur', 'mengigau', 'sleepwalking', 'kantuk', 'ngantuk'],
            'adhd': ['adhd', 'hiperaktif', 'hyperactive', 'sulit fokus', 'tidak bisa diam', 'impulsif', 'pelupa', 'ceroboh', 'attention deficit', 'konsentrasi', 'susah berkonsentrasi', 'mudah teralihkan'],
            'autism': ['autis', 'autism', 'spektrum autisme', 'komunikasi sosial', 'interaksi sosial', 'repetitif', 'stimming', 'sensori', 'rutinitas', 'pola perilaku'],
            'bipolar': ['bipolar', 'mood swing', 'mania', 'manik', 'euforia', 'perubahan mood', 'naik turun', 'episode', 'hypomanic'],
            'ptsd': ['trauma', 'ptsd', 'flashback', 'kenangan buruk', 'terganggu', 'kekerasan', 'pelecehan', 'abuse', 'shock', 'terpukul'],
            'eating_disorder': ['makan', 'nafsu makan', 'anoreksia', 'bulimia', 'binge eating', 'diet berlebihan', 'tidak mau makan', 'muntah', 'body image'],
            'addiction': ['kecanduan', 'adiksi', 'ketergantungan', 'narkoba', 'alkohol', 'rokok', 'game online', 'media sosial', 'gambling', 'judi'],
            'ocd': ['ocd', 'obsesi', 'kompulsi', 'ritual', 'berulang-ulang', 'tidak bisa berhenti', 'terus menerus', 'terpaksa melakukan'],
            'schizophrenia': ['skizofrenia', 'halusinasi', 'delusi', 'waham', 'mendengar suara', 'melihat sesuatu', 'paranoid', 'curiga berlebihan']
        }
        self.demographic_keywords = {
            'child': ['anak', 'balita', 'bocah', 'kecil', 'sd', 'tk'],
            'teen': ['remaja', 'abg', 'smp', 'sma', 'teenager', 'adolescent'],
            'adult': ['dewasa', 'kuliah', 'kerja', 'karir', 'menikah'],
            'elderly': ['lansia', 'tua', 'lanjut usia', 'pensiunan']
        }
        self.severity_keywords = {
            'mild': ['ringan', 'sedikit', 'agak', 'kadang-kadang', 'sesekali'],
            'moderate': ['sedang', 'cukup', 'lumayan', 'sering'],
            'severe': ['parah', 'berat', 'sangat', 'ekstrem', 'selalu', 'terus menerus']
        }
        self.label_keywords = label_keywords_stored
        self.labels_ordered = sorted(self.label_keywords.keys())

    def extract_entities(self, text: str) -> Dict:
        """Ekstraksi entitas dari teks."""
        text_lower = text.lower()
        entities = {'mental_health': [], 'demographic': [], 'severity': []}
        for cat, words in self.mental_health_keywords.items():
            entities['mental_health'].extend([(cat, w) for w in words if w in text_lower])
        for cat, words in self.demographic_keywords.items():
            entities['demographic'].extend([(cat, w) for w in words if w in text_lower])
        for cat, words in self.severity_keywords.items():
            entities['severity'].extend([(cat, w) for w in words if w in text_lower])
        return entities

    def create_ner_features(self, text: str) -> np.ndarray:
        """Buat fitur NER untuk satu teks input."""
        entities = self.extract_entities(text)
        feature_vector = []
        text_lower = text.lower()

        # 1. Entity Counts
        mh_cats = list(self.mental_health_keywords.keys())
        demo_cats = list(self.demographic_keywords.keys())
        sev_cats = list(self.severity_keywords.keys())

        counts = defaultdict(int)
        for category, _ in entities['mental_health']: counts[category] += 1
        for category in mh_cats: feature_vector.append(counts[category])

        counts = defaultdict(int)
        for category, _ in entities['demographic']: counts[category] += 1
        for category in demo_cats: feature_vector.append(counts[category])

        counts = defaultdict(int)
        for category, _ in entities['severity']: counts[category] += 1
        for category in sev_cats: feature_vector.append(counts[category])

        # 2. Keyword Match Scores
        for label in self.labels_ordered:
            keywords = self.label_keywords.get(label, set())
            match_score = sum(1 for kw in keywords if kw in text_lower) / max(len(keywords), 1)
            feature_vector.append(match_score)

        return np.array(feature_vector)

class IndoBERTFeatureExtractorPredict:
    """IndoBERTFeatureExtractor untuk prediksi tunggal."""
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer, device: torch.device, max_length: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model.eval()

    def extract_features(self, text: str) -> np.ndarray:
        """Ekstraksi embedding [CLS] dari IndoBERT (dinormalisasi) untuk satu teks."""
        encoded = self.tokenizer(
            text, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            features = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

        return normalize(features.reshape(1, -1)).flatten()

# --- MODEL CHATBOT UTAMA ---
class MentalHealthChatbot:
    def __init__(self):
        print("Initializing MentalHealthChatbot components...")

        # 1. Load Model Klasifikasi (GB) dan LabelEncoder
        try:
            with open(PKL_MODEL_PATH, "rb") as f:
                model_data = pickle.load(f)
            self.gb_model = model_data['model']
            self.le: LabelEncoder = model_data['label_encoder']
            print(f"✅ Gradient Boosting Model dan LabelEncoder dimuat.")
        except Exception as e:
            print(f"❌ Gagal memuat {PKL_MODEL_PATH}: {e}", file=sys.stderr)
            raise

        # 2. Load KeywordNER Data dan IndoBERT Embeddings yang Tersimpan
        try:
            with open(PKL_COMPONENTS_PATH, "rb") as f:
                components = pickle.load(f)
            self.stored_bert_features = components['features_bert']
            print(f"✅ Komponen Preprocessing dimuat.")
        except Exception as e:
            print(f"❌ Gagal memuat {PKL_COMPONENTS_PATH}: {e}", file=sys.stderr)
            raise

        # 3. Load IndoBERT (Hanya untuk menghitung embedding pertanyaan baru)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            bert_model = AutoModel.from_pretrained(MODEL_NAME)
            bert_model.to(DEVICE)
            self.bert_extractor = IndoBERTFeatureExtractorPredict(bert_model, self.tokenizer, DEVICE)
            print(f"✅ IndoBERT Model dimuat ke {DEVICE}.")
        except Exception as e:
            print(f"❌ Gagal memuat IndoBERT: {e}", file=sys.stderr)
            raise

        # 4. Inisialisasi NER Predictor
        self.ner_predictor = KeywordNERPredict(components['label_keywords'])

        # 5. Load Dataset untuk Retrieval Jawaban
        try:
            self.df = pd.read_csv(DF_PATH)
            self.df["Pertanyaan_Clean"] = self.df["Pertanyaan"].apply(minimal_preprocessing)
            self.df["Jawaban_Clean"] = self.df["Jawaban"].apply(minimal_preprocessing)
            self.df = self.df.drop_duplicates(subset=["Pertanyaan_Clean"], keep="first").reset_index(drop=True)

            self.df['feature'] = [self.stored_bert_features[i] for i in self.df.index]

            self.retrieval_data = defaultdict(list)
            for i, row in self.df.iterrows():
                label_encoded = self.le.transform([row['Label']])[0]
                label = self.le.inverse_transform([label_encoded])[0]
                self.retrieval_data[label].append({
                    'question': row['Pertanyaan_Clean'],
                    'answer': row['Jawaban_Clean'],
                    'feature': row['feature']
                })

            print(f"✅ Data Retrieval dimuat. Siap melayani {len(self.le.classes_)} kategori.")
        except Exception as e:
            print(f"❌ Gagal memuat data retrieval dari {DF_PATH}: {e}", file=sys.stderr)
            raise

        print("🚀 Chatbot successfully initialized.")


    def get_combined_features(self, text: str) -> np.ndarray:
        """Ekstraksi dan gabungan fitur (BERT + NER) untuk prediksi klasifikasi."""
        bert_features = self.bert_extractor.extract_features(text)
        ner_features = self.ner_predictor.create_ner_features(text)
        return np.concatenate([bert_features, ner_features]).reshape(1, -1)

    def predict_and_respond(self, user_question: str) -> Tuple[str, str, str, float]:
        """
        Melakukan klasifikasi, retrieval jawaban, dan mengembalikan respons.
        """
        clean_question = minimal_preprocessing(user_question)
        if not clean_question:
            return "Maaf, pertanyaan Anda kosong.", "N/A", "N/A", 0.0

        # 2. Feature Extraction & Klasifikasi
        X_pred = self.get_combined_features(clean_question)
        predicted_label_idx = self.gb_model.predict(X_pred)[0]
        predicted_label = self.le.inverse_transform([predicted_label_idx])[0]

        # 3. Retrieval Jawaban (Cosine Similarity)
        if predicted_label not in self.retrieval_data:
            return "Maaf, kategori ini belum memiliki data retrieval.", predicted_label, "N/A", 0.0

        category_data = self.retrieval_data[predicted_label]
        user_bert_feature = self.bert_extractor.extract_features(clean_question)
        target_features = np.array([item['feature'] for item in category_data])

        similarities = cosine_similarity(user_bert_feature.reshape(1, -1), target_features)

        best_match_index = np.argmax(similarities)
        best_similarity_score = similarities[0, best_match_index]

        best_match_item = category_data[best_match_index]
        matched_question = best_match_item['question']
        final_answer = best_match_item['answer']

        # 4. Penerapan Batas (Threshold)
        if best_similarity_score < SIMILARITY_THRESHOLD:
            final_answer = (
                f"Maaf, pertanyaan Anda memiliki kompleksitas unik yang **sangat berbeda** dari data yang kami miliki (Skor Kecocokan: {best_similarity_score:.2f}). "
                "Kami sangat menyarankan Anda untuk **berkonsultasi langsung dengan profesional kesehatan mental (psikolog/psikiater)** untuk mendapatkan penanganan dan diagnosis yang akurat. Keselamatan Anda adalah prioritas utama kami."
            )
            matched_question = "Rekomendasi Rujukan Profesional"

        return final_answer, predicted_label, matched_question, best_similarity_score

# --- LOAD CHATBOT & JOURNALING MODELS ---
chatbot: MentalHealthChatbot = None
mh_model = None

try:
    chatbot = MentalHealthChatbot()
except Exception:
    print("FATAL: Chatbot failed to load. API endpoints will fail.", file=sys.stderr)

try:
    # Load Model Deteksi Mental Health (Journaling)
    mh_model = joblib.load(MENTAL_HEALTH_MODEL_PATH)
    print("✅ Model deteksi mental health (mental_health_final.joblib) berhasil dimuat.")
except Exception as e:
    print(f"⚠️ Gagal memuat mental_health_final.joblib: {e}. Journaling endpoint akan non-fungsional.", file=sys.stderr)


# --- DATA MOTIVASI & PESAN PANDUAN ---
motivational_quotes = [
    "Kamu hebat sudah bisa sampai di titik ini, jangan menyerah ya!",
    "Hari ini mungkin berat, tapi percayalah badai pasti berlalu.",
    "Setiap langkah kecilmu tetap berarti, terus maju meskipun pelan.",
    "Tidak apa-apa merasa lelah, istirahat sebentar lalu bangkit lagi.",
    "Kamu lebih kuat dari yang kamu kira.",
    "Ingat, setiap pagi adalah kesempatan baru untuk mencoba lagi.",
    "Hidup tidak harus sempurna untuk tetap indah.",
    "Terima dirimu apa adanya, kamu berharga.",
    "Kesalahan bukan akhir dari segalanya, tapi awal dari pembelajaran.",
    "Kamu pantas mendapatkan kebahagiaan, jadi jangan berhenti berjuang."
]

phq9_message = (
    "Terima kasih sudah berbagi perasaanmu 💙. "
    "Dari isi jurnalmu, kami mendeteksi adanya tanda-tanda yang mungkin mengarah pada gangguan kesehatan mental. "
    "Kami menyarankanmu untuk melanjutkan dengan mengisi <b>kuesioner PHQ-9</b> agar dapat membantu memahami kondisimu lebih baik. "
    "Tenang, ini bukan diagnosis — hanya langkah awal untuk mengenal dirimu lebih dalam 😊."
)

# --- MODELS UNTUK FASTAPI ---
class UserQuery(BaseModel):
    query: str

class JournalEntry(BaseModel):
    text: str

# --- ENDPOINT CHATBOT ---
@app.get("/")
def home():
    return {"message": "Mental Health Chatbot & Journaling API is running. Send POST to /predict (chatbot) or /journal (analysis)."}

@app.post("/predict")
def predict_response(query: UserQuery):
    """Endpoint chatbot utama (Klasifikasi + Retrieval)"""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot is not initialized. Check server logs.")

    try:
        response, label, matched_q, score = chatbot.predict_and_respond(query.query)
        return {
            "user_query": query.query,
            "predicted_label": label,
            "matched_question_ref": matched_q,
            "cosine_similarity_score": float(round(score, 4)),
            "bot_response": response
        }
    except Exception as e:
        print(f"Error during prediction: {str(e)}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Internal Server Error during prediction: {str(e)}")

# ----------------------------------------------------
# --- ENDPOINT UTAMA JOURNALING (TEXT INPUT) ---
# ----------------------------------------------------
@app.post("/journal")
def analyze_journal(entry: JournalEntry):
    """
    Endpoint untuk menganalisis teks journaling pengguna. (Menggunakan pipeline manual BERT + Classifier)
    """
    try:
        if mh_model is None:
            raise HTTPException(status_code=503, detail="Model mental health (journaling) belum dimuat. Periksa log server.")

        text_clean = minimal_preprocessing(entry.text)

        # --- Pipeline Manual Sesuai Permintaan (BERT Feature Extraction + GB Classifier) ---
        if isinstance(mh_model, dict):
            # Model dimuat sebagai dictionary. Ambil classifier dan encoder.
            try:
                classifier = mh_model['gb_classifier']
                label_encoder = mh_model['label_encoder']
            except KeyError as ke:
                raise HTTPException(status_code=500, detail=f"Model Journaling (dict) tidak memiliki kunci yang diperlukan: {ke}. Model mungkin rusak.")

            # Re-initialize BERT for this specific pipeline (indobenchmark/indobert-base-p1)
            # Karena MODEL_NAME_MH sekarang didefinisikan di atas, ini akan berfungsi.
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_MH)
            bert_model = AutoModel.from_pretrained(MODEL_NAME_MH)
            bert_model.to(DEVICE)
            bert_model.eval()

            encoded = tokenizer(
                [text_clean],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(DEVICE)

            with torch.no_grad():
                outputs = bert_model(**encoded)
                features = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # Prediksi label dari GradientBoostingClassifier
            pred_idx = classifier.predict(features)[0]
            prediction = (
                label_encoder.inverse_transform([pred_idx])[0]
                if label_encoder is not None
                else str(pred_idx)
            )

        elif hasattr(mh_model, 'predict'):
            # Fallback jika model adalah pipeline scikit-learn tunggal (bukan dict)
            prediction = mh_model.predict([text_clean])[0]
        
        else:
            # Jika model dimuat tetapi tidak valid
            raise HTTPException(status_code=500, detail=f"Objek model mental health tidak valid (bukan dict atau tidak memiliki .predict()): {type(mh_model)}")

        # --------------------------------------------------------------------------

        # --- Buat pesan hasil ---
        if prediction.lower() == "normal":
            message = random.choice(motivational_quotes)
        else:
            message = phq9_message

        return {
            "journal_text": entry.text,
            "prediction": prediction,
            "response_message": message
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error analyzing journal: {str(e)}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Kesalahan internal server saat menganalisis jurnal: {str(e)}")

# ----------------------------------------------------
# --- ENDPOINT UNTUK JOURNALING DARI AUDIO (STT) ---
# ----------------------------------------------------
@app.post("/journal/audio")
async def analyze_journal_audio(file: UploadFile = File(...)):
    """
    Endpoint untuk menerima file audio (WAV) dan menganalisis konten jurnal setelah STT.
    """
    try:
        if not file.filename.lower().endswith('.wav'):
             raise HTTPException(status_code=400, detail="Format file harus WAV.")
        
        audio_bytes = await file.read()
        
        if not audio_bytes:
             raise HTTPException(status_code=400, detail="File audio kosong.")

        # 1. Konversi Audio ke Teks (Speech-to-Text)
        transcribed_text = transcribe_audio_bytes(audio_bytes)
        
        if not transcribed_text:
             return {
                 "journal_text": "",
                 "prediction": "N/A",
                 "response_message": "Analisis Gagal! Maaf, tidak ada ucapan yang terdeteksi dalam audio.",
                 "stt_status": "No Speech Detected"
             }

        # 2. Analisis Teks (Menggunakan fungsi analyze_journal secara internal)
        analysis_result = analyze_journal(JournalEntry(text=transcribed_text))

        # 3. Kembalikan Hasil Gabungan
        return {
            "journal_text": transcribed_text,
            "prediction": analysis_result["prediction"],
            "response_message": analysis_result["response_message"],
            "stt_status": "Success"
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error processing audio journal: {str(e)}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Internal Server Error during audio processing: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)