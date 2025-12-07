"""
💎 Toxic Comment Detection System V2.1
Aplikasi Modern untuk Komunitas Gaming Indonesia
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import time
from datetime import datetime, date, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import UI_CONFIG, LABELS, DATASET_PATH, MODELS_DIR, SVM_MODEL_PATH
from modules.preprocessing import TextPreprocessor
from modules.training import ModelTrainer
from modules.predictor import ToxicityPredictor
from modules.scraper import ScraperFactory

# --- Page Config & Setup ---
st.set_page_config(
    page_title="ToxicShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Init ---
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Home"
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = None

# --- Custom Premium CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* General Settings */
    * {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(15, 23, 42) 0%, rgb(18, 18, 23) 90%);
        color: #ffffff;
    }
    
    /* Hide Default Elements */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {width: 8px;}
    ::-webkit-scrollbar-track {background: #1e1e24;}
    ::-webkit-scrollbar-thumb {background: #4f46e5; border-radius: 4px;}
    
    /* Hero Section */
    .hero-box {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.1), rgba(124, 58, 237, 0.1));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 3rem;
        text-align: center;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.2rem;
        font-weight: 300;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Input Area Styling */
    .stTextArea textarea, .stTextInput input {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        padding: 1rem !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        text-transform: uppercase;
        font-size: 0.9rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.4);
    }
    
    /* Result Cards */
    .prediction-card {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
        margin-top: 1rem;
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .badge-safe {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #059669;
    }
    
    .badge-toxic {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #dc2626;
    }
    
    /* Progress Bars */
    .prob-bar-bg {
        background: rgba(255, 255, 255, 0.1);
        height: 8px;
        border-radius: 4px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    
    .prob-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 1s ease-out;
    }
    
    /* MetricsGrid */
    .metric-box {
        background: rgba(15, 23, 42, 0.6);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
        transition: transform 0.3s;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #818cf8;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        color: #818cf8;
        border-bottom: 2px solid #818cf8;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def load_predictor():
    """Load the model securely."""
    if not st.session_state.model_loaded:
        try:
            st.session_state.predictor = ToxicityPredictor()
            if st.session_state.predictor.is_loaded:
                st.session_state.model_loaded = True
                return True
        except Exception:
            return False
    return st.session_state.model_loaded

def get_training_results():
    results_path = os.path.join(MODELS_DIR, 'training_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None

def batch_predict(df):
    """Predict labels for scraping results."""
    if not load_predictor():
        st.error("Model Error")
        return df
    
    predictions = []
    confidences = []
    
    with st.spinner("Analyzing comments..."):
        progress_bar = st.progress(0)
        total = len(df)
        
        for i, text in enumerate(df['content']):
            res = st.session_state.predictor.predict(str(text))
            predictions.append(res['label_name']) # "Aman" or "Toxic" based on threshold
            confidences.append(f"{res['confidence']*100:.1f}%")
            progress_bar.progress((i + 1) / total)
            
    df['Prediction'] = predictions
    df['Confidence'] = confidences
    return df

# --- UI Components ---

def render_sidebar():
    with st.sidebar:
        st.markdown("### 🎛️ Control Center")
        
        # Status Card
        model_ok = os.path.exists(SVM_MODEL_PATH)
        status_color = "#34d399" if model_ok else "#f87171"
        status_text = "SYSTEM ONLINE" if model_ok else "MODEL MISSING"
        
        st.markdown(f"""
        <div style="background: rgba(30,41,59,0.5); padding: 1rem; border-radius: 12px; border-left: 4px solid {status_color}; margin-bottom: 2rem;">
            <div style="color: #94a3b8; font-size: 0.8rem;">ENGINE STATUS</div>
            <div style="color: white; font-weight: 600;">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Training Actions
        st.markdown("#### 🧠 AI Training")
        if st.button("RETRAIN MODEL", use_container_width=True):
            with st.status("🚀 Training Neural Engine...", expanded=True) as status:
                st.write("📂 Loading dataset...")
                time.sleep(0.5)
                trainer = ModelTrainer()
                st.write("⚙️ Vectorizing features...")
                time.sleep(0.5)
                st.write("🧠 Optimizing Hyperplanes...")
                results = trainer.train()
                st.session_state.model_loaded = False 
                st.write("✅ Model Saved.")
                status.update(label="Training Complete!", state="complete", expanded=False)
                time.sleep(1)
                st.rerun()
                
        st.markdown("---")
        st.markdown("#### 📊 Dataset Info")
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            st.caption(f"Total Samples: {len(df):,}")
            st.caption(f"Toxic Ratio: {(len(df[df.label==1])/len(df)*100):.1f}%")

def render_hero():
    st.markdown("""
    <div class="hero-box">
        <div class="hero-title">ToxicShield AI</div>
        <div class="hero-subtitle">
            Advanced Toxic Comment Detection for Roblox Indonesia.<br>
            Powered by Support Vector Machines & NLP.
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_home():
    render_hero()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Analysis Console")
        input_text = st.text_area(
            "Input Text",
            placeholder="Ketik komentar di sini... (Contoh: 'dasar nubs lu')",
            label_visibility="collapsed",
            height=150
        )
        
        c1, c2 = st.columns([1, 1])
        with c1:
            analyze_btn = st.button("🔍 ANALYZE CONTENT", use_container_width=True)
        with c2:
            show_details = st.checkbox("Show Neural Logic", value=False)
    
        if analyze_btn and input_text:
            if not load_predictor():
                st.error("⚠️ Model belum siap. Silakan train model di sidebar.")
            else:
                with st.spinner("Processing..."):
                    time.sleep(0.3)
                    result = st.session_state.predictor.predict(input_text, return_details=True)
                    st.session_state.last_prediction = result
    
        if st.session_state.last_prediction:
            res = st.session_state.last_prediction
            is_toxic = res['label'] == 1
            
            badge_class = "badge-toxic" if is_toxic else "badge-safe"
            badge_text = "DETECTED: TOXIC" if is_toxic else "DETECTED: SAFE"
            emoji = "⚠️" if is_toxic else "✅"
            
            st.markdown(f"""
            <div class="prediction-card">
                <div class="status-badge {badge_class}">{emoji} {badge_text}</div>
                <h3 style="margin:0">Confidence Score</h3>
                <h1 style="font-size: 3.5rem; margin: 0.5rem 0;">{res['confidence']*100:.1f}%</h1>
                <p style="color: #94a3b8;">Probability of being {res['label_name']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Probability Distribution")
            prob_safe = res['probabilities']['aman'] * 100
            prob_toxic = res['probabilities']['toxic'] * 100
            
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">
                    <span style="color:#34d399">Total Safe</span>
                    <span style="color:#34d399">{prob_safe:.1f}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{prob_safe}%; background:#34d399;"></div>
                </div>
            </div>
            
            <div>
                <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">
                    <span style="color:#f87171">Toxic Content</span>
                    <span style="color:#f87171">{prob_toxic:.1f}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{prob_toxic}%; background:#f87171;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if show_details:
                st.markdown("### 🔧 Neural Processing Steps")
                steps = res['preprocessing_steps']
                with st.expander("View Pipeline Details", expanded=True):
                    for step, text in steps.items():
                        if step != 'original':
                            st.markdown(f"**{step.replace('after_', '').title()}**")
                            st.code(text)

    with col2:
        st.markdown("### ⚡ Quick Test")
        examples = [
            ("Main bareng yuk", "safe"),
            ("Dasar nubs bego", "toxic"),
            ("Gg wp mantap", "safe"), 
            ("Anjing lo sampah", "toxic")
        ]
        
        for text, type_ in examples:
            emoji = "✅" if type_ == "safe" else "☢️"
            if st.button(f"{emoji} {text}", key=text, use_container_width=True):
                if load_predictor():
                    res = st.session_state.predictor.predict(text, return_details=True)
                    st.session_state.last_prediction = res
                    st.rerun()

def render_scraping():
    st.markdown("## 🕷️ Web Scraping Engine")
    st.write("Gather real-time data from social platforms for analysis.")
    
    col_input, col_params = st.columns(2)
    
    with col_input:
        st.markdown("#### 1. Select Source")
        platform = st.radio(
            "Platform",
            ["Google Play", "Twitter (X)"],
            captions=["API-based (Fast & Reliable)", "Selenium-based (Experimental)"]
        )
        
        keyword = ""
        if platform == "Google Play":
            keyword = st.text_input("App ID / Package Name", value="com.roblox.client")
            st.caption("Example: com.roblox.client")
        else:
            keyword = st.text_input("Search Keyword", value="Roblox Indonesia")
            
            with st.expander("Authentication (Required for Twitter)", expanded=True):
                username = st.text_input("Username/Email")
                password = st.text_input("Password", type="password")
                
    with col_params:
        st.markdown("#### 2. Scraping Parameters")
        
        limit = st.slider("Limit Data", 10, 500, 50)
        
        d1 = st.date_input("Start Date", date.today() - timedelta(days=7))
        d2 = st.date_input("End Date", date.today())
        
        st.info("💡 Note: Twitter filtering logic stops scraping once posts older than Start Date are found.")

    start_scrape = st.button("🚀 START SCRAPING", use_container_width=True)
    
    if start_scrape:
        scraper = ScraperFactory.get_scraper(platform)
        if not scraper:
            st.error("Platform not supported yet.")
        else:
            with st.status(f"Running {platform} Scraper...", expanded=True) as status:
                try:
                    st.write("🔌 Connecting to engine...")
                    
                    if platform == "Twitter":
                        if not username or not password:
                            st.error("Twitter requires login credentials.")
                            status.update(label="Failed", state="error")
                            return
                        
                        st.write("🔑 Authenticating...")
                        if not scraper.login(username, password):
                            st.error("Login failed. Check credentials or captcha.")
                            status.update(label="Login Failed", state="error")
                            return
                    
                    st.write(f"🔎 Fetching data for '{keyword}'...")
                    df = scraper.scrape(
                        keyword, 
                        limit=limit,
                        start_date=d1,
                        end_date=d2
                    )
                    
                    if not df.empty:
                        st.session_state.scraped_data = df
                        st.success(f"Successfully scraped {len(df)} items!")
                        status.update(label="Scraping Complete", state="complete")
                    else:
                        st.warning("No data found matching criteria.")
                        status.update(label="No Data", state="complete")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    status.update(label="Error Occurred", state="error")
    
    # Results Section
    if st.session_state.scraped_data is not None:
        st.markdown("---")
        st.markdown("#### 📝 Scraped Data Preview")
        
        df_show = st.session_state.scraped_data
        st.dataframe(df_show, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            csv = df_show.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV",
                csv,
                "scraped_data.csv",
                "text/csv",
                use_container_width=True
            )
        
        with c2:
            if st.button("🤖 PREDICT WITH AI (Batch Analysis)", use_container_width=True):
                result_df = batch_predict(df_show.copy())
                st.session_state.scraped_data = result_df # Update with predictions
                st.rerun()

def render_metrics():
    st.markdown("## 📊 Model Performance")
    results = get_training_results()
    
    if results:
        m = results['metrics']
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{m['accuracy']*100:.1f}%</div>
                <div class="metric-label">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{m['precision']*100:.1f}%</div>
                <div class="metric-label">Precision</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{m['recall']*100:.1f}%</div>
                <div class="metric-label">Recall</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
             st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{m['f1_score']*100:.1f}%</div>
                <div class="metric-label">F1-Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📈 Confusion Matrix")
        cm = results['confusion_matrix']
        cm_df = pd.DataFrame(
            cm, 
            columns=['Pred: Safe', 'Pred: Toxic'],
            index=['Actual: Safe', 'Actual: Toxic']
        )
        st.dataframe(cm_df, use_container_width=True)
        
        st.markdown("### 📝 Classification Report")
        st.json(results['classification_report'])
        
    else:
        st.info("No training results found. Please train the model.")

# --- App Entry Point ---

def main():
    render_sidebar()
    
    tab_home, tab_scraping, tab_metrics, tab_guide = st.tabs(["🚀 DETECTOR", "🕷️ SCRAPING", "📊 METRICS", "📚 GUIDE"])
    
    with tab_home:
        render_home()
        
    with tab_scraping:
        render_scraping()
    
    with tab_metrics:
        render_metrics()
        
    with tab_guide:
        st.markdown("""
        ### How to use ToxicShield AI
        
        **1. Single Detection**
        - Type text in the **DETECTOR** tab and click Analyze.
        
        **2. Web Scraping**
        - Go to **SCRAPING** tab.
        - Choose 'Google Play' for app reviews or 'Twitter' for tweets.
        - Set parameters and click 'Start Scraping'.
        - Once data is loaded, click 'Predict with AI' to classify all comments.
        
        **3. Training**
        - Use the sidebar 'Retrain Model' button to update the AI with new data.
        """)

if __name__ == "__main__":
    main()
