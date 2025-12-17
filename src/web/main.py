from fasthtml.common import *
import tensorflow as tf
import pickle
import numpy as np
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deep_translator import GoogleTranslator
import os
import re
import string
import markdown
import csv
from datetime import datetime

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FEEDBACK_FILE = os.path.join(BASE_DIR, 'data', 'feedback', 'user_feedback.csv')

# Asegurar que el archivo CSV exista con cabeceras
os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'text', 'model_prediction', 'user_correction'])

# Variables globales
current_model = None
tokenizer = None
current_model_name = ""
text_language = "es"  # Idioma del texto a analizar (no de la UI)

def load_resources(model_name="Exp1_Base_LSTM.keras"):
    global current_model, tokenizer, current_model_name
    if tokenizer is None:
        tok_path = os.path.join(MODELS_DIR, 'tokenizer.pkl')
        with open(tok_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
    if current_model is None or current_model_name != model_name:
        model_path = os.path.join(MODELS_DIR, model_name)
        current_model = tf.keras.models.load_model(model_path)
        current_model_name = model_name

# --- LÓGICA DE NEGOCIO ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

def get_prediction(text, lang="es"):
    """
    Predice si una noticia es FAKE o REAL.
    Si lang='es', traduce a inglés primero.
    Si lang='en', usa el texto directamente.
    """
    load_resources(current_model_name if current_model_name else "Exp1_Base_LSTM.keras")
    
    # Solo traducir si el texto está en español
    if lang == "es":
        try:
            translated = GoogleTranslator(source='es', target='en').translate(text[:2000])
        except:
            translated = text
    else:
        translated = text  # Ya está en inglés, no traducir

    cleaned = clean_text(translated)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=250, padding='post', truncating='post')
    
    pred_prob = current_model.predict(padded)[0][0]
    is_fake = pred_prob > 0.85
    confidence = pred_prob * 100 if is_fake else (1 - pred_prob) * 100
    label = "FAKE" if is_fake else "REAL"
    
    return label, confidence, translated

def scrape_article(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('h1').get_text().strip() if soup.find('h1') else "Sin título"
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        return title, text
    except Exception as e:
        return None, str(e)

# --- FASTHTML APP ---
app = FastHTML(hdrs=(picolink,))

# Estilos modernos y minimalistas
style_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-main: #0a0a0b;
        --bg-panel: #111113;
        --bg-card: #18181b;
        --bg-card-hover: #1f1f23;
        --glass: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.06);
        --accent-primary: #6366f1;
        --accent-primary-soft: rgba(99, 102, 241, 0.15);
        --accent-secondary: #8b5cf6;
        --accent-success: #22c55e;
        --accent-success-soft: rgba(34, 197, 94, 0.15);
        --accent-danger: #ef4444;
        --accent-danger-soft: rgba(239, 68, 68, 0.15);
        --accent-warning: #f59e0b;
        --text-main: #fafafa;
        --text-secondary: #a1a1aa;
        --text-muted: #71717a;
        --border-color: rgba(255, 255, 255, 0.08);
        --border-hover: rgba(255, 255, 255, 0.15);
        --sidebar-width: 340px;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
        --shadow-md: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    body { 
        background-color: var(--bg-main);
        color: var(--text-main); 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        line-height: 1.6;
        letter-spacing: -0.01em;
        -webkit-font-smoothing: antialiased;
    }
    
    .layout-container { 
        display: grid; 
        grid-template-columns: var(--sidebar-width) 1fr; 
        min-height: 100vh;
    }
    
    .sidebar { 
        background: linear-gradient(180deg, var(--bg-panel) 0%, var(--bg-main) 100%);
        padding: 1.5rem; 
        border-right: 1px solid var(--border-color);
        display: flex; 
        flex-direction: column; 
        gap: 1rem;
        position: sticky;
        top: 0;
        height: 100vh;
        overflow-y: auto;
    }
    
    .sidebar::-webkit-scrollbar { width: 4px; }
    .sidebar::-webkit-scrollbar-track { background: transparent; }
    .sidebar::-webkit-scrollbar-thumb { 
        background: var(--accent-primary); 
        border-radius: 4px; 
    }
    
    .content-area { 
        padding: 3rem; 
        background: radial-gradient(ellipse at top, rgba(99, 102, 241, 0.03) 0%, transparent 50%);
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    h2.sidebar-title {
        color: var(--text-main);
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .section-card {
        background: var(--glass);
        backdrop-filter: blur(10px);
        padding: 1.25rem;
        border-radius: var(--radius-md);
        border: 1px solid var(--glass-border);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .section-card:hover {
        background: var(--bg-card-hover);
        border-color: var(--border-hover);
        transform: translateY(-2px);
        box-shadow: var(--shadow-sm);
    }
    
    label { 
        color: var(--text-secondary); 
        font-weight: 500; 
        font-size: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    input, textarea, select { 
        background: var(--bg-main) !important; 
        border: 1px solid var(--border-color) !important; 
        color: var(--text-main) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.7rem 0.9rem !important;
        width: 100%;
        transition: all 0.2s ease;
        font-size: 0.9rem;
        font-family: inherit;
    }
    
    input:focus, textarea:focus, select:focus {
        border-color: var(--accent-primary) !important;
        outline: none;
        box-shadow: 0 0 0 3px var(--accent-primary-soft);
    }
    
    input::placeholder, textarea::placeholder {
        color: var(--text-muted);
    }
    
    button { 
        padding: 0.7rem 1.2rem;
        border-radius: var(--radius-sm);
        font-weight: 500;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        border: none;
        font-family: inherit;
        letter-spacing: -0.01em;
    }
    
    button.contrast { 
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        color: white;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    }
    
    button.contrast:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45);
    }
    
    button.secondary { 
        background: var(--bg-card);
        color: var(--text-main);
        border: 1px solid var(--border-color);
    }
    
    button.secondary:hover {
        background: var(--bg-card-hover);
        border-color: var(--border-hover);
        transform: translateY(-1px);
    }
    
    .divider {
        text-align: center;
        color: var(--text-muted);
        font-weight: 500;
        margin: 0.75rem 0;
        position: relative;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .divider::before, .divider::after {
        content: '';
        position: absolute;
        top: 50%;
        width: 40%;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
    }
    
    .divider::before { left: 0; }
    .divider::after { right: 0; }
    
    #model-status, #language-status {
        background: var(--accent-primary-soft);
        padding: 0.5rem 0.75rem;
        border-radius: var(--radius-sm);
        font-size: 0.75rem;
        color: var(--accent-primary);
        text-align: center;
        font-weight: 500;
        border: 1px solid rgba(99, 102, 241, 0.2);
        margin-top: 0.5rem;
    }
    
    .feedback-area { 
        margin-top: 1.5rem; 
        padding: 1.5rem;
        background: var(--glass);
        backdrop-filter: blur(10px);
        border-radius: var(--radius-md);
        border: 1px solid var(--glass-border);
        text-align: center;
    }
    
    .feedback-btn { 
        margin: 0.4rem; 
        padding: 0.6rem 1.5rem; 
        font-size: 0.85rem;
        cursor: pointer;
        border-radius: var(--radius-sm);
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .btn-real { 
        background: var(--accent-success-soft);
        color: var(--accent-success);
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .btn-real:hover {
        background: var(--accent-success);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(34, 197, 94, 0.35);
    }
    
    .btn-fake { 
        background: var(--accent-danger-soft);
        color: var(--accent-danger);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .btn-fake:hover {
        background: var(--accent-danger);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.35);
    }
    
    .status-card { 
        padding: 1.5rem 2rem; 
        border-radius: var(--radius-lg);
        color: white; 
        font-weight: 600;
        text-align: center; 
        margin: 1.5rem 0;
        font-size: 1.25rem;
        animation: slideUp 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: -0.01em;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .bg-real { 
        background: linear-gradient(135deg, #16a34a, #22c55e);
        box-shadow: 0 8px 32px rgba(34, 197, 94, 0.3);
    }
    
    .bg-fake { 
        background: linear-gradient(135deg, #dc2626, #ef4444);
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.3);
    }
    
    .md-result {
        background: var(--bg-card);
        padding: 2rem;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-color);
        animation: fadeIn 0.4s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .md-result h1 { 
        color: var(--text-main); 
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    .md-result h3 {
        color: var(--accent-primary);
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        font-size: 1rem;
        font-weight: 500;
    }
    
    .md-result blockquote { 
        border-left: 2px solid var(--accent-secondary);
        background: var(--glass);
        padding: 1rem 1.25rem;
        color: var(--text-secondary);
        font-style: italic;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    
    .md-result p {
        line-height: 1.7;
        margin-bottom: 1rem;
        color: var(--text-secondary);
    }
    
    .hero-section {
        text-align: center;
        padding: 3rem 2rem;
        background: var(--glass);
        backdrop-filter: blur(20px);
        border-radius: var(--radius-lg);
        border: 1px solid var(--glass-border);
        max-width: 600px;
        animation: fadeIn 0.6s ease;
    }
    
    .hero-section h1 {
        font-size: 1.75rem;
        color: var(--text-main);
        margin-bottom: 0.75rem;
        font-weight: 600;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, var(--text-main), var(--text-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-section p {
        font-size: 0.95rem;
        color: var(--text-muted);
        max-width: 450px;
        margin: 0 auto 1.5rem;
        line-height: 1.7;
    }
    
    .hero-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }
    
    .placeholder-box {
        border: 1px dashed var(--border-color);
        padding: 2rem;
        border-radius: var(--radius-md);
        text-align: center;
        margin-top: 1.5rem;
        color: var(--text-muted);
        font-size: 0.85rem;
        background: var(--glass);
    }
    
    .htmx-indicator {
        display: none;
    }
    
    .htmx-request .htmx-indicator {
        display: block;
    }
    
    #loading {
        font-size: 0.95rem;
        color: var(--accent-primary);
        text-align: center;
        padding: 1.25rem;
        background: var(--bg-card);
        border-radius: var(--radius-md);
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
    }
    
    #loading::before {
        content: '';
        width: 18px;
        height: 18px;
        border: 2px solid var(--accent-primary-soft);
        border-top-color: var(--accent-primary);
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 1rem 0;
    }
    
    .lang-selector-info {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.2);
        color: var(--accent-warning);
        padding: 0.6rem 0.8rem;
        border-radius: var(--radius-sm);
        font-size: 0.75rem;
        margin-top: 0.5rem;
        line-height: 1.5;
    }
    
    .w-full { width: 100%; }
    .mb-2 { margin-bottom: 0.5rem; }
    
    @media (max-width: 1024px) {
        .layout-container {
            grid-template-columns: 1fr;
        }
        .sidebar {
            position: relative;
            height: auto;
            border-right: none;
            border-bottom: 1px solid var(--border-color);
        }
        .content-area {
            padding: 2rem 1.5rem;
        }
    }
"""

@app.route("/")
def home():
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.keras')]
    
    sidebar = Aside(
        H2("Detector de Fake News", cls="sidebar-title"),
        
        # Selector de idioma del TEXTO (no de la UI)
        Div(
            Label("🌐 Idioma del Texto a Analizar"),
            Select(
                Option("Español (se traducirá)", value="es", selected=(text_language=="es")),
                Option("Inglés (directo al modelo)", value="en", selected=(text_language=="en")),
                name="text_lang",
                hx_post="/set_text_language",
                hx_target="#language-status"
            ),
            Div(
                f"⚙️ Configurado: {'Español → Inglés' if text_language == 'es' else 'Inglés (sin traducción)'}",
                id="language-status"
            ),
            Div(
                "ℹ️ El modelo solo funciona en inglés. Si tu texto está en español, se traducirá automáticamente.",
                cls="lang-selector-info"
            ),
            cls="section-card"
        ),
        
        # Selector de modelo
        Div(
            Label("🧠 Modelo de IA"),
            Select(*[Option(m, value=m) for m in models], name="model_name", hx_post="/set_model", hx_target="#model-status"),
            Div(f"Estado: Listo", id="model-status"),
            cls="section-card"
        ),
        
        Hr(),
        
        # Opción URL
        Div(
            Label("🔗 Análisis por URL"),
            Form(
                Input(type="url", name="url", placeholder="https://ejemplo.com/noticia", required=True, cls="mb-2"),
                Button("Analizar URL", cls="w-full contrast"), 
                hx_post="/predict_url", 
                hx_target="#result-container", 
                hx_indicator="#loading"
            ),
            cls="section-card"
        ),
        
        Div("— O —", cls="divider"),
        
        # Opción Texto
        Div(
            Label("📝 Análisis por Texto"),
            Form(
                Textarea(name="text", placeholder="Pega aquí el contenido de la noticia para analizar...", rows=6, required=True, cls="mb-2"),
                Button("Analizar Texto", cls="w-full secondary"), 
                hx_post="/predict_text", 
                hx_target="#result-container", 
                hx_indicator="#loading"
            ),
            cls="section-card"
        ),
        
        cls="sidebar"
    )

    content = Main(
        Div("⚙️ Procesando análisis...", id="loading", cls="htmx-indicator"),
        Div(
            Div(
                Div("🔍", cls="hero-icon"),
                H1("Detector de Noticias Falsas con IA"),
                P("Utiliza inteligencia artificial para verificar la autenticidad de noticias. El modelo analiza patrones lingüísticos para detectar desinformación."),
                cls="hero-section"
            ),
            Div("Esperando entrada...", cls="placeholder-box"),
            id="result-container"
        ),
        cls="content-area"
    )
    
    return Title("Detector de Fake News"), Style(style_css), Div(sidebar, content, cls="layout-container")

@app.post("/set_text_language")
def set_text_language(text_lang: str):
    global text_language
    text_language = text_lang
    lang_display = "Español → Inglés" if text_lang == "es" else "Inglés (sin traducción)"
    return f"⚙️ Configurado: {lang_display}"

@app.post("/set_model")
def set_model(model_name: str):
    load_resources(model_name)
    return f"✅ Modelo activo: {model_name}"

def render_full_result(title, text, label, confidence, translated, was_translated):
    color_class = "bg-fake" if label == "FAKE" else "bg-real"
    icon = "⚠️" if label == "FAKE" else "✓"
    
    # Construir el markdown del resultado
    translation_section = ""
    if was_translated:
        translation_section = f"""
### 🔄 Traducción Aplicada (ES → EN)
> *"{translated[:300]}..."*
"""
    else:
        translation_section = """
### ℹ️ Texto Original en Inglés
> *El texto ya estaba en inglés, se procesó directamente sin traducción.*
"""
    
    md = f"""
# Análisis Completado
<div class="status-card {color_class}">
    {icon} Resultado: {label} <br>
    <span style="font-size: 1rem; opacity: 0.9;">Confianza del modelo: {confidence:.2f}%</span>
</div>
### 📰 {title}
> {text[:400]}...
{translation_section}
    """
    html_report = markdown.markdown(md)

    feedback_form = Div(
        P("¿El modelo se equivocó? Ayúdanos a mejorar:", cls="mb-2 font-bold", style="font-size: 1rem;"),
        Form(
            Input(type="hidden", name="text_content", value=text),
            Input(type="hidden", name="model_pred", value=label),
            Button("Era REAL", name="user_correction", value="REAL", cls="feedback-btn btn-real"),
            Button("Era FAKE", name="user_correction", value="FAKE", cls="feedback-btn btn-fake"),
            hx_post="/submit_feedback",
            hx_target="#feedback-response"
        ),
        id="feedback-response",
        cls="feedback-area"
    )

    return Div(
        Div(NotStr(html_report), cls="md-result"),
        feedback_form
    )

@app.post("/predict_url")
def predict_url(url: str, model_name: str = None):
    if model_name: load_resources(model_name)
    title, text = scrape_article(url)
    if not title: 
        return Div(
            f"❌ Error al obtener la URL: {text}", 
            style="color: var(--accent-danger); padding: 2rem; text-align: center; background: var(--bg-card); border-radius: 8px;"
        )
    
    full_text = title + " " + text
    was_translated = (text_language == "es")
    label, conf, trans = get_prediction(full_text, text_language)
    
    return render_full_result(title, text, label, conf, trans, was_translated)

@app.post("/predict_text")
def predict_text(text: str, model_name: str = None):
    if model_name: load_resources(model_name)
    was_translated = (text_language == "es")
    label, conf, trans = get_prediction(text, text_language)
    return render_full_result("Texto Manual", text, label, conf, trans, was_translated)

@app.post("/submit_feedback")
def submit_feedback(text_content: str, model_pred: str, user_correction: str):
    try:
        numeric_label = 1 if user_correction == "FAKE" else 0
        
        with open(FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), text_content, model_pred, numeric_label])
            
        return Div(
            H4("¡Gracias por tu aporte! 🙏"),
            P(f"Hemos registrado que esta noticia era {user_correction}. Esta información se utilizará para mejorar el modelo."),
            style="color: var(--accent-success); font-weight: 500; padding: 1.5rem;"
        )
    except Exception as e:
        return Div(f"Error: {str(e)}", style="color: var(--accent-danger)")

if __name__ == "__main__":
    try:
        load_resources()
    except: pass
    serve()