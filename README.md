# Fake News Detector 🔍

Aplicación web para detectar noticias falsas usando modelos de Deep Learning (LSTM, CNN, Dense). Utiliza inteligencia artificial para analizar patrones lingüísticos y detectar desinformación.

---

## 🚀 Quick Start con Docker

```bash
# Construir y ejecutar
docker compose up --build
```

Abre tu navegador en: **http://localhost:5001**

### Comandos Docker útiles

```bash
# Ejecutar en background
docker compose up -d

# Ver logs
docker compose logs -f

# Detener
docker compose down
```

---

## 📦 Instalación Manual

```bash
# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o: venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
python src/web/main.py
```

---

## 🧠 Modelos Disponibles

| Modelo | Arquitectura | Descripción |
|--------|--------------|-------------|
| `Exp1_Base_LSTM.keras` | LSTM | Modelo base con LSTM |
| `Exp2_Simple_Dense.keras` | Dense | Red densa simple |
| `Exp3_Complex_LSTM.keras` | LSTM | LSTM con más capas |
| `Exp4_CNN_Spatial.keras` | CNN | Convolucional para patrones espaciales |

---

## 📁 Estructura del Proyecto

```
fake_news/
├── models/              # Modelos entrenados (.keras) + tokenizer
├── src/
│   ├── web/             # Interfaz web FastHTML
│   ├── model/           # Arquitecturas y entrenamiento
│   └── features/        # Preprocesamiento
├── data/
│   ├── raw/             # Datasets originales
│   ├── processed/       # Datos procesados
│   └── feedback/        # Retroalimentación de usuarios
├── notebook/            # Jupyter notebooks de análisis
├── config/              # Configuración YAML
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🔗 Fuentes para Probar el Modelo

### 1. Sitios de Sátira (Falsos "Seguros")

Muchos datasets de entrenamiento (como el que usamos) incluyen noticias de sátira etiquetadas como FAKE porque, técnicamente, no son hechos reales. El modelo probablemente detectará el tono absurdo.

| Sitio | Idioma | URL |
|-------|--------|-----|
| The Onion | 🇺🇸 Inglés | [theonion.com](https://theonion.com) |
| The Babylon Bee | 🇺🇸 Inglés | [babylonbee.com](https://babylonbee.com) |
| El Mundo Today | 🇪🇸 Español | [elmundotoday.com](https://elmundotoday.com) |

> **Nota:** Como la app traduce, puedes pegar una URL de El Mundo Today. Ejemplo: *"El gobierno obliga a las palomas a llevar pañal"*. El modelo traducirá y probablemente dirá FAKE.

### 2. Sitios de Fact-Checking (La mina de oro)

Los sitios que se dedican a desmentir bulos recopilan las noticias falsas virales del momento. 

> **⚠️ Truco:** No copies el artículo del periodista desmintiendo. Copia el texto del **bulo original** que citan.

| Sitio | País | URL |
|-------|------|-----|
| Snopes | 🇺🇸 USA | [snopes.com](https://snopes.com) (sección "Fact Checks" → "False") |
| Chequeado | 🇦🇷 Argentina | [chequeado.com](https://chequeado.com) (buscar etiqueta "Falso") |
| Maldita.es | 🇪🇸 España | [maldita.es](https://maldita.es) |

---

## 📝 Ejemplo de Fragmento de Prueba

```
BANGKOK, Dec 13 (Reuters)- Thailand's leader vowed on Saturday to keep 
fighting on the disputed border with Cambodia as fighter jets struck 
targets hours after U.S. President Donald Trump said he had brokered 
a new ceasefire.

Caretaker Thai Prime Minister Anutin Charnvirakul said the Southeast 
Asian nation would "continue to perform military actions until we feel 
no more harm and threats to our land and people".
```

---

## 🛠️ Características

- ✅ Análisis por URL (extrae automáticamente el contenido)
- ✅ Análisis por texto manual
- ✅ Soporte para textos en español (traducción automática)
- ✅ Múltiples modelos seleccionables
- ✅ Feedback de usuarios para mejora continua
- ✅ Interfaz moderna y responsiva

---

## 📄 Licencia

MIT License