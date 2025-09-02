# Proyecto Final: Predicción de Reemplazo de Uniformes Técnicos Ignífugos

Este proyecto aplica Machine Learning para predecir cuándo un uniforme ignífugo necesita ser reemplazado, optimizando costos y seguridad en sectores industriales.

## 📂 Estructura del Proyecto
proyecto-ml-uniformes/
├── data/               # Dataset sintético
├── src/                # Scripts principales
├── models/             # Modelo entrenado
├── results/            # Recomendaciones generadas
└── notebooks/          # (Opcional) EDA en Colab


## 🚀 Cómo Ejecutar

1. Clona o crea la carpeta del proyecto.
2. Instala dependencias:
   ```bash
   pip install -r requirements.txt
3. Entrena el modelo:
    python src/train.py
4. Hacer una predicción:
    python src/predict.py
5. Generar recomendaciones:
    python src/recommendations.py