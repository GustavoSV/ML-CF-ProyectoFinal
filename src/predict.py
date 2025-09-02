# src/predict.py
import joblib
import pandas as pd

def predict_reemplazo(data):
    """Predice si un uniforme necesita reemplazo."""
    model = joblib.load("../models/model.pkl")
    pred = model.predict(data)
    prob = model.predict_proba(data)[:, 1]
    return pred[0], prob[0]

if __name__ == "__main__":
    # Cargar modelo
    try:
        model = joblib.load("../models/model.pkl")
    except FileNotFoundError:
        print("❌ Modelo no encontrado. Ejecuta 'python src/train.py' primero.")
        exit()

    # Datos de ejemplo
    new_data = pd.DataFrame([{
        'tipo_uniforme': 'Nomex',
        'tipo_trabajo': 'Soldadura',
        'horas_mensuales': 140,
        'lavados_mensuales': 6,
        'exposicion_calor': 'Alta',
        'exposicion_quimicos': 1,
        'humedad_ambiente': 75,
        'riesgo_uso': 0.85,
        'exposicion_calor_num': 3
    }])

    pred, prob = predict_reemplazo(new_data)
    print(f"\n🔍 Predicción para nuevo caso:")
    print(f"Uniforme: {new_data['tipo_uniforme'].iloc[0]} | Trabajo: {new_data['tipo_trabajo'].iloc[0]}")
    print(f"Resultado: {'🔴 Requiere reemplazo' if pred == 1 else '🟢 No requiere reemplazo'}")
    print(f"Probabilidad de reemplazo: {prob:.2%}")