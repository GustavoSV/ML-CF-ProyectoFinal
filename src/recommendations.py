# src/recommendations.py
import pandas as pd
import joblib
import os
from preprocess import create_features

def generate_recommendations():
    # Verificar si el modelo existe
    model_path = "../models/model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado en {model_path}")
        print("Por favor, ejecuta primero: python src/train.py")
        return None

    # Cargar modelo
    try:
        model = joblib.load(model_path)
        print("✅ Modelo cargado correctamente")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return None

    # Definir escenarios
    escenarios = [
        {
            "tipo_uniforme": "Nomex",
            "tipo_trabajo": "Soldadura",
            "horas_mensuales": 150,
            "lavados_mensuales": 6,
            "exposicion_calor": "Alta",
            "exposicion_quimicos": 1,
            "humedad_ambiente": 70
        },
        {
            "tipo_uniforme": "Modacrilico",
            "tipo_trabajo": "Mantenimiento",
            "horas_mensuales": 120,
            "lavados_mensuales": 5,
            "exposicion_calor": "Media",
            "exposicion_quimicos": 1,
            "humedad_ambiente": 65
        },
        {
            "tipo_uniforme": "Cotton_Flame_Retardant",
            "tipo_trabajo": "Operacion",
            "horas_mensuales": 100,
            "lavados_mensuales": 3,
            "exposicion_calor": "Baja",
            "exposicion_quimicos": 0,
            "humedad_ambiente": 60
        }
    ]

    df = pd.DataFrame(escenarios)
    df = create_features(df)  # Agrega riesgo_uso y exposicion_calor_num
    print("✅ Escenarios creados. Shape:", df.shape)

    # ⚠️ Extraer las mismas columnas que se usaron en X durante el entrenamiento
    # (mismo orden y mismas columnas)
    try:
        # Obtener el preprocesador del modelo
        preprocessor = model.named_steps['preprocessor']
        
        # Asegurarnos de que las columnas sean las mismas que en entrenamiento
        # (el modelo fue entrenado con X, que no incluye 'meses_uso' ni 'estado_reemplazo')
        # pero sí todas las que quedaron después de create_features

        # Hacer la transformación completa
        X_new = preprocessor.transform(df)  # Aquí se aplica StandardScaler y OneHotEncoder
        print("✅ Datos transformados. Shape después del preprocesamiento:", X_new.shape)

        # Ahora sí predecir
        prob = model.predict_proba(df)[:, 1]  # El pipeline ya incluye el preprocessor
        pred = (prob >= 0.5).astype(int)

        # Crear tabla de recomendaciones
        recomendaciones = pd.DataFrame({
            "Tipo de Uniforme": df["tipo_uniforme"],
            "Tipo de Trabajo": df["tipo_trabajo"],
            "Prob. Reemplazo": (prob * 100).round(1).astype(str) + "%",
            "Acción Recomendada": ["Reemplazar" if p == 1 else "Monitorear" for p in pred]
        })

        print("\n📋 Recomendaciones de Reemplazo")
        print(recomendaciones.to_string(index=False))

        # Guardar
        os.makedirs("../results", exist_ok=True)
        recomendaciones.to_csv("../results/recomendaciones.csv", index=False)
        print("\n✅ Recomendaciones guardadas en ../results/recomendaciones.csv")

        return recomendaciones

    except Exception as e:
        print(f"\n❌ Error durante la predicción: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    generate_recommendations()