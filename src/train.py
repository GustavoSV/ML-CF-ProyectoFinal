# src/train.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline

# Importar funciones de preprocess
from preprocess import load_data, create_features, build_preprocessor

def load_data(path):
    return pd.read_csv(path, encoding="latin1")

def main():
    # 1. Cargar y procesar datos
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "uniformes_ignifugos.csv")
    df = load_data(data_path)
    df = create_features(df)
    
    # 2. Definir variables
    X = df.drop(columns=['meses_uso', 'estado_reemplazo'])
    y = df['estado_reemplazo']

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Construir pipeline
    preprocessor = build_preprocessor()
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 4. Entrenar modelo
    model.fit(X_train, y_train)
    print("Modelo entrenado")

    # 5. Evaluar
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"\nResultados del modelo:")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"ROC-AUC: {auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Guardar modelo
    project_dir = os.path.dirname(os.path.dirname(__file__))  # Carpeta raíz del proyecto
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"\n✅ Carpeta de modelos: {models_dir}")
    print(f"✅ Modelo guardado en: {model_path}")

    # 7. Análisis de errores
    errores = X_test[y_pred != y_test]
    print(f"\nErrores del modelo en {len(errores)} muestras.")
    if len(errores) > 0:
        print(errores[['tipo_uniforme', 'tipo_trabajo']].head())

if __name__ == "__main__":
    main()