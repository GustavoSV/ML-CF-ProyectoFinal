# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(path):
    """Carga el dataset desde un archivo CSV."""
    df = pd.read_csv(path)
    print("✅ Datos cargados. Shape:", df.shape)
    return df

def create_features(df):
    """Crea nuevas características a partir de las existentes."""
    # Normalizamos y asignamos pesos basados en conocimiento del sector
    df['riesgo_uso'] = (
        (df['horas_mensuales'] / 160) * 0.3 +
        (df['lavados_mensuales'] / 8) * 0.3 +
        (df['humedad_ambiente'] / 100) * 0.1 +
        (df['exposicion_quimicos']) * 0.3
    )
    # Mapeo de exposición al calor
    calor_map = {"Baja": 1, "Media": 2, "Alta": 3}
    df['exposicion_calor_num'] = df['exposicion_calor'].map(calor_map)
    df['riesgo_uso'] += df['exposicion_calor_num'] * 0.2
    return df

def build_preprocessor():
    """Construye un preprocesador para variables numéricas y categóricas."""
    cat_cols = ['tipo_uniforme', 'tipo_trabajo']
    num_cols = ['horas_mensuales', 'lavados_mensuales', 'humedad_ambiente', 'riesgo_uso', 'exposicion_calor_num']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols)
    ])
    return preprocessor