import pandas as pd
import numpy as np

def load_mockdata():
    # Creamos un mock dataset más grande
    np.random.seed(42) # Para reproducibilidad

    n_samples = 50

    tipo_uniforme_choices = ['Nomex', 'Modacrilico', 'Cotton_Flame_Retardant']
    tipo_trabajo_choices = ['Soldadura', 'Mantenimiento', 'Operacion', 'Inspeccion']
    exposicion_calor_choices = ['Baja', 'Media', 'Alta']

    data_large = {
        'tipo_uniforme': np.random.choice(tipo_uniforme_choices, n_samples),
        'tipo_trabajo': np.random.choice(tipo_trabajo_choices, n_samples),
        'horas_mensuales': np.random.randint(150, 160, n_samples),
        'lavados_mensuales': np.random.randint(1, 10, n_samples),
        'exposicion_calor': np.random.choice(exposicion_calor_choices, n_samples),
        'exposicion_quimicos': np.random.randint(0, 2, n_samples),
        'humedad_ambiente': np.random.randint(40, 90, n_samples),
        'meses_uso': np.random.randint(2, 6, n_samples),
        'estado_reemplazo': np.random.randint(0, 2, n_samples) # Generamos aleatoriamente por ahora
    }

    df_large = pd.DataFrame(data_large)

    return df_large