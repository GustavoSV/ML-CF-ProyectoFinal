# Proyecto Final: Predicción de Reemplazo de Uniformes Técnicos Ignífugos

Este proyecto aplica Machine Learning para predecir cuándo un uniforme ignífugo necesita ser reemplazado, optimizando costos y seguridad en sectores industriales de hidrocarburos, metalúrgica, minería, gasífera o eléctrificación.

Los uniformes ignífugos, son construidos con textiles que inhiben la propagación del fuego, esto le da al trabajador una ventana de tiempo para que pueda salvar su vida. Sin embargo, los textiles pueden perder sus caracteríscas ignífugas debido a diferentes condiciones poniendo en riesgo la vida de quien lo usa.

## 📂 Estructura del Proyecto
proyecto-ml-uniformes/
├── data/               # Dataset de experiencia
├── src/                # Scripts principales
├── models/             # Modelo entrenado
├── results/            # Recomendaciones generadas
└── notebooks/          # EDA en Colab


## 🚀 Cómo Ejecutar

1. Clona o crea la carpeta del proyecto.
2. Instala dependencias:
   pip install -r requirements.txt
3. Entrena el modelo:
    python src/train.py
4. Hacer una predicción:
    python src/predict.py
5. Generar recomendaciones:
    python src/recommendations.py

## ⭐ Conclusiones
Este modelo intenta predecir la necesidad de reemplazo de uniformes ignífugos basado en condiciones reales de uso. Dado que no existe información precisa, se analizaron puntualmente los 15 primeros elementos del dataset con base en las PQRs recibidas y en las fechas de pedido de los clientes, y los otros 50 se generaron de manera random.

Sin embargo el Accuracy considero debe ser más alto para considerar que puede usarse realmente, además, la simulación de posibles escenarios 'Recomendaciones de Reemplazo' arroja un dato que no es confiable, ya que un uniforme de cuando se expone a trabajos de soldadura es muy susceptible a dañarse por impactos en el textil, por las altas temperaturas a las que normalmente se trabaja, por lo tanto, al estar por debajo del 50%, es algo que debe revisarse mejor.

La intención de esta tabla de recomendaciones es que pueda usarse para dar un valor agregado a nuestros clientes de manera tal que puedan guiar sus decisiones de mantenimiento y compras.

Se hizo un preprocesamiento de la información para dejarla en términos razonables para el entrenamiento del modelo, en primer lugar, se separan las columnas de texto de las columnas numéricas, luego se revisan las magnitudes de las columnas numéricas para evitar que el modelo le de más importancia a una columna por que se expresa en magnitudes más grandes, como horas_mensuales y lavados_mensuales, finalmente las columnas de texto o categorías se convierten a números.

Un aprendizaje en mi caso, se da con la eliminación para el entrenamiento del modelo, de la variable 'meses_uso', debido a que esta información se conoce cuando ya han ocurrido los hechos, es decir, meses después de haber tomado la decisión de comprar o reemplazar un uniforme, en otras palabras, al momento de tomar la decisión de comprar los uniformes, ese dato aún no se ha concretado porque ocurre meses adelante, en cambio, usamos solo variables disponibles antes de la decisión:

    Tipo de uniforme
    Condiciones de trabajo
    Horas y lavados mensuales
    Exposición a calor/químicos
    Grado de humedad