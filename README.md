# Predicción de abandono de clientes (Customer Churn) | TP FINAL INTEGRADOR Grupo 22

## Descripción del proyecto

Este proyecto desarrolla y compara modelos de **Machine Learning** para predecir el abandono de clientes (*customer churn*) en una empresa de telecomunicaciones. El objetivo es transformar los datos de clientes en información útil para **priorizar acciones de retención**, especialmente cuando el equipo comercial solo puede contactar a una parte de la cartera.

El análisis contempla la preparación de datos, la construcción de pipelines, la comparación de algoritmos, la optimización de hiperparámetros y la evaluación final sobre datos no utilizados durante el entrenamiento.

## Objetivo de negocio

Identificar a los clientes con mayor riesgo de abandono para orientar las campañas de retención. Además de evaluar métricas de clasificación, se analiza un escenario comercial concreto: **contactar al 20 % de los clientes con mayor probabilidad estimada de abandono**.

## Tecnologías utilizadas

- **Python** para el análisis y modelado.
- **Pandas y NumPy** para el procesamiento de datos.
- **Scikit-learn** para preprocesamiento, pipelines, entrenamiento, validación y optimización.
- **Matplotlib y Seaborn** para visualizaciones.
- **Jupyter Notebook** para documentar el proceso.

> El notebook también importa TensorFlow, aunque los modelos evaluados en este análisis se implementan con Scikit-learn.

## Dataset

Se utiliza un conjunto de datos de clientes de telecomunicaciones, cargado en el notebook desde el archivo `Telco_customer_churn-_1_.csv`.

La variable objetivo es **`Churn Label`**, transformada a una clasificación binaria:

- `1`: el cliente abandona.
- `0`: el cliente permanece.

Para evitar incorporar información no apropiada para la predicción, se excluyen identificadores y variables como `Churn Score`, `Churn Reason`, `Churn Category`, `Churn Value` y `Customer Status`, entre otras.

## Metodología

### 1. Preparación de datos

- Conversión de `Total Charges` a formato numérico.
- Revisión de valores faltantes e imputación con cero cuando corresponden a clientes con `Tenure Months = 0`.
- Comprobación de registros duplicados.
- Selección de variables predictoras y separación de la variable objetivo.
- División estratificada en **80 % entrenamiento y 20 % prueba**, con `random_state=42`.

### 2. Preprocesamiento y pipelines

Se construye un pipeline que aplica las transformaciones necesarias según el tipo de variable:

- **Numéricas:** imputación por mediana y estandarización.
- **Categóricas:** imputación por moda y codificación *One-Hot Encoding*.

Estas transformaciones se integran con los modelos para mantener un proceso reproducible y evitar ajustar el preprocesamiento sobre el conjunto de prueba.

### 3. Entrenamiento y validación

Se comparan tres algoritmos:

1. **Regresión Logística**.
2. **Árbol de Decisión**.
3. **Random Forest**.

Se utiliza **validación cruzada estratificada de 5 particiones** y se analizan las métricas Recall, Precision, F1, ROC-AUC y PR-AUC.

### 4. Optimización de Random Forest

Se exploran hiperparámetros mediante `RandomizedSearchCV`, primero priorizando **Recall** y luego **PR-AUC**.

La optimización por Recall incrementa la detección de abandonos, pero reduce la precisión. Por eso se evalúa una segunda estrategia basada en PR-AUC, buscando un mejor equilibrio para identificar clientes de riesgo en un problema de clasificación con clases desbalanceadas.

También se incorpora **Recall@20 %**, que mide qué proporción de los abandonos reales se identifica al seleccionar al 20 % de clientes con mayor riesgo estimado.

## Resultados

### Comparación inicial: validación cruzada

| Modelo | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Regresión Logística | 0.564 | 0.658 | 0.607 | 0.851 | 0.660 |
| Árbol de Decisión | 0.593 | 0.526 | 0.557 | 0.700 | 0.421 |
| Random Forest | 0.695 | 0.602 | 0.645 | 0.854 | 0.657 |

### Comparación final: conjunto de prueba

| Modelo | Recall | Precision | F1 | ROC-AUC | PR-AUC | Recall@20 % |
|---|---:|---:|---:|---:|---:|---:|
| **Random Forest optimizado por PR-AUC** | **0.746** | 0.549 | **0.633** | **0.846** | **0.647** | **0.505** |
| Regresión Logística | 0.553 | **0.618** | 0.584 | 0.841 | 0.636 | 0.492 |

Los resultados muestran que el **Random Forest optimizado por PR-AUC** obtiene el mejor Recall, F1, ROC-AUC, PR-AUC y Recall@20 % entre los dos modelos finalistas en prueba. La Regresión Logística presenta una Precision superior.

### Impacto comercial: contactar al 20 % de los clientes

En el conjunto de prueba se identificaron **374 abandonos reales**. Al priorizar los **282 clientes** con mayor probabilidad estimada de abandono:

| Modelo | Clientes a contactar | Abandonos detectados | Recall@20 % |
|---|---:|---:|---:|
| **Random Forest optimizado por PR-AUC** | 282 | **189** | **50.5 %** |
| Regresión Logística | 282 | 184 | 49.2 % |

**Hallazgo principal:** el modelo Random Forest optimizado por PR-AUC permite identificar **189 de los 374 abandonos reales** al focalizar el contacto en el 20 % de los clientes del conjunto de prueba. Esto ilustra cómo un modelo predictivo puede ayudar a priorizar recursos comerciales limitados.

> Estos resultados corresponden a una evaluación sobre datos históricos. No implican que el contacto evite automáticamente el abandono: para medir el efecto real de una campaña de retención sería necesario evaluar su impacto comercial.

## Conclusiones

- **Random Forest optimizado por PR-AUC** presenta el mejor desempeño general entre los modelos finalistas evaluados en prueba.
- Optimizar exclusivamente por **Recall** puede aumentar la detección de abandonos a costa de generar más falsos positivos; por eso es importante considerar varias métricas.
- **Recall@20 %** conecta la evaluación técnica con una decisión de negocio: qué clientes priorizar cuando la capacidad de contacto es limitada.
- El uso de **pipelines**, validación cruzada estratificada y un conjunto de prueba separado contribuye a una evaluación más consistente.

## Cómo ejecutar el proyecto

1. Clonar o descargar el repositorio.
2. Colocar el archivo `Telco_customer_churn-_1_.csv` en la misma carpeta que el notebook, o actualizar la ruta de lectura del CSV.
3. Instalar las dependencias:

   ```bash
   python -m pip install pandas numpy matplotlib seaborn scikit-learn scipy tensorflow notebook
   ```

   > TensorFlow aparece entre los imports del notebook; no se utiliza para entrenar los modelos comparados en este proyecto.

4. Abrir `Notebook.ipynb` en Jupyter Notebook o Visual Studio Code y ejecutar las celdas en orden.

## Archivos del proyecto

```text
customer-churn-prediction/
├── README.md
├── Notebook.ipynb
└── Telco_customer_churn-_1_.csv   # Dataset requerido para ejecutar el notebook
```

**Nota:** verificá los permisos y condiciones de uso del dataset antes de publicarlo en GitHub. Si no podés incluirlo, indicá en el repositorio cómo obtenerlo.

---

**Enfoque del proyecto:** Machine Learning aplicado a la toma de decisiones comerciales y a la retención de clientes.
