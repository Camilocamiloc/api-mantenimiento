# Sistema de Priorización Predictiva de Mantenimiento

## Resumen
Sistema de Machine Learning orientado a priorizar diariamente qué vehículos deben ingresar a mantenimiento bajo restricciones de capacidad operativa.

El modelo estima la probabilidad de que un vehículo requiera mantenimiento en el corto plazo y transforma esta probabilidad en un ranking de riesgo para apoyar la toma de decisiones.

---

## Problema
La flota presenta una restricción operativa: no es posible intervenir todos los vehículos diariamente. El objetivo no es únicamente clasificar, sino priorizar correctamente.

Se define una función de costo asimétrica:

- Falso negativo: no detectar un vehículo que requiere mantenimiento (alto impacto)
- Falso positivo: intervenir un vehículo no crítico (impacto controlado)

El sistema se optimiza para minimizar falsos negativos, incluso si esto implica un aumento en falsos positivos.

---

## Enfoque

### Modelo predictivo
- Algoritmo: Random Forest
- Target: probabilidad de requerir mantenimiento en los próximos 3 días
- Manejo de desbalance: `class_weight='balanced'`

### Feature engineering
- Variables temporales:
  - Día de la semana, mes, semana del año
- Variables históricas por vehículo:
  - Días desde último y penúltimo mantenimiento
  - Promedio y desviación estándar de intervalos históricos
- Ventanas móviles:
  - Número de mantenimientos en últimos 30 y 60 días
- Features derivadas:
  - Relación entre frecuencia reciente e histórica

### Validación
- División estricta por fecha:
  - Train: datos históricos
  - Validation: ajuste del modelo
  - Test: evaluación out-of-sample
- Prevención de data leakage mediante construcción temporal de features

---

## Resultados

### Validation set
- Recall clase 1: ~0.99
- Precision clase 1: ~0.66
- AUC-PR: ~0.67

### Test set
- Recall clase 1: ~0.98
- Precision clase 1: ~0.86
- AUC-PR: ~0.85

El modelo mantiene un desempeño consistente en datos fuera de muestra, evidenciando buena capacidad de generalización y degradación controlada.

---

## Interpretación
El modelo prioriza la detección de la clase positiva (vehículos que requieren mantenimiento), logrando un recall cercano al 100%.

Este comportamiento es consistente con el objetivo del negocio, donde el costo de un falso negativo es significativamente mayor que el de un falso positivo.

---

## Uso operativo
Las probabilidades generadas por el modelo se utilizan para construir un ranking diario de riesgo. A partir de este ranking se seleccionan los vehículos a intervenir según la capacidad disponible.

---

## Datos
Los datos han sido anonimizados por motivos de confidencialidad. Se conserva la estructura, lógica y comportamiento del sistema original.



# API Mantenimiento Preventivo

API desarrollada con FastAPI para predicción diaria de mantenimiento vehicular usando Random Forest.

## Modelo
- Random Forest Classifier
- Feature engineering temporal avanzado
- Selección óptima con cupos por tipología

## Demo
Swagger:
https://api-mantenimiento-vlth.onrender.com/demo

---

## Stack tecnológico
- Python
- Scikit-learn
- Pandas
- NumPy
- Joblib
- FastApi

---
