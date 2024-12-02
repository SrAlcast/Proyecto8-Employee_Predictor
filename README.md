# 🌟 Employee Predictor: Unlock Workplace Insights 

![Employee Predictor](https://raw.githubusercontent.com/SrAlcast/Proyecto8-Employee_Predictor/refs/heads/main/src/imagen%20README.jpg?token=GHSAT0AAAAAACW5JJGYIIDOZME6LEIQDVUIZ2M7IYQ)

## 📖 Introducción

La retención de empleados es un desafío crítico para las organizaciones de todo el mundo. Comprender por qué los empleados deciden quedarse o irse puede impactar significativamente el desempeño y la cultura de una empresa. Este proyecto tiene como objetivo predecir la retención de empleados utilizando técnicas avanzadas de machine learning, revelando los factores clave que impulsan la satisfacción y las decisiones de los empleados.

A través del análisis de diversos conjuntos de datos, este proyecto no solo construye modelos predictivos, sino que también proporciona recomendaciones prácticas para fomentar un entorno laboral más positivo. Desde la limpieza de datos hasta la optimización de modelos, exploramos a fondo las dinámicas de la retención de empleados.

## 🗂️ Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

```
├── data/                # Conjuntos de datos crudos y procesados
├── encoders/            # Pkl para estandarizar y encodear datos
├── models/              # Modelos de machine learning entrenados
├── notebooks/           # Notebooks de Jupyter para análisis y modelado
├── results/             # Datos procesados y resultados
├── src/                 # Código fuente para preprocesamiento y modelado
├── streamlit/           # Código para la ejecucion de streamlit con el modelo
└── README.md            # Descripción del proyecto
```

## 🛠️ Instalación y Requisitos

Este proyecto utiliza las siguientes tecnologías y bibliotecas:

- **Lenguaje**: Python
- **Bibliotecas**:
  - [pandas](https://pandas.pydata.org/)
  - [numpy](https://numpy.org/)
  - [matplotlib](https://matplotlib.org/)
  - [seaborn](https://seaborn.pydata.org/)
  - [scikit-learn](https://scikit-learn.org/)
  - [xgboost](https://xgboost.readthedocs.io/)
  - [joblib](https://joblib.readthedocs.io/)
  
Puedes instalar las bibliotecas utilizando tu gestor de paquetes favorito.

## 📊 Resultados e Insights

### **1. Logistic Regression**
La regresión logística ofrece un desempeño básico con un **AUC de 0.80**, siendo el modelo menos efectivo en discriminar entre las clases.  
- **Errores:** 65 falsos positivos y 76 falsos negativos.  
- **Ventajas:** Simplicidad y velocidad, útil como punto de partida o referencia.  
- **Desventajas:** No captura relaciones complejas, lo que lo hace menos confiable para problemas donde la clase positiva es crucial.  
- **Conclusión:** Adecuado solo para problemas simples o como modelo interpretativo preliminar.



### **2. Árbol de Decisión**
El árbol de decisión mejora significativamente sobre Logistic Regression, logrando un **AUC de 0.90** y reduciendo los errores.  
- **Errores:** 28 falsos positivos y 45 falsos negativos.  
- **Ventajas:** Interpretable y capaz de identificar patrones más complejos.  
- **Desventajas:** Propenso al sobreajuste y menos eficiente para problemas grandes o complejos.  
- **Conclusión:** Útil para tareas donde la interpretabilidad es importante, pero no tan preciso como otros modelos más avanzados.



### **3. Random Forest**
Random Forest es el modelo más robusto, con un **AUC sobresaliente de 0.98** y un excelente equilibrio en su matriz de confusión.  
- **Errores:** 1 falso positivo y 36 falsos negativos.  
- **Ventajas:** Alta precisión, excelente para manejar datos complejos y múltiples características.  
- **Desventajas:** Indicios de sobreajuste (100% de precisión en entrenamiento), requiere ajuste de hiperparámetros.  
- **Conclusión:** Ideal para problemas donde la precisión es crítica, aunque es necesario ajustar parámetros para evitar sobreajuste.



### **4. Gradient Boosting**
Gradient Boosting combina precisión y generalización de manera eficiente, logrando un **AUC de 0.95**.  
- **Errores:** 13 falsos positivos y 38 falsos negativos.  
- **Ventajas:** Menos propenso al sobreajuste que Random Forest, balance entre rendimiento y eficiencia.  
- **Desventajas:** Ligeramente menos preciso que Random Forest.  
- **Conclusión:** Opción confiable para problemas complejos, especialmente cuando se busca un equilibrio entre precisión y uso de recursos.



### **5. XGBoost**
XGBoost es muy similar a Gradient Boosting en desempeño, con un **AUC ligeramente superior de 0.96**.  
- **Errores:** 10 falsos positivos y 42 falsos negativos.  
- **Ventajas:** Flexibilidad en optimización de hiperparámetros, rapidez en entrenamiento, ideal para escenarios con recursos limitados.  
- **Desventajas:** Rendimiento ligeramente inferior a Random Forest.  
- **Conclusión:** Excelente opción para problemas que requieren procesamiento rápido y eficiente.


### **Conclusión General**
- **Mejor Modelo:** **Random Forest** se destaca como el modelo más preciso, ideal para problemas donde los errores mínimos son críticos. Sin embargo, requiere ajustes para reducir el sobreajuste.  
- **Alternativas Sólidas:** **Gradient Boosting y XGBoost** ofrecen un excelente equilibrio entre rendimiento y eficiencia, siendo más generalizables y menos propensos al sobreajuste.  
- **Modelos Básicos:** Logistic Regression y Árbol de Decisión son adecuados para tareas más simples o como referencias iniciales, pero no son suficientemente potentes para problemas complejos o de alta dimensionalidad.

## 🔄 Próximos Pasos y Contribuciones

Mejoras y extensiones futuras incluyen:
- Refinar los modelos predictivos con características adicionales y técnicas avanzadas.
- Explorar relaciones causales entre la satisfacción laboral y la retención.

¡Las contribuciones al proyecto son bienvenidas! Puedes:
- Enviar un **pull request** para mejoras en el código.
- Abrir **issues** para preguntas, sugerencias o reportes de errores.

## 🤝 Créditos

Este proyecto forma parte de una iniciativa más amplia para explorar soluciones basadas en datos en la gestión organizacional. Agradecimientos especiales a todos los colaboradores que hicieron posible este proyecto.

---
