# Natural-Language-Processing-with-Disaster-

**English: This project focuses on developing a machine learning model to detect tweets related to natural disasters. The goal is to quickly identify tweets reporting catastrophic events, which can be useful for alerting authorities and affected individuals.

**Español: Este proyecto se centra en desarrollar un modelo de aprendizaje automático para detectar tweets relacionados con desastres naturales. El objetivo es identificar rápidamente tweets que informan sobre eventos catastróficos, lo que puede ser útil para alertar a las autoridades y a las personas afectadas.

---

## Dataset

**English: We use a labeled dataset containing English tweets. Each tweet is labeled as "0" if it's not related to a natural disaster and as "1" if it reports a natural disaster. The training data is split into training and test data for evaluating the model's performance.

**Español: Utilizamos un conjunto de datos etiquetado que contiene tweets en inglés. Cada tweet está etiquetado como "0" si no está relacionado con un desastre natural y como "1" si informa sobre un desastre natural. Los datos de entrenamiento se dividen en datos de entrenamiento y prueba para evaluar el rendimiento del modelo.

---

## Text Processing

**English: Before training the model, we perform text processing, including:
- Conversion of text to lowercase.
- Removal of special characters and numbers.
- Tokenization to represent words as numbers.

**Español: Antes de entrenar el modelo, realizamos un procesamiento de texto que incluye:
- Conversión del texto a minúsculas.
- Eliminación de caracteres especiales y números.
- Tokenización para representar las palabras como números.

---

## Machine Learning Model

**English: The machine learning model is built using TensorFlow and Keras. It consists of the following layers:
1. Text Tokenization Layer.
2. Embedding Layer to convert tokens to vectors.
3. LSTM Layer to model sequences.
4. Dense Layer with sigmoid activation for binary classification.

**Español: El modelo de aprendizaje automático se construye utilizando TensorFlow y Keras. Está compuesto por las siguientes capas:
1. Capa de Tokenización de Texto.
2. Capa de Embedding para convertir tokens en vectores.
3. Capa LSTM para modelar secuencias.
4. Capa Densa con activación sigmoide para la clasificación binaria.

---

## Model Evaluation

**English: The model is evaluated using performance metrics, including:
- Accuracy.
- Precision.
- Recall.
- F1 Score.
- Confusion Matrix.

**Español: El modelo se evalúa utilizando métricas de rendimiento, que incluyen:
- Precisión (Accuracy).
- Precisión (Precision).
- Recall.
- Puntuación F1.
- Matriz de Confusión.

---

## Using Trained Model

**English: Once trained, the model can be used to predict whether a given tweet is related to a natural disaster or not.

**Español: Una vez entrenado, el modelo puede utilizarse para predecir si un tweet dado está relacionado con un desastre natural o no.

---

## How to Run the Code

**English Comment:** 
1. Clone this repository to your local machine.
2. Ensure you have all the dependencies installed.
3. Run the code to train the model and make predictions.

**Comentario en Español:** 
1. Clona este repositorio en tu máquina local.
2. Asegúrate de tener todas las dependencias instaladas.
3. Ejecuta el código para entrenar el modelo y realizar predicciones.

---

## Credits

**English Comment:** This project was created by Diego Roca Costa

**Comentario en Español:** Este proyecto fue creado por Diego Roca Costa

