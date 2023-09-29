import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers.experimental.preprocessing import TextVectorization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos de train
train_data = pd.read_csv('train.csv')

# Etiquetar tweets como 1 si contienen palabras clave de desastres naturales, de lo contrario, etiquetar como 0
keywords = ['fire', 'flood', 'earthquake', 'hurricane', 'tornado', 'storm', 'volcano', 'tsunami', 'wildfire', 'avalanche', 'blizzard', 'drought', 'heat wave', 'hailstorm']
train_data['target'] = train_data['text'].apply(lambda x: 1 if any(keyword in x.lower() for keyword in keywords) else 0)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(train_data['text'], train_data['target'], test_size=0.3, random_state=42)

# Tokenizar el texto utilizando TextVectorization
max_tokens = 10000  # Número máximo de tokens a considerar
tokenizer = TextVectorization(max_tokens=max_tokens)
tokenizer.adapt(X_train.values.astype(str))

# Crear el modelo de red neuronal
model = keras.Sequential([
    tokenizer,
    keras.layers.Embedding(input_dim=max_tokens, output_dim=128, mask_zero=True),
    keras.layers.LSTM(64),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train.values.astype(str), y_train, epochs=5, batch_size=32)

# Evaluar el modelo en datos de prueba
y_pred = model.predict(X_test.values.astype(str))
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

print("Evaluación del modelo en datos de prueba:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_binary)}")
print(f"Precision: {precision_score(y_test, y_pred_binary)}")
print(f"Recall: {recall_score(y_test, y_pred_binary)}")
print(f"F1 Score: {f1_score(y_test, y_pred_binary)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_binary))
print("Classification Report:")
print(classification_report(y_test, y_pred_binary))

# Guardar el modelo entrenado
model.save('model_saved')

# Cargar los datos de prueba
test_data = pd.read_csv('test.csv')

# Predecir los datos de prueba
test_pred = model.predict(test_data['text'].values.astype(str))
test_pred_binary = np.where(test_pred > 0.5, 1, 0)

#evaluar el modelo en datos de prueba
y_pred = model.predict(X_test.values.astype(str))
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

print("Evaluación del modelo en datos de prueba:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_binary)}")
print(f"Precision: {precision_score(y_test, y_pred_binary)}")
print(f"Recall: {recall_score(y_test, y_pred_binary)}")
print(f"F1 Score: {f1_score(y_test, y_pred_binary)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_binary))
print("Classification Report:")
print(classification_report(y_test, y_pred_binary))


# Crear el archivo de salida
submission = pd.DataFrame({'id': test_data['id'], 'target': test_pred_binary.flatten()})
submission.to_csv('submission.csv', index=False)

# Mostrar la matriz de confusión
cm = confusion_matrix(y_test, y_pred_binary)
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['No disaster', 'Disaster'])
ax.yaxis.set_ticklabels(['No disaster', 'Disaster'])
plt.show()

# Mostrar la curva ROC
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.show()
