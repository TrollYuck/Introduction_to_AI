import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import gzip

with gzip.open('data/mnist.pkl.gz', 'rb') as f:
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = pkl.load(f, encoding='latin1')
x_train, x_test = x_train / 255.0, x_test / 255.0
# Normalizacja danych to proces skalowania wartości danych wejściowych do określonego zakresu,
# zazwyczaj od 0 do 1 lub od -1 do 1. Celem normalizacji jest zapewnienie,
# że wszystkie cechy mają porównywalne skale, co pomaga modelowi w efektywnym uczeniu się.


x_train_reshaped = x_train.reshape(-1, 28, 28)
x_test_reshaped = x_test.reshape(-1, 28, 28)
x_valid_reshaped = x_valid.reshape(-1, 28, 28)


model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])


predictions = model(x_train_reshaped[:1]).numpy()

#Funkcja tf.nn.softmax przekształca logity na prawdopodobieństwa dla każdej klasy.
print(tf.nn.softmax(predictions).numpy())

#Logarytmiczna funkcja straty
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#nie przetrenowany model powinien dawać prawdopodobieństwo bliskie losowemu tf.math.log(1/10) ~= 2.3.
print(loss_fn(y_train[:1], predictions).numpy())

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train_reshaped, y_train, epochs=10, validation_data=(x_valid_reshaped, y_valid))

model.evaluate(x_test_reshaped, y_test, verbose=2)

model.save('model.h5')

y_pred = model.predict(x_test_reshaped)
y_pred_classes = np.argmax(y_pred, axis=1)

report = classification_report(y_test, y_pred_classes, target_names=[str(i) for i in range(10)])
print(report)

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=[str(i) for i in range(10)], yticklabels=[str(i) for i in range(10)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()