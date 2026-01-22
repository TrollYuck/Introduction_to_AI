import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = img_array.reshape(1, 28, 28)  # Reshape to match model input
    return img_array

image_dir = 'numbers'
image_paths = [os.path.join(image_dir, f'digit_{i}_{j:02d}.png') for i in range(10) for j in range(1, 6)]
actual_labels = [i for i in range(10) for _ in range(5)]

images = np.vstack([load_and_preprocess_image(path) for path in image_paths])

model = tf.keras.models.load_model('model.h5')

predictions = model.predict(images)

probabilities = tf.nn.softmax(predictions).numpy()

predicted_classes = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(actual_labels, predicted_classes)
print(f"Accuracy on my digits: {accuracy * 100:.2f}%")

report = classification_report(actual_labels, predicted_classes, target_names=[str(i) for i in range(10)], zero_division=1)
print(report)

cm = confusion_matrix(actual_labels, predicted_classes)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='PuRd', xticklabels=[str(i) for i in range(10)], yticklabels=[str(i) for i in range(10)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_my_digits.png', dpi=300)
plt.show()