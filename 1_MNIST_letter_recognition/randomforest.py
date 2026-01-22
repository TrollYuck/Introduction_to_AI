import gzip
import pickle as pkl
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

with gzip.open('data/mnist.pkl.gz', 'rb') as f:
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = pkl.load(f, encoding='latin1')

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train_reshaped = x_train.reshape(-1, 28 * 28)
x_test_reshaped = x_test.reshape(-1, 28 * 28)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=17, verbose=2)
rf_classifier.fit(x_train_reshaped, y_train)

y_pred = rf_classifier.predict(x_test_reshaped)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

report = classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)])
print(report)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=[str(i) for i in range(10)], yticklabels=[str(i) for i in range(10)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_random_forest.png', dpi=300)
plt.show()