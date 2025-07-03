!pip install tensorflow==2.14
!pip install pennylane==0.37

import pennylane as qml
import numpy as np
import tensorflow as tf
import pandas as pd

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
tf.keras.backend.set_floatx('float64')

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Load the data
x, y = load_breast_cancer(return_X_y=True, as_frame=True)

data = pd.concat([x, y], axis=1)
data.head()



x_tr, x_test, y_tr, y_test = train_test_split(x,y, train_size=0.8)

from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()

x_tr = scaler.fit_transform(x_tr)

x_tr
sss
x_test = scaler.transform(x_test)
x_test = np.clip(x_test,0,1)

from sklearn.decomposition import PCA

pca = PCA(n_components = 4)

xs_tr = pca.fit_transform(x_tr)
xs_test = pca.transform(x_test)

pip install silence_tensorflow

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import pennylane as qml

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=[0]))]


n_layers = 3
weight_shapes = {"weights": (n_layers, n_qubits)}

import matplotlib.pyplot as plt

# Sample inputs and weights
inputs = np.random.rand(n_qubits)
weights = np.random.rand(n_layers, n_qubits)

# Draw the circuit using matplotlib
fig, ax = qml.draw_mpl(qnode)(inputs, weights)

# Display the circuit
plt.show()

qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=1)
model = tf.keras.models.Sequential([ qlayer])

opt = tf.keras.optimizers.Adam(learning_rate=0.005)
model.compile(opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])

history = model.fit(xs_tr, y_tr, epochs=6, batch_size=5, validation_split=0.25)




import matplotlib.pyplot as plt

def plot_loss(history):
    tr_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = np.array(range(len(tr_loss))) + 1
    plt.plot(epochs, tr_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


plot_loss(history)


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

tr_ac = accuracy_score(model.predict(xs_tr)>=0.5,y_tr)

print("Training accuracy: ", tr_ac)
# Calculate accuracy, precision, recall, and F1 score
print("Accuracy:", accuracy_score(model.predict(xs_tr)>=0.5,y_tr))
print("Precision:", precision_score(model.predict(xs_tr)>=0.5,y_tr))
print("Recall:", recall_score(model.predict(xs_tr)>=0.5,y_tr))
print("F1 Score:", f1_score(model.predict(xs_tr)>=0.5,y_tr))








