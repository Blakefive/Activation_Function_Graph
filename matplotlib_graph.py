import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

x = np.linspace(-np.pi, np.pi, 12)
tanh_y = np.tanh(x)
sigmoid_y = tf.keras.activations.sigmoid(x)
swish_y = tf.keras.activations.swish(x)
selu_y = tf.keras.activations.selu(x)
gelu_y = tf.keras.activations.gelu(x)
relu_y = tf.keras.activations.relu(x)
elu_y = tf.keras.activations.elu(x)
elu_yd = tf.keras.activations.elu(x, alpha=1.5)

plt.axvline(x=0, color = 'black')
plt.axhline(y=0, color = 'black')
plt.plot(x, tanh_y, color = 'pink', label="'tanh'")
plt.plot(x, sigmoid_y, color = 'slategray', label="'sigmoid'")
plt.plot(x, swish_y, color = 'red', label="'swish'")
plt.plot(x, selu_y, color = 'blue', label="'selu'")
plt.plot(x, gelu_y, color = 'gold', label="'gelu'")
plt.plot(x, relu_y, color = 'aqua', label="'relu'")
plt.plot(x, elu_y, color = 'indigo', label="'elu'")
plt.plot(x, elu_yd, color = 'lime', label="'elu(1.5)'")
plt.title("activation graph")
plt.tight_layout()
plt.legend(loc='upper right', ncol=3)
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
