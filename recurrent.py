import numpy as np
import tensorflow as tf

def generate_sequence_data(sequence_length, num_sequences):
    X = np.zeros((num_sequences, sequence_length, 1))
    y = np.zeros((num_sequences, 1))
    for i in range(num_sequences):
        X[i, :, 0] = np.sin(np.linspace(0, 10, sequence_length) + i * 0.1)
        y[i, 0] = np.sum(X[i, :, 0])
    return X, y

sequence_length = 10
num_sequences = 1000
input_dim = 1
hidden_units = 50
output_dim = 1
learning_rate = 0.001
epochs = 50
batch_size = 32

X_train, y_train = generate_sequence_data(sequence_length, num_sequences)

X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

Wxh = tf.Variable(tf.random.normal([input_dim, hidden_units], stddev=0.1))
Whh = tf.Variable(tf.random.normal([hidden_units, hidden_units], stddev=0.1))
Why = tf.Variable(tf.random.normal([hidden_units, output_dim], stddev=0.1))
bh = tf.Variable(tf.zeros([1, hidden_units]))
by = tf.Variable(tf.zeros([1, output_dim]))

def tanh(x):
    return tf.tanh(x)

def rnn_forward(X):
    h = tf.zeros([1, hidden_units])
    for t in range(sequence_length):
        x_t = X[:, t, :]
        h = tanh(tf.matmul(x_t, Wxh) + tf.matmul(h, Whh) + bh)
    y_pred = tf.matmul(h, Why) + by
    return y_pred

def compute_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        with tf.GradientTape() as tape:
            y_pred = rnn_forward(X_batch)
            loss = compute_loss(y_batch, y_pred)

        gradients = tape.gradient(loss, [Wxh, Whh, Why, bh, by])

        optimizer.apply_gradients(zip(gradients, [Wxh, Whh, Why, bh, by]))

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}")

X_test, y_test = generate_sequence_data(sequence_length, 100)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

predictions = rnn_forward(X_test)

for i in range(10):
    print(f"True sum: {y_test[i].numpy()}, Predicted sum: {predictions[i].numpy()}")