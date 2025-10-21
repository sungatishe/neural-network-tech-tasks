import numpy as np
from tensorflow.keras import layers, models

# Generate all possible input combinations for a, b, c (2^3 = 8 cases)
def gen_dataset():
    inputs = np.array([[0, 0, 0],
                       [0, 0, 1],
                       [0, 1, 0],
                       [0, 1, 1],
                       [1, 0, 0],
                       [1, 0, 1],
                       [1, 1, 0],
                       [1, 1, 1]])
    # Target: (a and b) or (a and c)
    targets = np.array([[int((a & b) | (a & c))] for a, b, c in inputs])
    return inputs, targets

# Neural network model
model = models.Sequential([
    layers.Dense(4, activation='relu', input_shape=(3,)),  # Input layer for a, b, c
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function 1: Element-wise tensor operations
def simulate_tensor(inputs, weights):
    # Assuming weights[0] is W1, weights[1] is b1, weights[2] is W2, weights[3] is b2
    W1, b1, W2, b2 = weights
    layer1 = np.maximum(0, np.dot(inputs, W1) + b1)  # ReLU activation
    output = 1 / (1 + np.exp(-(np.dot(layer1, W2) + b2)))  # Sigmoid activation
    return (output >= 0.5).astype(int)

# Function 2: NumPy operations
def simulate_numpy(inputs, weights):
    W1, b1, W2, b2 = weights
    layer1 = np.clip(np.dot(inputs, W1) + b1, 0, None)  # ReLU activation
    output = 1 / (1 + np.exp(-(np.dot(layer1, W2) + b2)))  # Sigmoid activation
    return (output >= 0.5).astype(int)

# Generate dataset
train_data, train_labels = gen_dataset()
test_data, test_labels = train_data, train_labels  # Full dataset for training

# Step 1: Initialize model and get weights
initial_weights = [layer.get_weights() for layer in model.layers]
print("Initial weights:", initial_weights)

# Step 2: Run untrained model and functions
untrained_pred = model.predict(train_data)
untrained_result = (untrained_pred >= 0.5).astype(int)
tensor_result = simulate_tensor(train_data, initial_weights)
numpy_result = simulate_numpy(train_data, initial_weights)
print("Untrained model prediction:", untrained_result.flatten())
print("Tensor function result:", tensor_result.flatten())
print("NumPy function result:", numpy_result.flatten())
print("Match with target (untrained):", np.all(untrained_result == train_labels) and
      np.all(tensor_result == train_labels) and np.all(numpy_result == train_labels))

# Step 3: Train the model
model.fit(train_data, train_labels, epochs=100, batch_size=8, verbose=0)
trained_weights = [layer.get_weights() for layer in model.layers]
print("Trained weights:", trained_weights)

# Step 4: Run trained model and functions
trained_pred = model.predict(train_data)
trained_result = (trained_pred >= 0.5).astype(int)
tensor_result_trained = simulate_tensor(train_data, trained_weights)
numpy_result_trained = simulate_numpy(train_data, trained_weights)
print("Trained model prediction:", trained_result.flatten())
print("Tensor function result (trained):", tensor_result_trained.flatten())
print("NumPy function result (trained):", numpy_result_trained.flatten())
print("Match with target (trained):", np.all(trained_result == train_labels) and
      np.all(tensor_result_trained == train_labels) and np.all(numpy_result_trained == train_labels))