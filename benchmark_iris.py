import pandas as pd
from sklearn.model_selection import train_test_split

file_path = './datasets/iris.txt'  # Replace with your file path

# Read the file into a DataFrame
data = []
with open(file_path, 'r') as file:
    for line in file:
        parts = line.strip().split(':')
        y = int(parts[0])  # Class label
        X = [float(x) for x in parts[1].split(',')]  # Convert features to float
        data.append([y] + X)

# Convert to DataFrame
df = pd.DataFrame(data, columns=['target'] + [f'feature_{i}' for i in range(4)])  # 4 features

# Split data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import tensorflow as tf

# Model configuration
input_shape = X_train.shape[1]
output_shape = len(y.unique())  # Number of classes in the target
learning_rate = 0.01
epochs = 100
batch_size = 20

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='tanh', input_shape=(input_shape,)),  # First hidden layer with tanh
    tf.keras.layers.Dense(10, activation='tanh'),                             # Second hidden layer with tanh
    tf.keras.layers.Dense(output_shape, activation='sigmoid')                 # Output layer with sigmoid
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")
