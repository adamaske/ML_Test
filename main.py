import numpy as np

amount = 10000
split = 0.5
data = np.random.rand(amount)

labels = data > 0.5
labels = np.array(labels)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf


x_train,  x_val, y_train, y_val = train_test_split(data, labels, train_size=split, test_size=split, random_state=23142, shuffle=True)

x_train = x_train.reshape(-1, int(amount * split), 1)
x_val = x_val.reshape(-1, int(amount * split), 1)

y_train = y_train.reshape(-1, int(amount * split), 1)
y_val = y_val.reshape(-1, int(amount * split), 1)

print("============ TRAINING DATA================")
print(f"x_train : {x_train.shape}")
print(x_train[0][0:20])
print(f"y_train : {y_train.shape}")
print(y_train[0][0:20])
print("============ VALIDATION DATA================")
print(f"x_val : {x_val.shape}")
print(x_val[0][0:20])
print(f"y_val : {y_val.shape}")
print(y_val[0][0:20])
print("=============================================")

#-- create model --
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(512, 4, activation='relu6', input_shape=x_train.shape[1:], padding='same'),
    tf.keras.layers.Dense(256, activation='relu6'),
    tf.keras.layers.Conv1D(128, 3, activation='relu6', input_shape=x_train.shape[1:], padding='same'),
    tf.keras.layers.Dense(64, activation='relu6'),
    tf.keras.layers.Conv1D(32, 3, activation='relu6', input_shape=x_train.shape[1:], padding='same'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Print model summary
model.summary()

loss, acc = model.evaluate(x_val, y_val, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

model.fit(x = x_train, y = y_train, batch_size=1024, epochs=50, verbose="auto", validation_data=(x_val, y_val))

# Re-evaluate the model
loss, acc = model.evaluate(x_val, y_val, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

input = [0.3, 0.7, 0.1]
correct = [ 0, 1, 0]
input = np.array(input).reshape(-1, len(input), 1)
output = model.predict(input)
predictions = output.reshape(len(input[0]))
verdicts = (predictions > 0.5) == correct
correct_num = 0
for verdict in verdicts:
    if verdict == True:
        correct_num += 1
accuracy = round((len(verdicts) / correct_num) * 100, 2)

print("============== TESTING ===============")
print(f"input : {input[0]}")
print(f"correct : {correct}")
print(f"predictions : {predictions}")
print(f"verdicts : {verdicts}")
print(f"accuracy : {accuracy}%")
