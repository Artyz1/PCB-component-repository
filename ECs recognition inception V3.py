import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Directories containing the images and labels
train_directory = 'path/to/train/images'
val_directory = 'path/to/valid/images'
test_directory = 'path/to/test/images'
train_labels_directory = 'path/to/train/labels'
val_labels_directory = 'path/to/valid/labels'
test_labels_directory = 'path/to/test/labels'

# Note: Various numbers refer to various electronic components (ECs).
# names: {0: 'Button', 1: 'Capacitor Jumper', 2: 'Capacitor', 3: 'Clock', 4: 'Connector', 5: 'Diode', 6: 'EM', 7: 'Electrolytic Capacitor', 8: 'Ferrite Bead', 9: 'IC', 10: 'Inductor', 11: 'Jumper', 12: 'Led', 13: 'Pads', 14: 'Pins', 15: 'Resistor Jumper', 16: 'Resistor Network', 17: 'Resistor', 18: 'Switch', 19: 'Test Point', 20: 'Transistor', 21: 'Unknown Unlabeled', 22: 'iC'}

# Load images and labels
def load_images_and_labels(image_directory, label_directories):
    images = []
    labels = []
    for filename in os.listdir(image_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_directory, filename)
            label_path = None
            for label_dir in label_directories:
                candidate_label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
                if os.path.exists(candidate_label_path):
                    label_path = candidate_label_path
                    break
            if label_path is None:
                print(f"Label file not found for {filename}")
                continue
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))  # InceptionV3 expects 299x299 images
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            with open(label_path, 'r') as file:
                lines = file.readlines()
                inductor_present = any('10' in line.split()[0] for line in lines)
                labels.append(1 if inductor_present else 0)  # Map inductor class (10) to 1, others to 0
    return np.array(images), np.array(labels)

# Load images and labels from directories
train_images, train_labels = load_images_and_labels(train_directory, [train_labels_directory, val_labels_directory, test_labels_directory])
val_images, val_labels = load_images_and_labels(val_directory, [train_labels_directory, val_labels_directory, test_labels_directory])
test_images, test_labels = load_images_and_labels(test_directory, [train_labels_directory, val_labels_directory, test_labels_directory])

# Normalize images
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights))

# Define the model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(units=128, activation='relu', kernel_regularizer=l2(0.00066172))(x)
x = Dropout(rate=0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the model 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model with class weights
history = model.fit(train_images, train_labels,
                    epochs=20,
                    batch_size=32,
                    validation_data=(val_images, val_labels),
                    class_weight=class_weights)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy}")

# Predict on the test set
y_pred = (model.predict(test_images) > 0.5).astype("int32")

# Calculate evaluation metrics
print("Confusion Matrix:")
print(confusion_matrix(test_labels, y_pred))

print("Classification Report:")
print(classification_report(test_labels, y_pred, zero_division=0))
