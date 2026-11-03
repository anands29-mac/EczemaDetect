import os
import cv2
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Set paths and initialize variables
data_path = 'data/'  # Make sure to create a 'data' folder with subfolders for each class
images, labels = [], []

# Load and preprocess data
for category in os.listdir(data_path):
    category_path = os.path.join(data_path, category)
    for img_file in os.listdir(category_path):
        img = cv2.imread(os.path.join(category_path, img_file))
        img = cv2.resize(img, (128, 128))
        images.append(img)
        labels.append(category)

images = np.array(images) / 255.0
labels = np.array(labels)

# Convert labels to numeric values
unique_labels = list(set(labels))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
labels = np.array([label_map[label] for label in labels])

# Split data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(unique_labels), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Save model
model.save('eczema_cnn_model.h5')

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# New survey questions
def eczema_survey():
    print("Please answer the following questions for eczema assessment:")
    gender = input("Are you male or female? ")
    age = input("What is your age? ")
    duration = input("How long have you been experiencing skin irritation? (days/weeks/months) ")
    redness = input("Do you experience redness on the affected area? (No/Mild/Severe) ")
    itchiness = input("Is the affected area itchy? (No/Mild/Severe) ")
    dryness = input("Have you noticed any dryness or flakiness? (No/Mild/Severe) ")
    family_history = input("Do you have a family history of eczema or skin allergies? (Yes/No) ")
    health_conditions = input("Do you have any underlying health conditions like asthma or hay fever? (Yes/No) ")
    return [gender, age, duration, redness, itchiness, dryness, family_history, health_conditions]

eczema_survey()