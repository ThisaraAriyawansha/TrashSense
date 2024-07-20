import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet')

# Function to preprocess the frame
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

# Function to decode predictions
def decode_predictions(predictions):
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    return decoded[0][1], decoded[0][2]

# Function to categorize the predictions
def categorize_prediction(prediction):
    # Define your categories here
    category_mapping = {
        'Category 1': ['plastic_bag', 'water_bottle'],
        'Category 2': ['paper', 'cardboard'],
        'Category 3': ['metal_can', 'aluminum_foil'],
        'Category 4': ['organic', 'food_waste']
    }

    for category, class_names in category_mapping.items():
        if prediction in class_names:
            return category
    return 'Uncategorized'

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    # Make predictions
    predictions = model.predict(preprocessed_frame)
    prediction_class, prediction_prob = decode_predictions(predictions)

    # Categorize the prediction
    category = categorize_prediction(prediction_class)

    # Display the results
    cv2.putText(frame, f'Class: {prediction_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Prob: {prediction_prob:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Category: {category}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Garbage Classification', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
