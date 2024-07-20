from flask import Flask, render_template, Response, jsonify, request
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('garbage_classification_model.h5')

# Define the class labels
class_labels = [
    'battery', 'biological', 'brown glass', 'cardboard', 'clothes', 'green glass',
    'metal', 'paper', 'plastics', 'shoes', 'trash', 'white glass'
]

# Define the groups and map each class to a group
class_groups = {
    'battery': 'Hazardous',
    'biological': 'Organic',
    'brown glass': 'Recyclable',
    'cardboard': 'Recyclable',
    'clothes': 'Non-Recyclable',
    'green glass': 'Recyclable',
    'metal': 'Recyclable',
    'paper': 'Recyclable',
    'plastics': 'Recyclable',
    'shoes': 'Non-Recyclable',
    'trash': 'Non-Recyclable',
    'white glass': 'Recyclable'
}

# Define a function to preprocess the frame
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (150, 150))
    frame_array = img_to_array(frame_resized)
    frame_array = np.expand_dims(frame_array, axis=0)
    frame_array /= 255.0
    return frame_array

# Video streaming generator function
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_array = preprocess_frame(frame)
        predictions = model.predict(frame_array)
        class_idx = np.argmax(predictions, axis=1)[0]

        try:
            class_label = class_labels[class_idx]
            group_label = class_groups[class_label]
        except IndexError:
            class_label = "Unknown"
            group_label = "Unknown"

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        global latest_classification
        latest_classification = {
            'class': class_label,
            'group': group_label
        }

    cap.release()

@app.route('/latest_classification')
def get_latest_classification():
    return jsonify(latest_classification)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img_array = preprocess_frame(img)
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions, axis=1)[0]
        
        try:
            class_label = class_labels[class_idx]
            group_label = class_groups[class_label]
        except IndexError:
            class_label = "Unknown"
            group_label = "Unknown"

        return jsonify({'class': class_label, 'group': group_label})
    return jsonify({'class': 'Unknown', 'group': 'Unknown'})

if __name__ == '__main__':
    latest_classification = {
        'class': 'Unknown',
        'group': 'Unknown'
    }
    app.run(debug=True)
