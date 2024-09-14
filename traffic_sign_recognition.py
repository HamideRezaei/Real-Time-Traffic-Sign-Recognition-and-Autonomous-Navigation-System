import cv2
import numpy as np
import tensorflow as tf

# Load YOLO model (or any pre-trained traffic sign detection model)
model = tf.keras.models.load_model('yolo_traffic_sign_model.h5')

# Load class labels (e.g., stop sign, speed limit)
class_labels = ['Stop', 'Speed Limit', 'Yield', 'No Parking']

def preprocess_frame(frame):
    # Convert frame to appropriate input size and format for the model
    resized_frame = cv2.resize(frame, (416, 416))
    normalized_frame = resized_frame / 255.0
    return np.expand_dims(normalized_frame, axis=0)

def detect_traffic_signs(frame):
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)
    # Process predictions to extract bounding boxes and class labels
    # This will be specific to the model's output format
    return predictions

def draw_predictions(frame, predictions):
    for prediction in predictions:
        # Draw bounding boxes and labels on the frame
        cv2.rectangle(frame, (prediction['x'], prediction['y']), 
                      (prediction['x'] + prediction['w'], prediction['y'] + prediction['h']), 
                      (0, 255, 0), 2)
        cv2.putText(frame, class_labels[prediction['class_id']], 
                    (prediction['x'], prediction['y'] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    predictions = detect_traffic_signs(frame)
    frame_with_predictions = draw_predictions(frame, predictions)
    
    cv2.imshow('Traffic Sign Recognition', frame_with_predictions)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
