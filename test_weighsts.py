import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, boxes, classes, scores, threshold=0.1):
    for box, _class, score in zip(boxes, classes, scores):
        if score < threshold:
            continue

        y_min, x_min, y_max, x_max = box
        x_min, x_max, y_min, y_max = (x_min * image.shape[1], x_max * image.shape[1], 
                                      y_min * image.shape[0], y_max * image.shape[0])

        # Draw rectangle (bounding box)
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        # Draw label and score
        label = f'Class: {_class}, Score: {score:.2f}'
        cv2.putText(image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Step 1: Load the pre-trained MobileNetV2 SSD model
model = tf.saved_model.load('./results_model/saved_model/')

# Step 2: Load and preprocess an image
img_path = '/edgar1/datase_license_plate/test/0123f836ce94678a_jpg.rf.c2c72263ce513bd9a8b56d2a162e1950.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions using the model
concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
tensor_input = tf.convert_to_tensor(img_array, dtype=tf.float32)
tensor_input = tf.cast(tensor_input, tf.uint8)
predictions = concrete_func(tensor_input)
print(predictions)
# Extract prediction data
boxes = predictions['detection_boxes'].numpy()[0]  # Bounding box coordinates
classes = predictions['detection_classes'].numpy()[0]  # Class IDs
scores = predictions['detection_scores'].numpy()[0]  # Confidence scores

# Load the original image (not preprocessed) for drawing
original_image = cv2.imread(img_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Draw bounding boxes on the original image
draw_bounding_boxes(original_image, boxes, classes, scores)

# Display the image with bounding boxes
plt.imshow(original_image)
plt.axis('off')
plt.show()
