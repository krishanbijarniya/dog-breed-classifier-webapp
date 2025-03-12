from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import yaml
import base64


from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras import layers
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def build_resnet_model():
    resnet50_v2 = ResNet50V2(
    include_top=False,
    input_shape=(128, 128, 3)
    )

    for layer in resnet50_v2.layers:
        layer.trainable = False

    # Input and output from the pre-trained model
    model_input = resnet50_v2.input
    hidden = resnet50_v2.output

    # Flatten the output from ResNet
    hidden = layers.Flatten()(hidden)

    # Output layer for classification
    output = layers.Dense(units=120, activation='softmax')(hidden)

    # Create the model
    model = Model(inputs=model_input, outputs=output, name='ResNet50V2')
    return model

# class Detection:
#     def __init__(self):
#         self.model = YOLO(r"Model/yolo8.pt")

#     def predict(self, img, classes=[], conf=0.5):
#         if classes:
#             results = self.model.predict(img, classes=classes, conf=conf)
#         else:
#             results = self.model.predict(img, conf=conf)

#         return results

#     def predict_and_detect(self, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
#         results = self.predict(img, classes, conf=conf)
#         for result in results:
#             for box in result.boxes:
#                 cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
#                               (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
#                 cv2.putText(img, f"{result.names[int(box.cls[0])]}",
#                             (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
#                             cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
#         return img, results

#     def detect_from_image(self, image):
#         result_img, _ = self.predict_and_detect(image, classes=[], conf=0.5)
#         return result_img


#detection = Detection()

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv8 model
yolo_model = YOLO("Model/yolo8.pt")  # Make sure this path is correct
yolo_model.compile()

# Load ResNet50 model
resnet_model = load_model('Model/ResNet50_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    model_type = request.form.get('model')
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    if model_type == 'ResNet50':
        return classify_resnet(filepath)
    elif model_type == 'YOLOv8':
        return detect_yolo(filepath)
    else:
        return jsonify({'error': 'Invalid model selection'})


def classify_resnet(filepath):
    image = Image.open(filepath).resize((224, 224))
    image = np.expand_dims(preprocess_input(np.array(image)), axis=0)
    preds = resnet_model.predict(image)
    label = decode_predictions(preds, top=1)[0][0][1]
    return jsonify({'model': '1','label': label})


def detect_yolo(filepath):
    label = ''
    with open('data.yaml', 'r') as f:
     data_yaml = yaml.safe_load(f)

    class_names = data_yaml['names']

    print(class_names)

    img = cv2.imread(filepath)
    if img is None:
        return jsonify({'error': 'Failed to load image'}), 400

    print(f"Image shape: {img.shape}")  # Debugging

    img = cv2.resize(img, (640, 640))  # Resize image for YOLOv8
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB


    print('Model loaded')
    # Run YOLOv8 inference
    results = yolo_model.predict(img_rgb)
    print('result predicted')

    if results and len(results[0].boxes.xyxy) > 0:
    # Draw bounding boxes and add predicted class text
        for box, cls in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box[:4])
            class_id = int(cls)  # Get the class ID
            class_name = class_names[class_id]  # Get the class name
            
            # Draw the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Put the predicted class label on the bounding box
            label = f"{class_name}"  # Class name
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            label_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            label_x = x1
            label_y = y1 - 10  # Position the text slightly above the box
            
            # Draw label background (optional)
            cv2.rectangle(img, (label_x, label_y - label_size[1]), (label_x + label_size[0], label_y), (255, 0, 0), -1)
            
            # Put the class name text
            cv2.putText(img, label, (label_x, label_y), font, font_scale, (255, 255, 255), font_thickness)

    else:
        return jsonify({'error': 'No objects detected in the image'}), 400

    # Convert the image with bounding boxes to PNG and save it to a buffer
    # output = Image.fromarray(img)
    # buf = io.BytesIO()
    # output.save(buf, format="PNG")
    # buf.seek(0)

    # # Save image to static folder so it can be accessed by the frontend
    # image_url = "/static/output_image.png"
    # with open("static/output_image.png", "wb") as f:
    #     f.write(buf.getvalue())

    # Convert the image to base64
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    #os.remove(file_path)  # Clean up uploaded file
    print(jsonify({'output_image': img_base64, 'label': label}))
    return jsonify({'model': '2','output_image': img_base64, 'label': label})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

