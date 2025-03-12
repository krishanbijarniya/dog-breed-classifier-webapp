from flask import Flask, request, render_template, jsonify
import torch
import os
from model import load_model1, load_model2, preprocess_image

app = Flask(__name__)

# Load both models
model1 = load_model1()
model2 = load_model2()

# Route to UI
@app.route('/')
def index():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file
    image_path = os.path.join("static", file.filename)
    file.save(image_path)

    # Preprocess Image
    image = preprocess_image(image_path)

    # Get predictions
    with torch.no_grad():
        output1 = model1(image)
        output2 = model2(image)

    # Convert output to readable format
    pred1 = torch.argmax(output1, dim=1).item()
    pred2 = torch.argmax(output2, dim=1).item()

    return jsonify({'model1_prediction': pred1, 'model2_prediction': pred2, 'image_path': image_path})

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
