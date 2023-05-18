from flask import Flask, render_template, request
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import os
import torch

app = Flask(__name__)

model_path = 'mobilenet_v2.onnx'
ort_session = ort.InferenceSession(model_path)

def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = transform(image).unsqueeze(0)
    return image

def predict(image_path):
    try:
        image = preprocess_image(image_path)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        with torch.no_grad():
            output = ort_session.run([output_name], {input_name: image.numpy()})[0]
            predicted = torch.from_numpy(output)
            _, predicted = torch.max(predicted, 1)
            class_index = predicted.cpu().numpy()[0]
            with open('imagenet_classes.txt') as f:
                classes = [line.strip() for line in f.readlines()]
                prediction = classes[class_index]
                prediction = prediction.split(",")[1].strip().capitalize()
            return prediction
    except Exception as e:
        error_msg = "Error: Failed to preprocess or classify the image. Please ensure the image is in a supported format and try again."
        print(f"{error_msg}\n{e}")
        return error_msg

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    uploaded_image = None
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join('static', filename)
        file.save(file_path)
        uploaded_image = file_path
        prediction = predict(file_path)
    if request.args.get('clear') == 'True':
        uploaded_image = None
        prediction = None
    return render_template('index.html', prediction=prediction, image=uploaded_image)

if __name__ == '__main__':
    app.run(debug=True)
