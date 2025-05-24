from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import cv2



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model
model = YOLO('best.pt')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Run YOLO
            results = model(file_path, device='cpu')
            annotated_frame = results[0].plot()  # Get annotated image (numpy array)

            # Save output image
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
            cv2.imwrite(output_path, annotated_frame)

            return render_template('index.html', result_image='uploads/result.jpg')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
