from flask import Flask, request, jsonify, render_template
import base64
import io
import cv2
from PIL import Image
import numpy as np
from detect import detect_objects
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the image file from the request
        image_file = request.files['image'].read()

        # Convert the image file to an OpenCV image
        image_np = np.fromstring(image_file, np.uint8)
        image2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Process the image if needed
        print(image2[0:50])
        image=  detect_objects(image2)
        # Convert the OpenCV image to a PIL image
        image_pil = Image.fromarray(image)

        # Convert the PIL image to a base64 string
        buffer = io.BytesIO()
        image_pil.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Return the base64 string to display the image
        
        #return jsonify(image_base64=image_base64)    #To GET base64 String As an OUTPUT
        return render_template('index.html', image_base64=image_base64) #To GET image as an output
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)