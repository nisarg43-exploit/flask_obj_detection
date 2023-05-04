import cv2
import numpy as np

# Load the COCO class names
class_names = []
with open('required_data\coco.names', 'r') as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Load the pre-trained COCO model
net = cv2.dnn.readNet('required_data\yolov3.weights', 'required_data\yolov3.cfg')

# Specify the input image dimensions
input_size = (416, 416)

# Define a function to detect objects in an image
def detect_objects(image):
    # Create a blob from the input image
    blob = cv2.dnn.blobFromImage(image, 1/255, input_size, swapRB=True, crop=False)

    # Set the input blob for the neural network
    net.setInput(blob)

    # Run forward inference to get the network output
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)

    # Get the bounding boxes, confidence scores, and class IDs
    boxes = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                width = int(detection[2] * image.shape[1])
                height = int(detection[3] * image.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes and labels on the input image
    for i in indices:
        #i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        label = f'{class_names[class_ids[i]]}: {confidences[i]:.2f}'
        color = (0, 255, 0)
        cv2.rectangle(image, (left, top), (left+width, top+height), color, 2)
        cv2.putText(image, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

