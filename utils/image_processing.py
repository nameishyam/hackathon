import cv2
import tensorflow as tf
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective, contours
import imutils
from PIL import Image

# Approach 1 functions
def load_model():
    model = tf.lite.Interpreter(model_path='models/modelinn7.tflite')
    model.allocate_tensors()
    return model

def approach1(img):
    model = load_model()
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    
    image = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    
    model.set_tensor(input_details[0]['index'], image)
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])
    probabilities = np.array(output_data[0])
    
    labels = {0: "ACCEPTED", 1: "REJECTED"}
    predicted_label = labels[np.argmax(probabilities)]
    color = "green" if predicted_label == "ACCEPTED" else "red"
    
    return predicted_label, color

# Approach 2 functions
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def approach2(image_np, width=6.33):
    ground_truth_width = 1.3
    ground_truth_length = 6.3
    threshold = 0.2
    
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None
    
    statuses = {'accept_reject': 'REJECT', 'width': '', 'length': ''}
    
    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue
        
        orig = image_np.copy()
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        
        (tl, tr, br, bl) = box
        tltrX, tltrY = midpoint(tl, tr)
        blbrX, blbrY = midpoint(bl, br)
        tlblX, tlblY = midpoint(tl, bl)
        trbrX, trbrY = midpoint(tr, br)
        
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / width
        
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric
        
        width_status = check_dimension(dimA, ground_truth_width, threshold, "Width")
        length_status = check_dimension(dimB, ground_truth_length, threshold, "Length")
        
        if width_status['status'] and length_status['status']:
            statuses['accept_reject'] = 'ACCEPT'
        else:
            statuses['accept_reject'] = 'REJECT'
        
        statuses['width'] = width_status
        statuses['length'] = length_status
        
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        cv2.putText(orig, f"{dimA:.2f}in", (int(tltrX - 15), int(tltrY - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        
        return Image.fromarray(orig), statuses
    
    return Image.fromarray(image_np), statuses

def check_dimension(value, ground_truth, threshold, name):
    if abs(value - ground_truth) <= threshold:
        return {'status': True, 'message': f"{name} correct"}
    elif value < ground_truth - threshold:
        return {'status': False, 'message': f"{name} too small"}
    else:
        return {'status': False, 'message': f"{name} too large"}