import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import *
import time
import cv2
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils
import pyttsx3
import base64

st.title("Tool Anomaly Detection")
st.markdown("All Rights Reserved. © 2023.")

# Function for Approach 1
def main(image):
    model = "modelinn7.tflite"
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    image = np.array(image.resize((224, 224)), dtype=np.float32) 
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probabilities = np.array(output_data[0])

    labels = {0: "ACCEPTED", 1: "REJECTED"} 
    predicted_label = labels[np.argmax(probabilities)]
    color = "green" if predicted_label == "ACCEPTED" else "red"

    return f"<h1 style='text-align: center; color: {color} ; font-weight: bold;'>QUALITY DECISION (Approach 1): {predicted_label} </h1>"

# Function for Approach 2
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def process_image(image, width):
    # Define the ground truth dimensions
    ground_truth_width = 1.3  # inches
    ground_truth_length = 6.3  # inches

    start_time = time.time()    # Convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort the contours from left to right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    # Loop over the contours individually
    for c in cnts:
        # If the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue

        # Compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # Order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # Loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # Unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # Compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # Draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # Draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        # Compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # If the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to the supplied metric
        # (in this case, inches)
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / width

        # Compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # Define the threshold for accepting or rejecting the object
        threshold = 0.2  # inches

        # Check if the object matches the ground truth dimensions
        if abs(dimA - ground_truth_width) <= threshold and abs(dimB - ground_truth_length) <= threshold:
            accept_reject_text = "ACCEPT"
            accept_reject_color = "green"
            if abs(dimA - ground_truth_width) <= threshold:
                width_status = "✓ Correct width"
                width_status_colour = "green"
            else:
                width_status = "✕ Incorrect width"
                width_status_colour = "red"
            if abs(dimB - ground_truth_length) <= threshold:
                length_status = "✓ Correct length"
                length_status_colour = "green"
            else:
                length_status = "✕ Incorrect length"
                length_status_colour = "red"
        else:
            accept_reject_text = "REJECT"
            accept_reject_color = "red"
            if dimA < ground_truth_width - threshold:
                width_status = "✕ Width too small"
                width_status_colour = "red"
            elif dimA > ground_truth_width + threshold:
                width_status = "✕ Width too large"
                width_status_colour = "red"
            else:
                width_status = "✓ Width within range"
                width_status_colour = "green"
            if dimB < ground_truth_length - threshold:
                length_status = "✕ Length too small"
                width_status_colour = "red"
            elif dimB > ground_truth_length + threshold:
                length_status = "✕ Length too large"
                length_status_colour = "red"
            else:
                length_status = "✓ Length within range"
                length_status_colour = "green"

        # Draw the object sizes and class on the image
        # cv2.putText(orig, "{:.2f}in x {:.2f}in".format(dimA, dimB),
        cv2.putText(orig, "{:.2f}in".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)


        # # Display the image with measurements
        # st.image(orig, channels="BGR")

        # Display the acceptance/rejection status and reasons
        st.markdown(f"<h1 style='color:{accept_reject_color};'>{accept_reject_text}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='color:{width_status_colour};'>{width_status}</h1>", unsafe_allow_html=True)
        # st.markdown(f"<h1 style='color:{length_status_colour};'>{length_status}</h1>", unsafe_allow_html=True)


        # st.markdown(f"- {width_status}")
        # st.markdown(f"- {length_status}")
        end_time = time.time()
                 # Text-to-speech
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)  # Adjust the speaking rate
        if accept_reject_text == "ACCEPT":
            engine.say("ACCEPT")
        else:
            engine.say("REJECT")
        engine.runAndWait()

        return orig

# Upload image file or take a photo
img_file_buffer = st.file_uploader("Upload Image File (PNG, JPG, or JPEG)", type=["png", "jpg", "jpeg"])
if img_file_buffer is None:
    img_file_buffer = st.camera_input("Take a Photo")

if img_file_buffer is not None:
    # Read image file buffer as a PIL Image
    img = Image.open(img_file_buffer)

    # Convert PIL Image to bytes
    img_bytes = img_file_buffer.getvalue()

    # Save the image to a temporary file
    with open("captured_image.jpg", "wb") as f:
        f.write(img_bytes)

    # Display the uploaded or captured image
    st.image(img, caption='Uploaded/Captured Image', use_column_width=True)

    # Process image for Approach 1
    with st.spinner('Approach 1 working....'):
        start_time_approach_1 = time.time()
        result_approach_1 = main(img)
        end_time_approach_1 = time.time()
        st.markdown(result_approach_1, unsafe_allow_html=True)
        st.markdown(f"<h3>Processing time (Approach 1): {end_time_approach_1 - start_time_approach_1:.3f} seconds</h3>", unsafe_allow_html=True)

    # Process image for Approach 2
    with st.spinner('Approach 2 working....'):
        start_time_approach_2 = time.time()
        # Set the fixed width for Approach 2
        width_approach_2 = 6.33  # inches
        result_approach_2 = process_image(np.array(img), width_approach_2)
        end_time_approach_2 = time.time()
        st.image(result_approach_2, channels="BGR")
        st.markdown(f"<h3>Processing time (Approach 2): {end_time_approach_2 - start_time_approach_2:.3f} seconds</h3>", unsafe_allow_html=True)

    # Summary
    summary_text = ""
    if "ACCEPTED" in result_approach_1 and "ACCEPT" in result_approach_2:
        summary_text = "Tool is ACCEPTED based on both approaches."
    else:
        summary_text = "Tool is REJECTED based on at least one approach."

    st.markdown(f"<h2>{summary_text}</h2>", unsafe_allow_html=True)