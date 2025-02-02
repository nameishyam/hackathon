from flask import Flask, render_template, request, send_from_directory
import os
from PIL import Image
import numpy as np
from utils.image_processing import approach1, approach2
from utils.tts import generate_audio
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/uploads'
app.config['PROCESSED_FOLDER'] = 'static/images/processed'
app.config['AUDIO_FOLDER'] = 'static/audio'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Process uploaded file
        file = request.files['file']
        if file:
            # Save original image
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original.jpg')
            file.save(img_path)
            
            # Process approaches
            img = Image.open(img_path)
            results = process_image(img)
            
            # Generate audio
            audio_path = os.path.join(app.config['AUDIO_FOLDER'], 'result.mp3')
            generate_audio(results['summary_text'], audio_path)
            
            return render_template('index.html', results=results)
    
    return render_template('index.html')

def process_image(img):
    results = {}
    
    # Approach 1
    start_time_1 = time.time()
    approach1_result, approach1_color = approach1(img)
    results['approach1'] = {
        'result': approach1_result,
        'color': approach1_color,
        'time': time.time() - start_time_1
    }
    
    # Approach 2
    start_time_2 = time.time()
    approach2_result, statuses = approach2(np.array(img))
    processed_img_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed.jpg')
    approach2_result.save(processed_img_path)
    
    results['approach2'] = {
        'image': processed_img_path,
        'statuses': statuses,
        'time': time.time() - start_time_2
    }
    
    # Generate summary
    if "ACCEPTED" in approach1_result and "ACCEPT" in statuses['accept_reject']:
        results['summary_text'] = "Tool is ACCEPTED based on both approaches."
    else:
        results['summary_text'] = "Tool is REJECTED based on at least one approach."
    
    return results

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)