import os
import tempfile
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from infer import infer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Get ORIN_IP from environment variable or use default
ORIN_IP = os.getenv('ORIN_IP', 'localhost')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image (png, jpg, jpeg, gif, bmp)'}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        # Run inference
        logits, pred = infer(temp_path, ORIN_IP)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        # Prepare response
        result = {
            'success': True,
            'logits': logits.tolist(),
            'prediction': int(pred),
            'label': 'HGG' if pred == 1 else 'LGG',
            'confidence': float(logits[pred])
        }
        
        return jsonify(result)
    
    except Exception as e:
        # Clean up on error
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

