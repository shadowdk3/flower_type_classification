from flask import Flask, request, render_template, jsonify

import os
from src.pipeline.predict_pipeline import PredicPipeline

application = Flask(__name__)

app = application

@app.route('/predictdata', methods=['GET', 'POST'])
def update_image():
    result = "" 
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', result = 'No file part')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', result = 'No selected file')
        
        if file:
            upload_folder = os.path.join(app.root_path, 'templates/static')
            if not os.path.isdir(upload_folder):
                os.makedirs(upload_folder)
                
            file.save(os.path.join(upload_folder, 'new_image.jpg'))
            
            predict_pipeline = PredicPipeline()
            result = predict_pipeline.predict(os.path.join(upload_folder, 'new_image.jpg'))

    return render_template('index.html', result = result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)