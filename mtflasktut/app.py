from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/upload-image', methods=['POST'])
def upload_image():
    # Check if the request contains a file
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    
    file = request.files['image']
    
    # Check if an image file was uploaded
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    
    # image will be processed here 
    # crossword puzzle will be solved here 


    # return a 2d array of a crossword puzzle 
    # for now, dummy 2d array where . represents a grid item that cannot be filled 
    crossword = [
    ['C', 'A', 'T', '.', '.', '.'],
    ['.', '.', 'O', '.', '.', '.'],
    ['.', 'H', 'A', 'T', '.', '.'],
    ['P', 'E', 'N', 'C', 'I', 'L'],

    ]
    for row in crossword:
        print(' '.join(row))

    print(crossword)
    
    return jsonify(crossword)

if __name__ == '__main__':
    app.run(debug=True)


# import os
# from flask import Flask, flash, request, redirect, url_for
# from werkzeug.utils import secure_filename

# cwd = os.getcwd()
# cwd += '/images'

# UPLOAD_FOLDER = cwd
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# app = Flask(__name__)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/uploads/<name>')
# def download_file(name):
#     return f'File uploaded: {name}'

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # If the user does not select a file, the browser submits an
#         # empty file without a filename.
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return redirect(url_for('download_file', name=filename))
#     return '''
#     <!doctype html>
#     <title>Upload new File</title>
#     <h1>Upload new File</h1>
#     <form method=post enctype=multipart/form-data>
#       <input type=file name=file>
#       <input type=submit value=Upload>
#     </form>
#     '''