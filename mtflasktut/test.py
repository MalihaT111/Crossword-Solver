from flask import Flask, request, jsonify, flash, redirect, url_for
import os
from werkzeug.utils import secure_filename


upload_folder = os.getcwd() + '/images' # os.getcwd() gets the curr working directory
extensions = {'png', 'jpg', 'jpeg'} # valid file extensions 
app = Flask(__name__)
app.config['upload_folder'] = upload_folder

def allowed_file(filename):
    return '.' in filename and \
        # we are not excepting anything other than images, 
           filename.rsplit('.', 1)[1].lower() in extensions

@app.route('/uploads/<name>')
def download_file(name):
    # visual cue of downloaded file 
    return f'File uploaded: {name}'

@app.route('/upload-image', methods=['GET','POST'])
def upload_image():
    if request.method == 'POST':
        # Check if the request contains a file
        if 'image' not in request.files:
            flash('No file part')
            redirect(request.url)
            # return jsonify({"error": "No image part in the request"}), 400
        
        file = request.files['image']
        
        # Check if an image file was uploaded
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            # return jsonify({"error": "No file selected for uploading"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename))

        # if file and allowed_file(file.filename):
        #     # image will be processed here 
        #     # crossword puzzle will be solved here 


        #     # return a 2d array of a crossword puzzle 
        #     # for now, dummy 2d array where . represents a grid item that cannot be filled 
        #     crossword = [
        #     ['C', 'A', 'T', '.', '.'],
        #     ['.', '.', 'O', '.', 'D'],
        #     ['.', 'H', 'A', 'T', '.'],
        #     ['P', 'E', 'N', 'C', 'I'],
        #     ['.', '.', 'E', '.', '.']
        #     ]
        #     for row in crossword:
        #         print(' '.join(row))

        #     print(crossword)
            
        #     return jsonify(crossword)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)






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