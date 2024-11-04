from flask import Flask, request, jsonify
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import io
from PIL import Image

app = Flask(__name__)

@app.route('/', methods=['POST'])

def upload_image():
    # Check if the request contains a file
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    
    # Get the image from the request
    file = request.files['image']
    
    # Check if an image file was uploaded
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    
     # Extract the file extension and check for supported formats
    _, file_extension = os.path.splitext(file.filename)
    file_extension = file_extension.lower()
    supported_formats = ['.png', '.jpg', '.jpeg']
    
    if file_extension not in supported_formats:
        return jsonify({"error": f"Unsupported file format: {file_extension}. Supported formats are: {supported_formats}"}), 400
    
    # Read the image file as a numpy array
    in_memory_file = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(in_memory_file, cv2.IMREAD_GRAYSCALE) # Directly load image as grayscale

    # image will be processed here 

    # Part A: Identifying corners of the paper for basic corner detection to flatten the image
    # By detecting the edges of the paper prior to finding the edges within the image we can apply
    # flattening techniques in a less intensive way prior to more complicated processing of objects
    # within the image itself.
    
    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply Canny Edge Detection to find edges of the document
    edges = cv2.Canny(blurred_image, 50, 150)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area and keep the largest one (assuming it's the document boundary)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    document_contour = None
    
    for contour in contours:
        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # If we have a four-sided polygon, we assume it's the document
        if len(approx) == 4:
            document_contour = approx
            break
    
    if document_contour is None:
        return jsonify({"error": "Could not detect document edges"}), 400
    
    # Transform perspective to get a top-down view of the document
    # Extract points from the contour and order them as required
    src_points = np.float32([point[0] for point in document_contour])
    dst_points = np.float32([[0, 0], [500, 0], [500, 500], [0, 500]])  # Square output size
    
    # Compute perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed = cv2.warpPerspective(image, matrix, (500, 500))  # Keep it grayscale
    
    # Save the transformed image as a JPEG to reduce size
    transformed_pil = Image.fromarray(transformed)
    transformed_byte_arr = io.BytesIO()
    transformed_pil.save(transformed_byte_arr, format='JPEG')  # Save as JPEG
    
    # Get byte data
    transformed_byte_arr.seek(0)
    
    return jsonify({
        "transformed_image": transformed_byte_arr.getvalue().hex(),
        "contour_points": src_points.tolist()  # Return contour points if further processing is needed
    })


    # # Step 2: Apply Harris Corner Detection
    # grayscale = np.float32(grayscale)
    # corners = cv2.cornerHarris(grayscale, blockSize=3, ksize=5, k=0.04)
    
    # # Dilate corner image to enhance corner points
    # corners = cv2.dilate(corners, None)
    
    # # Set threshold to identify strong corners
    # threshold = 0.02 * corners.max()
    
    
    # corner_image = image.copy()
    
    
    # image[corners > threshold] = [0, 0, 255]  # Highlight detected corners in red






    # # Step 3: Identify the four main corner points of the grid
    # # Assuming four significant points; here, use contour or clustering techniques to isolate these points
    # contours, _ = cv2.findContours((corners > threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # # Find the bounding box corners based on contour centroids or extreme points (this is an example)
    # approx_corners = []
    # for cnt in contours:
    #     # Approximate contour and find the centroid or extreme points
    #     if cv2.contourArea(cnt) > 100:  # Filter out small areas
    #         M = cv2.moments(cnt)
    #         if M["m00"] != 0:
    #             cx = int(M["m10"] / M["m00"])
    #             cy = int(M["m01"] / M["m00"])
    #             approx_corners.append([cx, cy])

    # # Sort detected corners to identify four main ones (assuming top-left, top-right, bottom-left, bottom-right)
    # if len(approx_corners) >= 4:
    #     approx_corners = np.float32(sorted(approx_corners, key=lambda x: (x[1], x[0])))[:4]  # Take only 4 corners
    #     src_points = np.float32(approx_corners)
    # else:
    #     # Fallback to using arbitrary points in case detection fails
    #     src_points = np.float32([[10, 10], [image.shape[1]-10, 10], [10, image.shape[0]-10], [image.shape[1]-10, image.shape[0]-10]])

    # dst_points = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    
    # # Step 4: Perspective Transformation
    # matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    # transformed = cv2.warpPerspective(image, matrix, (500, 500))

    # # Step 5: Prepare output for display
    # # Convert images to RGB for display
    # corner_image_rgb = cv2.cvtColor(corner_image, cv2.COLOR_BGR2RGB)
    # transformed_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)

    # # Convert to PIL Images for returning in response
    # original_pil = Image.fromarray(corner_image_rgb)
    # transformed_pil = Image.fromarray(transformed_rgb)

    # # Save the images to bytes
    # original_byte_arr = io.BytesIO()
    # transformed_byte_arr = io.BytesIO()
    # original_pil.save(original_byte_arr, format='JPG')
    # transformed_pil.save(transformed_byte_arr, format='JPG')

    # # Get byte data
    # original_byte_arr.seek(0)
    # transformed_byte_arr.seek(0)

    # return jsonify({
    #     "original_image": original_byte_arr.getvalue().hex(),
    #     "transformed_image": transformed_byte_arr.getvalue().hex(),
    #     "corners": corners.tolist()  # Return corner values for further processing if needed
    # })
    
if __name__ == '__main__':
    app.run(debug=True)





# from flask import Flask, request, jsonify
# from werkzeug.utils import secure_filename
# import os

# app = Flask(__name__)

# # Allowed file extensions for upload
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/', methods=['POST'])
# def upload_image():
#     # Check if the request contains a file
#     if 'image' not in request.files:
#         return jsonify({"error": "No image part in the request"}), 400

#     file = request.files['image']

#     # Check if an image file was uploaded
#     if file.filename == '':
#         return jsonify({"error": "No file selected for uploading"}), 400
    
#     # Check if the file has an allowed extension
#     if not allowed_file(file.filename):
#         return jsonify({"error": "Unsupported file format"}), 400

#     # Log the file details for debugging
#     filename = secure_filename(file.filename)
#     file_format = filename.rsplit('.', 1)[1].lower()
#     print(f"Received file: {filename}, Format: {file_format}")

#     # Additional image processing code would go here, e.g., corner detection, etc.
#     # Placeholder 2D array for response to confirm receipt of the file.
#     # crossword = [
#     #     ['C', 'A', 'T', '.', '.', '.'],
#     #     ['.', '.', 'O', '.', '.', '.'],
#     #     ['.', 'H', 'A', 'T', '.', '.'],
#     #     ['P', 'E', 'N', 'C', 'I', 'L'],
#     # ]
    
#     return jsonify({"message": "File received", "filename": filename, "file_format": file_format, "crossword": crossword})

# if __name__ == '__main__':
#     app.run(debug=True)




# @app.route('/', methods=['POST'])
# def upload_image():
#     # Check if the request contains a file
#     if 'image' not in request.files:
#         return jsonify({"error": "No image part in the request"}), 400
    
#     file = request.files['image']
    
#     # Check if an image file was uploaded
#     if file.filename == '':
#         return jsonify({"error": "No file selected for uploading"}), 400
    
#     # image will be processed here 
#     # crossword puzzle will be solved here 


#     # return a 2d array of a crossword puzzle 
#     # for now, dummy 2d array where . represents a grid item that cannot be filled 
#     crossword = [
#     ['C', 'A', 'T', '.', '.', '.'],
#     ['.', '.', 'O', '.', '.', '.'],
#     ['.', 'H', 'A', 'T', '.', '.'],
#     ['P', 'E', 'N', 'C', 'I', 'L'],

#     ]
#     for row in crossword:
#         print(' '.join(row))

#     print(crossword)
    
#     return jsonify(crossword)

# if __name__ == '__main__':
#     app.run(debug=True)









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
