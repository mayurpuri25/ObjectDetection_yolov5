from io import BytesIO
from flask import Flask, render_template, request, url_for
import numpy as np
import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os

app = Flask(__name__)

os.environ['AZURE_STORAGE_CONNECTION_STRING'] = 'DefaultEndpointsProtocol=https;AccountName=objectdetectionimage;AccountKey=g7t7i71TIUhgTHZ5Ql3V6IwHzdWQUHu02maWwAFG8FKj3DwUt5m3CAw0k2flwV80d/TTzxxGEvz5+AStDfsjig==;EndpointSuffix=core.windows.net'
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING') 
container_name = "images"
blob_service_client = BlobServiceClient.from_connection_string(conn_str=connect_str) # create a blob service client to interact with the storage account

try:
    container_client = blob_service_client.get_container_client(container = container_name)
    container_client.get_container_properties()
except Exception as e:
    container_client = blob_service_client.create_container(container_name)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/', methods=["POST"])
def predict():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return 'No image part in the request', 400
        
        imagefile = request.files['imagefile']

        # If no file was selected
        if imagefile.filename == '':
            return 'No selected image', 400
        
        try:
            container_client.upload_blob(imagefile.filename, imagefile)
        except Exception as e:
            print(e)
        
        blob_client = blob_service_client.get_blob_client(container_name, imagefile.filename)

        blob_url = blob_client.url
        # Make a GET request to the image URL
        response = requests.get(blob_url)

        # Make sure the request was successful
        response.raise_for_status()

        # Create a BytesIO object from the response content
        image_data = BytesIO(response.content)

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        # Open the image with PIL
        im1 = Image.open(image_data)

        # Inference
        results = model([im1], size=640) # batch of images


        # Render results on the original image
        results.render()  # This updates results.imgs with boxes and labels
        
        # After detection and rendering
        results_img = Image.fromarray(results.render()[0])

        # Save the image to a BytesIO object
        image_io = BytesIO()
        results_img.save(image_io, format='JPEG')

        # Seek to the start of the BytesIO object
        image_io.seek(0)

        # Create a new blob client for the results image
        result_blob_name = imagefile.filename.split('.')[0] + '_result.jpeg'
        result_blob_client = blob_service_client.get_blob_client(container_name, result_blob_name)

        # Upload the BytesIO object as a blob
        result_blob_client.upload_blob(image_io, overwrite=True)

        # Get the URL of the results image blob
        result_blob_url = result_blob_client.url

        return render_template('index.html', image_path=result_blob_url)
    else :
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)