import flask
import base64
from app import app
import pix2tex
from flask import request
from PIL import Image

base64_photo = ""

@app.route('/', methods=['GET'])
def getfunc():
    base64_photo = request.args["base64"]
    return "HelloGET + " + str(base64_photo)
    png_recovered = base64.decodebytes(base64_photo)
    f = open("temp.png", "w")
    f.write(png_recovered)
    f.close()
    return pix2tex.main()
    
@app.route('/upload_image', methods=['POST'])
def postfunc():
    file = request.files['image']
    img = Image.open(file.stream)
    file.save("result.png")
    return "saved!!!"
    base64_photo = request.args["base64"]
    return "Hello + " + str(base64_photo)
