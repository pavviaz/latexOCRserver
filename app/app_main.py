import flask
import base64
import pix2tex
from flask import request


app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def getfunc():
    base64_photo = request.args["base64"]
    return str(base64_photo)
    png_recovered = base64.decodebytes(base64_photo)
    f = open("temp.png", "w")
    f.write(png_recovered)
    f.close()
    return pix2tex.main()
