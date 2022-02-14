from flask import Flask, jsonify, request,Blueprint
from algorithm.cutimg.cutimg import mycutimg
import cv2

cutimg=Blueprint('cutimg',__name__)

@cutimg.route('/')
def hello_world():
    return 'Hello cutting!'

@cutimg.route('/upload')
def upload():
    return "upload"

@cutimg.route('/cutting')
def cutting():
    img_input = cv2.imread('E:\back_dev_flask\algorithm\cutimg\static\images_GLCM_original\images_camouflage\mix\20m\1.JPG')
    mycutimg(img_input)
