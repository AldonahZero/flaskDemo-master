from flask import Flask, jsonify, request,Blueprint

cutting=Blueprint('cutting',__name__)

@cutting.route('/')
def hello_world():
    return 'Hello cutting!'

