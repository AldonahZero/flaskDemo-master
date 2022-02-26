import json
from flask import Blueprint
from flask_restx import Api
from flask import request, render_template, make_response, jsonify
from flask_restx import Namespace, Resource, fields
from config.setting import API_VERSION

# api1: 该模块路由
api_v1 = Blueprint('api1', __name__, url_prefix=API_VERSION)

api = Api(
    api_v1,
    version='1.0',
    title='平台 API',
    description='平台 API'
)



from api.swagger_demo import swag_ns
from api.featureExtraction import fea_ns
from api.multiplePerspectives import mul_ns
api.add_namespace(swag_ns)
api.add_namespace(fea_ns)
api.add_namespace(mul_ns)