import json
from flask import Blueprint, current_app
from flask_restx import Api
from flask import request, render_template, make_response, jsonify
from flask_restx import Namespace, Resource, fields
from config.setting import API_VERSION

# api1: 该模块路由
api_v1 = Blueprint('api1', __name__, url_prefix=API_VERSION)


class CustomApi(Api):
    def handle_error(self, e):
        for val in current_app.error_handler_spec.values():
            for handler in val.values():
                registered_error_handlers = list(filter(lambda x: isinstance(e, x), handler.keys()))
                if len(registered_error_handlers) > 0:
                    raise e
        return super().handle_error(e)

api = CustomApi(
    api_v1,
    title='平台 API',
    version='0.0.2-dev',
    doc='/',
    description='平台 API'
)


from api.swagger_demo import swag_ns
from api.featureExtraction import fea_ns
from api.featureExtraction2 import fea2_ns
from api.multiplePerspectives import mul_ns
from api.HSI import hsi_ns
from api.mosaic import mos_ns

api.add_namespace(swag_ns)
api.add_namespace(fea_ns)
api.add_namespace(fea2_ns)
api.add_namespace(mul_ns)
api.add_namespace(hsi_ns)
api.add_namespace(mos_ns)

cors_headers = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Allow-Credentials": "true",
    "Access-Control-Allow-Methods": "*"
}