# server api.py
import json
from flask import Blueprint
from flask_restx import Api
from flask import request, render_template, make_response, jsonify
from flask_restx import Namespace, Resource, fields


api_v1 = Blueprint('api1', __name__, url_prefix='/api')

api = Api(
    api_v1,
    version='1.0',
    title='平台 API',
    description='平台 API'
)


ns = Namespace('test', description='test')


@ns.route("/hello", strict_slashes=False)  # 实际访问地址 /api/test/hello
class TestHandler(Resource):
    @ns.doc('获取数据')
    @ns.param('id', 'The task identifier')
    def get(self):
        # 如果使用模板的块，需要使用 make_response
        # return make_response(render_template('index.html', data=res), 200)

        # 使用 jsonify 是为了返回json数据的同时，相比于 json.dumps() 其会自动修改 content-type 为 application/json
        # 另外，如果使用 jsonify()的同时，还想自定义返回状态码，可以使用 make_response(jsonify(data=data), 201)
        return jsonify("1")

    def post(self):
        pass

    def put(self):
        pass

    def delete(self):
        pass


@ns.route("/hello2", strict_slashes=False)  # 实际访问地址 /api/test/hello2
class TestHandler(Resource):
    @ns.doc('获取数据')
    @ns.param('id', 'The task identifier')
    def get(self):
        # 如果使用模板的块，需要使用 make_response
        # return make_response(render_template('index.html', data=res), 200)

        # 使用 jsonify 是为了返回json数据的同时，相比于 json.dumps() 其会自动修改 content-type 为 application/json
        # 另外，如果使用 jsonify()的同时，还想自定义返回状态码，可以使用 make_response(jsonify(data=data), 201)
        return jsonify("2")

    def post(self):
        pass

    def put(self):
        pass

    def delete(self):
        pass

api.add_namespace(ns)