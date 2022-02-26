# server api.py
import json
from flask import request, render_template, make_response, jsonify
from flask_restx import Namespace, Resource, fields

swag_ns = Namespace('swaggerUI_demo', description='swaggerUI 演示')


@swag_ns.route("/hello", strict_slashes=False)  # 实际访问地址 /api/v1/swaggerUI_demo/hello
class TestHandler(Resource):
    @swag_ns.doc('获取数据')
    @swag_ns.param('id', 'The task identifier')
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


@swag_ns.route("/hello2", strict_slashes=False)  # 实际访问地址 /api/test/hello2
class TestHandler(Resource):
    @swag_ns.doc('获取数据')
    @swag_ns.param('id', 'The task identifier')
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
