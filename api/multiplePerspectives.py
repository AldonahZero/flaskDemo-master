# coding:utf-8
from flask import jsonify, request,Blueprint, render_template, redirect,make_response
from flask_restx import Api, Resource, fields,Namespace
import os
from os.path import isfile, join
from os import listdir
from common.mysql_operate import db_session, Pic
from common.file_tools import unzip_file

from algorithm.multiplePerspectives.A import grey_compare, canny_compare

# mul = Blueprint('mul',__name__)
mul_ns = Namespace('mul', description='multiplePerspectives 多视角')


UPLOAD_PATH = os.path.join(os.path.dirname(__file__), '../static')
print(UPLOAD_PATH)
@mul_ns.route("/zipfile", strict_slashes=False,doc={"description": "type in here"})  # 实际访问地址 /api/v1/mul/zipfile
class UploadHandler(Resource):
    @mul_ns.doc('查看所以解压图片')
    @mul_ns.doc(params={"id": "An ID", "description": "My resource"})
    @mul_ns.doc(response={403: '查看失败'})
    def get(self):
        # 普通参数获取
        session = db_session()
        pics = session.query(Pic).all()
        # json.dumps(data, cls=MyEncoder)
        # print(pics.__dict__)
        data = []
        for pic in pics:
            data.append(pic.to_json())
        # print(data)
        return jsonify({'code': 1, 'message': '上传成功', 'data': data})

    @mul_ns.doc('上传图片压缩包')
    @mul_ns.doc(response={403: '上传失败'})
    @mul_ns.param('file', '文件')
    def post(self):
        # 普通参数获取
        # 获取pichead文件对象
        file = request.files.get('file')
        path = os.path.join(UPLOAD_PATH, file.filename)
        print(path)
        file.save(path)

        # 解压缩

        unzip_file(path, UPLOAD_PATH)

        # 前端路径
        loadpath = ('/static/images/%s' % file.filename)[0:-4] + "/"
        filenames = [f for f in listdir(loadpath) if isfile(join(loadpath, f))]

        session = db_session()
        for filename in filenames:
            new_file = Pic(url=loadpath + filename)
            session.add(new_file)
            session.commit()
        session.close()
        return '上传压缩包成功', 201

    UPLOAD_PATH = os.path.join(os.path.dirname(__file__), '../static/images')

    @mul_ns.doc('根据图片调用算法')
    def put(self):
        # 普通参数获取
        pids_str = request.values.get('pids').split(',')
        pids = list(map(int, pids_str))
        session = db_session()
        pics = Pic.query.filter(Pic.pid.in_(pids)).all()
        # pics = session.query(Pic).filter(Pic.pid.in_(pids)).all()

        data = grey_compare(pics)
        return jsonify({'code': 1, 'message': '上传成功', 'data': data})

    def delete(self):
        pass

#
# class TestHandler(Resource):
#     # 描述你的这个方法的作用
#     @swag_ns.doc('获取数据')
#     @swag_ns.param('id', 'The task identifier')
#     def get(self):
#         # 如果使用模板的块，需要使用 make_response
#         # return make_response(render_template('index.html', data=res), 200)
#
#         # 使用 jsonify 是为了返回json数据的同时，相比于 json.dumps() 其会自动修改 content-type 为 application/json
#         # 另外，如果使用 jsonify()的同时，还想自定义返回状态码，可以使用 make_response(jsonify(data=data), 201)
#         return jsonify("1")
#
#     def post(self):
#         pass
#
#     def put(self):
#         pass
#
#     def delete(self):
#         pass
