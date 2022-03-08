# coding:utf-8
from flask import jsonify, request, Blueprint, render_template, redirect, make_response
from flask_restx import Api, Resource, fields, Namespace
from flask_restx.reqparse import RequestParser
import os
from os.path import isfile, join
from os import listdir
import uuid

from common.mysql_operate import db_session, Pic
from common.file_tools import unzip_file
from common.getUploadLocation import get_upload_location
from common.remove_file_dir import remove_file_dir

from algorithm.multiplePerspectives.A import grey_compare, canny_compare
from werkzeug.datastructures import FileStorage

# from models.mul_model import MulModel

mul = Blueprint('mul',__name__)
mul_ns = Namespace('mul', description='multiplePerspectives 多视角')

# # 文件上传🚫
parser: RequestParser = mul_ns.parser()
parser.add_argument('file', location='files',
                    type=FileStorage, required=True)
# 上传图片路径
UPLOAD_PATH = get_upload_location("/multiplePerspectives/static/images")
# print(UPLOAD_PATH)

# 实际访问地址 /api/v1/mul/zipfile
@mul_ns.route("/zipfile", strict_slashes=False, doc={"description": "灰度特征"})
class UploadHandler(Resource):
    @mul_ns.doc(description="查看所有图片")
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
        return jsonify({'code': 201, 'message': '查找成功', 'data': data})

    @mul_ns.doc(description="上传图片压缩包")
    @mul_ns.doc(response={403: '上传失败'})
    @mul_ns.expect(parser, validate=True)
    def post(self):
        # 普通参数获取
        # 获取pichead文件对象
        file = request.files.get('file')
        save_filename = str(uuid.uuid1())
        path = os.path.join(UPLOAD_PATH, file.filename)
        file.save(path)
        # 解压缩

        unzip_file(path, UPLOAD_PATH)
        unzip_file_loaction =os.path.join(UPLOAD_PATH,  file.filename) [0:-4]
        unzip_file_uid_loaction = os.path.join(UPLOAD_PATH,  save_filename)
        os.rename(unzip_file_loaction,unzip_file_uid_loaction)
        remove_file_dir(path)

        # 前端路径
        proLoadPath = os.path.join('algorithm/multiplePerspectives/static/images', save_filename) + "/"
        realProLoadPath = os.path.join(UPLOAD_PATH, save_filename)+ "/"
        print(realProLoadPath)
        filenames = [f for f in listdir(realProLoadPath) if isfile(join(proLoadPath, f))]

        session = db_session()
        for filename in filenames:
            new_file = Pic(url=proLoadPath + filename)
            session.add(new_file)
            session.commit()
        session.close()
        return jsonify({'code': 201, 'message': '上传压缩包成功'})

    UPLOAD_PATH = os.path.join(os.path.dirname(__file__), '../static/images')

    @mul_ns.doc(description="根据图片调用算法")
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
