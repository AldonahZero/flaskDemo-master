# coding:utf-8
from flask import jsonify, request, Blueprint, render_template, redirect, make_response, current_app
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

from algorithm.multiplePerspectives.A import grey_compare, canny_compare, lbp_compare, kaze_compare
from werkzeug.datastructures import FileStorage
import traceback
# from models.mul_model import MulModel

mul = Blueprint('mul', __name__)
mul_ns = Namespace('mul', description='multiplePerspectives 多视角')

# 文件上传格式
parser: RequestParser = mul_ns.parser()
parser.add_argument('file', location='files',
                    type=FileStorage, required=True)
# 上传图片路径
UPLOAD_PATH = get_upload_location(os.path.join('multiplePerspectives','static','images'))


# print(UPLOAD_PATH)

# 实际访问地址 /api/v1/mul/zipfile
@mul_ns.route("/zipfile", doc={"description": "灰度特征"})
class UploadHandler(Resource):
    def get(self):
        '''查看所有图片'''
        try:
            session = db_session()
            pics = session.query(Pic).all()
            data = []
            for pic in pics:
                data.append(pic.to_json())
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 201, 'message': '查找成功', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})

    @mul_ns.expect(parser, validate=True)
    def post(self):
        '''上传图片压缩包'''
        try:
            # 普通参数获取
            # 获取pichead文件对象
            file = request.files.get('file')
            save_filename = str(uuid.uuid1())
            path = os.path.join(UPLOAD_PATH, file.filename)
            # 保存压缩包
            file.save(path)
            # 解压缩
            unzip_file(path, UPLOAD_PATH)

            # 改名
            unzip_file_loaction = os.path.join(UPLOAD_PATH, file.filename)[0:-4]
            unzip_file_uid_loaction = os.path.join(UPLOAD_PATH, save_filename)
            os.rename(unzip_file_loaction, unzip_file_uid_loaction)
            # 删除压缩包
            remove_file_dir(path)

            # 前端路径
            proLoadPath = os.path.join('algorithm/multiplePerspectives/static/images', save_filename) + "/"
            realProLoadPath = os.path.join(UPLOAD_PATH, save_filename) + "/"
            print(realProLoadPath)
            filenames = [f for f in listdir(realProLoadPath) if isfile(join(proLoadPath, f))]

            session = db_session()
            for filename in filenames:
                new_file = Pic(url=proLoadPath + filename)
                session.add(new_file)
                session.commit()
            session.close()
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 201, 'message': '查找成功', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '上传压缩包成功'})


@mul_ns.route('/grey_compare/<pids>')
@mul_ns.param('pids', '图片id序号')
class rt_grey_compare(Resource):
    def get(self, pids):
        '''灰度特征'''
        # pids=31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60
        try:
            pidss = list(map(int, pids.split(',')))
            session = db_session()
            pics = Pic.query.filter(Pic.pid.in_(pidss)).all()
            # pics = session.query(Pic).filter(Pic.pid.in_(pids)).all()
            grey_compare_data = grey_compare(pics)
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': grey_compare_data})

@mul_ns.route('/canny_compare/<pids>')
@mul_ns.param('pids', '图片id序号')
class rt_canny_compare(Resource):
    def get(self, pids):
        '''边缘特征'''
        pidss = list(map(int, pids.split(',')))
        session = db_session()
        pics = Pic.query.filter(Pic.pid.in_(pidss)).all()
        current_app.logger.info(str(pics))
        # pics = session.query(Pic).filter(Pic.pid.in_(pids)).all()
        canny_compare_data = canny_compare(pics)
        # pids=31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60
        try:
            print()
            # pidss = list(map(int, pids.split(',')))
            # session = db_session()
            # pics = Pic.query.filter(Pic.pid.in_(pidss)).all()
            # current_app.logger.info(str(pics))
            # # pics = session.query(Pic).filter(Pic.pid.in_(pids)).all()
            # canny_compare_data = canny_compare(pics)
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': canny_compare_data})

@mul_ns.route('/lbp_compare/<pids>')
@mul_ns.param('pids', '图片id序号')
class rt_lbp_compare(Resource):
    def get(self, pids):
        '''lbp纹理特征'''
        # pids=31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60
        try:
            pidss = list(map(int, pids.split(',')))
            session = db_session()
            pics = Pic.query.filter(Pic.pid.in_(pidss)).all()
            # pics = session.query(Pic).filter(Pic.pid.in_(pids)).all()
            lbp_compare_data = lbp_compare(pics)
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': lbp_compare_data})

@mul_ns.route('/kaze_compare/<pids>')
@mul_ns.param('pids', '图片id序号')
class rt_kaze_compare(Resource):
    def get(self, pids):
        '''kaze角点特征'''
        # pids=31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60
        try:
            pidss = list(map(int, pids.split(',')))
            session = db_session()
            pics = Pic.query.filter(Pic.pid.in_(pidss)).all()
            # pics = session.query(Pic).filter(Pic.pid.in_(pids)).all()
            kaze_compare_data = kaze_compare(pics)
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': kaze_compare_data})