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

from algorithm.cutimg import gray_histogram_differential
from werkzeug.datastructures import FileStorage

# from models.mul_model import MulModel

fea = Blueprint('fea', __name__)
fea_ns = Namespace('fea', description='multiplePerspectives 多视角')

# 文件上传格式
parser: RequestParser = fea_ns.parser()
parser.add_argument('file', location='files',
                    type=FileStorage, required=True)
# 上传图片路径
CUTIMG_PATH = get_upload_location("/cutimg/static")


print(CUTIMG_PATH)

@fea_ns.route('/main_gray_hist_differential')
class rt_main_gray_hist_differential(Resource):
    def get(self):
        '''直方图图像'''
        # pids=31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60
        try:
            # path = 'static/images_GLCM_original'
            # path_bitwise = 'static/images_GLCM_bitwise'
            # path_gray_histogram_save = 'static/images_save/gray_histogram/'
            path = os.path.join(CUTIMG_PATH,'images_GLCM_original')
            path_bitwise = os.path.join(CUTIMG_PATH, 'images_GLCM_bitwise')
            path_gray_histogram_save = os.path.join(CUTIMG_PATH, 'images_save/gray_histogram')
            pic_url = gray_histogram_differential.main_gray_hist_differential(path=path, path_bitwise=path_bitwise, path_gray_histogram_save=path_gray_histogram_save)
        except BaseException as e:
            current_app.logger.error(str(e))
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': pic_url})