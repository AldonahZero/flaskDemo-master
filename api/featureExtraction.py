# coding:utf-8
from flask import jsonify, request, Blueprint, render_template, redirect, make_response, current_app, url_for
from flask_restx import Api, Resource, fields, Namespace
from flask_restx.reqparse import RequestParser

import os
from os.path import isfile, join
from os import listdir
import uuid

from common.mysql_operate import db_session, Pic
from common.file_tools import unzip_file
from common.getUploadLocation import get_upload_location, get_server_location
from common.remove_file_dir import remove_file_dir
from common.find_star_end import find_se_com

from algorithm.cutimg import gray_histogram_differential, main_color_demon, edge_batch
from werkzeug.datastructures import FileStorage
import traceback

# from models.mul_model import MulModel

fea = Blueprint('fea', __name__)
fea_ns = Namespace('fea', description='multiplePerspectives 多视角')

# 文件上传格式
parser: RequestParser = fea_ns.parser()
parser.add_argument('file', location='files',
                    type=FileStorage, required=True)
# 上传图片路径
CUTIMG_PATH = get_upload_location("/cutimg/static")
# /Users/aldno/Downloads/flaskDemo-master/algorithm/cutimg/static
# 服务器图片路径
CUTIMG_SERVER_PATH = get_server_location("/cutimg/static")
#  /algorithm/cutimg/static

@fea_ns.route('/main_gray_hist_differential')
class rt_main_gray_hist_differential(Resource):
    def get(self):
        '''直方图图像'''
        current_app.logger.info(CUTIMG_PATH)
        try:
            # path = 'static/images_GLCM_original'
            # path_bitwise = 'static/images_GLCM_bitwise'
            # path_gray_histogram_save = 'static/images_save/gray_histogram/'
            path = os.path.join(CUTIMG_PATH,'images_GLCM_original')
            path_bitwise = os.path.join(CUTIMG_PATH, 'images_GLCM_bitwise')
            path_gray_histogram_save = os.path.join(CUTIMG_PATH, 'images_save/gray_histogram/')
            gray_histogram_differential.main_gray_hist_differential(path=path, path_bitwise=path_bitwise, path_gray_histogram_save=path_gray_histogram_save)
            pic_url = os.path.join(CUTIMG_SERVER_PATH, 'images_save/gray_histogram/1.JPG')
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': pic_url})

@fea_ns.route('/mymain_color/<mymain_color_id>')
@fea_ns.param('mymain_color_id', '图片路径')
class rt_mymain_color(Resource):
    def get(self,mymain_color_id):
        '''提取主色'''
        try:
            pid = int(mymain_color_id)
            session = db_session()
            pics = session.query(Pic).filter(Pic.pid ==  pid ).first()

            real_mymain_color_path = find_se_com(CUTIMG_PATH,'/' + pics.url)
            current_app.logger.info(real_mymain_color_path)

            mymain_color_path = main_color_demon.mymain_color(real_mymain_color_path)
            # /algorithm/cutimg/static/images_GLCM_original/images_camouflage/mix/20m/2.JPG
            pic_url = os.path.join(CUTIMG_SERVER_PATH, 'images_save/main_color/main_color2.JPG')
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': pic_url})

@fea_ns.route('/main_edge')
class rt_main_edge(Resource):
    def get(self):
        '''边缘图像'''
        current_app.logger.info(CUTIMG_PATH)
        try:
            path = os.path.join(CUTIMG_PATH,'images_GLCM')
            path_edge = os.path.join(CUTIMG_PATH, 'images_GLCM_edge')
            url1 = os.path.join(CUTIMG_SERVER_PATH, 'images_GLCM_edge/images_camouflage/mix/20m_canny/14.JPG')
            url2 = os.path.join(CUTIMG_SERVER_PATH, 'images_GLCM_edge/images_camouflage/mix/20m_canny/15.JPG')
            urls = []
            urls.append(url1)
            urls.append(url2)
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': urls})