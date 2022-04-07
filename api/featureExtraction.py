# coding:utf-8
from flask import jsonify, request, Blueprint, render_template, redirect, make_response, current_app, url_for
from flask_restx import Api, Resource, fields, Namespace
from flask_restx.reqparse import RequestParser

import os
from os.path import isfile, join
from os import listdir
import uuid
from datetime import datetime

from common.mysql_operate import db_session, FEAPictureFile
from common.file_tools import unzip_file
from common.getUploadLocation import get_upload_location, get_server_location
from common.remove_file_dir import remove_file_dir
from common.find_star_end import find_se_com

from algorithm.cutimg import gray_histogram_differential, main_color_demon, edge_batch, GLCM_demo, coner_demon, blob_hist_correlation, cutimg
from werkzeug.datastructures import FileStorage
import traceback
import numpy as np

# from models.mul_model import MulModel

fea = Blueprint('fea', __name__)
fea_ns = Namespace('fea', description='featureExtraction 特征提取')

# json body
CutImgDataParser: RequestParser = fea_ns.parser()
CutImgDataParser.add_argument('cutposs',type=str, required=True,location="json" )
CutImgDataParser.add_argument('cutimg_pid',type=str, required=True,location="json" )

# 文件上传格式
file_parser: RequestParser = fea_ns.parser()
file_parser.add_argument('file', location='files',
                    type=FileStorage, required=True)

# 上传图片路径
UPLOAD_PATH = get_upload_location(os.path.join('cutimg', 'static', 'images'))
# print(UPLOAD_PATH)
# 图片路径
CUTIMG_PATH = get_upload_location(os.path.join('cutimg', 'static'))
# /Users/aldno/Downloads/flaskDemo-master/algorithm/cutimg/static
# 服务器图片路径
CUTIMG_SERVER_PATH = get_server_location(os.path.join('cutimg', 'static'))
#  /algorithm/cutimg/static

# 实际访问地址 /api/v1/fea/fileupload
@fea_ns.route("/fileupload", doc={"description": "特征提取"})
class FileUploadHandler(Resource):
    def get(self):
        '''查看所有图片'''
        try:
            session = db_session()
            fea_pics = session.query(FEAPictureFile).all()
            data = []
            for fea_pic in fea_pics:
                data.append(fea_pic.to_json())
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 201, 'message': '查找成功', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})

    @fea_ns.expect(file_parser, validate=True)
    def post(self):
        '''上传图片'''
        try:
            # 普通参数获取
            # 获取pichead文件对象
            file = request.files.get('file')
            # 文件格式
            last_pix = '.' + file.filename.split('.')[-1]
            save_filename = str(uuid.uuid1()) + last_pix
            path = os.path.join(UPLOAD_PATH, save_filename)
            # 保存压缩包
            file.save(path)

            # 前端路径

            proLoadPath = os.path.join('algorithm/cutimg/static/images', save_filename)
            # realProLoadPath = os.path.join(UPLOAD_PATH, save_filename)
            # print(realProLoadPath)
            filename = proLoadPath

            session = db_session()
            new_file = FEAPictureFile(url=proLoadPath, create_time=datetime.now())
            session.add(new_file)
            session.commit()

            fea_pics = session.query(FEAPictureFile).filter(FEAPictureFile.url == proLoadPath).first()
            session.close()
            data = fea_pics.to_json()
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 201, 'message': '查找成功', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '上传图片成功', 'data': data})


@fea_ns.route('/cutimg')
class rt_cutimg(Resource):
    @fea_ns.expect(CutImgDataParser)
    def post(self):
        '''分割处理'''
        try:
            params = CutImgDataParser.parse_args()
            cutposs = params["cutposs"]
            cutimg_pid = params["cutimg_pid"]
            cutposs_data = eval(cutposs)
            list_cut = np.asfarray(cutposs_data)
            pid = int(cutimg_pid)
            session = db_session()
            pics = session.query(FEAPictureFile).filter(FEAPictureFile.pid == pid).first()

            real_mymain_color_path = find_se_com(CUTIMG_PATH, '/' + pics.url)
            current_app.logger.info(real_mymain_color_path)
            real_path2 = os.path.join(CUTIMG_PATH,'images_GLCM_bitwise','images_camouflage','mix','20m')
            real_path3 = os.path.join(CUTIMG_PATH, 'images_GLCM','images_camouflage','mix','20m')
            path2,path3 = cutimg.mycutimg(real_mymain_color_path,real_path2,real_path3, list_cut)
            path2 = os.path.join(CUTIMG_SERVER_PATH, 'images_GLCM_bitwise','images_camouflage','mix','20m','1.JPG')
            # /algorithm/cutimg/static/images_GLCM_original/images_camouflage/mix/20m/2.JPG
            # pic_url = os.path.join(CUTIMG_SERVER_PATH, 'images_save/main_color/main_color2.JPG')

        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': path2, 'data1': path3})

# 待修复
@fea_ns.route('/main_gray_hist_differential/<gray_hist_differential_id>')
@fea_ns.param('gray_hist_differential_id', '图片id')
class rt_main_gray_hist_differential(Resource):
    def get(self, gray_hist_differential_id):
        '''直方图图像'''
        try:
            # path = 'static/images_GLCM_original'
            # path_bitwise = 'static/images_GLCM_bitwise'
            # path_gray_histogram_save = 'static/images_save/gray_histogram/'
            pid = int(gray_hist_differential_id)
            session = db_session()
            pics = session.query(FEAPictureFile).filter(FEAPictureFile.pid == pid).first()

            real_mymain_color_path = find_se_com(CUTIMG_PATH, '/' + pics.url)
            path_bitwise = os.path.join(CUTIMG_PATH, 'images_GLCM_bitwise')
            path_gray_histogram_save = os.path.join(CUTIMG_PATH, 'images_save', 'gray_histogram')
            gray_histogram_differential.main_gray_hist_differential(path=real_mymain_color_path, path_bitwise=path_bitwise, path_gray_histogram_save=path_gray_histogram_save)
            pic_url = os.path.join(CUTIMG_SERVER_PATH, 'images_save','gray_histogram', '1.JPG')
            list = []
            list.append(pic_url)
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': list})

@fea_ns.route('/mymain_color/<mymain_color_id>')
@fea_ns.param('mymain_color_id', '图片id')
class rt_mymain_color(Resource):
    def get(self,mymain_color_id):
        '''提取主色'''
        try:
            # 61
            pid = int(mymain_color_id)
            session = db_session()
            pics = session.query(FEAPictureFile).filter(FEAPictureFile.pid ==  pid ).first()

            real_mymain_color_path = find_se_com(CUTIMG_PATH,'/' + pics.url)
            current_app.logger.info(real_mymain_color_path)

            mymain_color_path = main_color_demon.mymain_color(real_mymain_color_path)
            # /algorithm/cutimg/static/images_GLCM_original/images_camouflage/mix/20m/2.JPG
            pic_url = os.path.join(CUTIMG_SERVER_PATH, 'images_save', 'main_color', 'main_color2.JPG')
            list = []
            list.append(pic_url)
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': list})

@fea_ns.route('/main_edge')
class rt_main_edge(Resource):
    def get(self):
        '''边缘图像'''
        current_app.logger.info(CUTIMG_PATH)
        try:
            path = os.path.join(CUTIMG_PATH,'images_GLCM')
            path_edge = os.path.join(CUTIMG_PATH, 'images_GLCM_edge')
            edge_batch.main_edge(path=path, path_edge=path_edge)
            url1 = os.path.join(CUTIMG_SERVER_PATH, 'images_GLCM_edge','images_camouflage', 'mix', '20m_canny', '14.JPG')
            url2 = os.path.join(CUTIMG_SERVER_PATH, 'images_GLCM_edge','images_camouflage', 'mix', '20m_canny', '15.JPG')
            urls = []
            urls.append(url1)
            urls.append(url2)
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': urls})

@fea_ns.route('/myGLCM_demo/<myGLCM_demo_id>')
@fea_ns.param('myGLCM_demo_id', '图片id')
class rt_myGLCM_demo(Resource):
    def get(self,myGLCM_demo_id):
        '''GLCM可视化结果'''
        try:
            # 62
            pid = int(myGLCM_demo_id)
            session = db_session()
            pics = session.query(FEAPictureFile).filter(FEAPictureFile.pid ==  pid ).first()

            real_myGLCM_demo_path = find_se_com(CUTIMG_PATH,'/' + pics.url)
            current_app.logger.info(real_myGLCM_demo_path)

            mymain_color_path = GLCM_demo.myGLCM_demo(real_myGLCM_demo_path)
            current_app.logger.info(mymain_color_path)
            # /algorithm/cutimg/static/images_GLCM_original/images_camouflage/mix/20m/2.JPG
            pic_url = os.path.join(CUTIMG_SERVER_PATH, 'images_save','GLCM_demo','GLCM_Features.png')
            list = []
            list.append(pic_url)
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': list})


@fea_ns.route('/myconer')
class rt_myconer(Resource):
    def get(self):
        '''角点匹配情况图像'''
        try:
            # path = 'static\\images_GLCM'
            # path_save_coner = 'static/images_save/coner/
            path = os.path.join(CUTIMG_PATH,'images_GLCM')
            path_save_coner = os.path.join(CUTIMG_PATH, 'images_save','coner')
            coner_demon.myconer(path=path,  path_save_coner=path_save_coner)
            pic_url = os.path.join(CUTIMG_SERVER_PATH, 'images_save','coner','coner.JPG')
            list = []
            list.append(pic_url)
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': list})

@fea_ns.route('/myblobhist')
class rt_myblobhist(Resource):
    def get(self):
        '''斑块图像以及对应直方图'''
        try:
            # path1 = 'static/images_GLCM/images_camouflage/mix/20m/'
            # path_blob_hist_save = 'static/images_save/blob_hist/'
            path1 = os.path.join(CUTIMG_PATH,'images_GLCM','images_camouflage','mix','20m')
            path_blob_hist_save = os.path.join(CUTIMG_PATH, 'images_save','blob_hist')
            server_path_blob_hist_save = os.path.join(CUTIMG_SERVER_PATH, 'images_save','blob_hist')
            blob_hist_correlation.myblobhist(path1=path1,  path_blob_hist_save=path_blob_hist_save)
            urls = []
            for filename in os.listdir(path_blob_hist_save):
                if not filename.startswith('.') and filename.endswith('JPG'):
                    url = os.path.join(server_path_blob_hist_save, filename)
                    urls.append(url)

        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': urls})