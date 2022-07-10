# coding:utf-8
from flask import jsonify, request, Blueprint, render_template, redirect, make_response, current_app, url_for
from flask_restx import Api, Resource, fields, Namespace
from flask_restx.reqparse import RequestParser

import os
from os.path import isfile, join
from os import listdir
import uuid
from datetime import datetime

from algorithm.cutimg2.edge_histogram2 import myEdgeHistogramCanny
from common.mysql_operate import db_session, FEAPictureFile
from common.file_tools import unzip_file
from common.getUploadLocation import get_upload_location, get_server_location, get_alg_location
from common.remove_file_dir import remove_file_dir
# from common.find_star_end import find_se_com
from common.get_server_file_path import get_server_file_path
from common.get_server_ip_and_port import get_server_ip_and_port

from algorithm.cutimg import gray_histogram_differential, main_color_demon, edge_batch, GLCM_demo, coner_demon, blob_hist_correlation, cutimg
from algorithm.cutimg2 import color_gray_mean, color_gray_mean_excelSave, color_gray_histogram, color_gray_histogram_excelSave, color_main_color, color_main_color_excelSave, texture_GLCM, texture_GLCM_excelSave, texture_GGCM, texture_GGCM_excelSave, texture_GLDS, texture_GLDS_excelSave, texture_Tamura, texture_Tamura_excelSave, texture_LBP_excelSave, Blob_Kmeans, coner_coner, coner_coner_excelsSave, edge, edge_histogram
from werkzeug.datastructures import FileStorage
import traceback
import numpy as np

# from models.mul_model import MulModel

fea2 = Blueprint('fea2', __name__)
fea2_ns = Namespace('fea2', description='featureExtraction 特征提取')

# json body
CutImgDataParser: RequestParser = fea2_ns.parser()
CutImgDataParser.add_argument(
    'cutposs',
    type=str,
    required=True,
    location="json")
CutImgDataParser.add_argument(
    'cutimg_pid',
    type=str,
    required=True,
    location="json")

# 文件上传格式
file_parser: RequestParser = fea2_ns.parser()
file_parser.add_argument('file', location='files',
                         type=FileStorage, required=True)

# 上传图片路径
UPLOAD_PATH = get_upload_location(
    os.path.join(
        'cutimg2',
        'static',
        'img_original'))

# 物理绝对图片路径
CUTIMG_ABS_PATH = get_upload_location(os.path.join('cutimg2', 'static'))
# /Users/aldno/Downloads/flaskDemo-master/algorithm/cutimg/static

# 物理绝对algorithm映射路径
ALG_ABS_PATH = get_alg_location()
# /Users/aldno/Downloads/flaskDemo-master/algorithm/

# src目录
_ = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.abspath(os.path.join(_, '..'))

# 服务器图片路径
CUTIMG_SERVER_PATH = get_server_location(os.path.join('cutimg2', 'static'))
#  /algorithm/cutimg/static

# 实际访问地址 /api/v1/fea2/fileupload


@fea2_ns.route("/fileupload", doc={"description": "特征提取"})
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

    @fea2_ns.expect(file_parser, validate=True)
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

            proLoadPath = os.path.join(
                'algorithm',
                'cutimg2',
                'static',
                'img_original',
                save_filename)
            # realProLoadPath = os.path.join(UPLOAD_PATH, save_filename)
            # print(realProLoadPath)
            filename = proLoadPath

            session = db_session()
            new_file = FEAPictureFile(
                url=proLoadPath, create_time=datetime.now())
            session.add(new_file)
            session.commit()

            fea_pics = session.query(FEAPictureFile).filter(
                FEAPictureFile.url == proLoadPath).first()
            session.close()
            data = fea_pics.to_json()
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 201, 'message': '查找成功', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '上传图片成功', 'data': data})


@fea2_ns.route('/cutimg')
class rt_cutimg(Resource):
    @fea2_ns.expect(CutImgDataParser)
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
            pics = session.query(FEAPictureFile).filter(
                FEAPictureFile.pid == pid).first()

            real_mymain_color_path = os.path.join(src_path, pics.url)

            current_app.logger.info(real_mymain_color_path)
            real_path2 = os.path.join(CUTIMG_ABS_PATH, 'img_save_bitwise')
            real_path3 = os.path.join(CUTIMG_ABS_PATH, 'img_save_cutimg')
            path2, path3 = cutimg.mycutimg(
                real_mymain_color_path, real_path2, real_path3, list_cut)
            path2 = os.path.join(
                CUTIMG_SERVER_PATH,
                'img_save_bitwise',
                '1.JPG')
            # /algorithm/cutimg/static/images_GLCM_original/images_camouflage/mix/20m/2.JPG
            # pic_url = os.path.join(CUTIMG_SERVER_PATH, 'images_save/main_color/main_color2.JPG')

        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功',
                            'data': path2, 'data1': path3})

# ---------------------------------------------颜色特征--------------------------------------------------------


@fea2_ns.route('/color_gray_mean')
class rt_color_gray_mean(Resource):
    def get(self):
        '''颜色特征-灰度均值 可视化展示部分'''
        try:
            data = {}
            path = os.path.join(CUTIMG_ABS_PATH, 'img_save_cutimg')
            arr = color_gray_mean.myGrayMean(path)

            data['arr'] = arr.tolist()
            path_excel_save = os.path.join(
                CUTIMG_ABS_PATH, 'excels_save', 'color_gray_mean')
            excel_path = color_gray_mean_excelSave.myGrayMean_excelSave(
                path, path_excel_save)
            data['excel_path'] = get_server_ip_and_port(
                get_server_file_path(excel_path))
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})


@fea2_ns.route('/myGrayHitogram/<myGrayHitogram_id>')
@fea2_ns.param('myGrayHitogram_id', '图片id')
class rt_myGrayHitogram(Resource):
    def get(self, myGrayHitogram_id):
        '''颜色特征-灰度直方图 可视化展示部分'''
        try:
            pid = int(myGrayHitogram_id)
            session = db_session()
            pics = session.query(FEAPictureFile).filter(
                FEAPictureFile.pid == pid).first()

            data = {}
            path = os.path.join(src_path, pics.url)
            path_bit = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_bitwise', '1.jpg')
            arr = color_gray_histogram.myGrayHitogram(path, path_bit)

            data['arr'] = arr.tolist()
            path_excel_save = os.path.join(
                CUTIMG_ABS_PATH, 'excels_save', 'color_gray_histogram')
            excel_path = color_gray_histogram_excelSave.myGrayHitogram_excelSave(
                path, path_bit, path_excel_save)
            data['excel_path'] = get_server_ip_and_port(
                get_server_file_path(excel_path))
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})


@fea2_ns.route('/myMainColor')
class rt_myMainColor(Resource):
    def get(self):
        '''算法处理的是原图与path_bitwise对应的掩膜图像，返回一个长度为256的一维数组'''
        try:
            data = {}
            path_bit = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_bitwise', '1.jpg')
            path_mainColor = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_mainColor')
            main_color_url = color_main_color.myMainColor(
                path_bit, path_mainColor)

            data['main_color_url'] = get_server_file_path(main_color_url)
            path_excel_save = os.path.join(
                CUTIMG_ABS_PATH, 'excels_save', 'color_main_color')
            excel_path = color_main_color_excelSave.myMainColor_excelSave(
                path_bit, path_excel_save)
            data['excel_path'] = get_server_ip_and_port(
                get_server_file_path(excel_path))
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})


# ---------------------------------------------边缘特征--------------------------------------------------------

@fea2_ns.route('/myEdge')
class rt_myEdge(Resource):
    def get(self):
        '''coner_coner'''
        try:
            data = {}
            path_cutimg = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_cutimg')
            path_edge = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_edge')
            path_coner_ORB = os.path.join(
                path_edge, 'canny')
            path_coner_FAST = os.path.join(
                path_edge, 'laplacian')
            path_coner_SURF = os.path.join(
                path_edge, 'log')
            path_coner_SIFT = os.path.join(
                path_edge, 'prewitt')
            path_coner_BRISKF = os.path.join(
                path_edge, 'roberts')
            path_coner_KAZE = os.path.join(
                path_edge, 'sobel')
            urls = edge.myEdge(path_cutimg, path_edge, path_coner_ORB, path_coner_FAST, path_coner_SURF, path_coner_SIFT,
                               path_coner_BRISKF, path_coner_KAZE)
            server_urls = []
            for url in urls:
                server_urls.append(get_server_file_path(url))
            data['urls'] = server_urls
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})


@fea2_ns.route('/myEdgeHistogram')
class rt_myEdgeHistogram(Resource):
    def get(self):
        '''myEdger'''
        try:
            data = {}
            path_cutimg = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_cutimg')
            path_edge = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_edge')
            path_coner_ORB = os.path.join(
                path_edge, 'canny')
            path_coner_FAST = os.path.join(
                path_edge, 'laplacian')
            path_coner_SURF = os.path.join(
                path_edge, 'log')
            path_coner_SIFT = os.path.join(
                path_edge, 'prewitt')
            path_coner_BRISKF = os.path.join(
                path_edge, 'roberts')
            path_coner_KAZE = os.path.join(
                path_edge, 'sobel')
            path_edge_histogram = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_edge_histogram')
            path_edge_histogram_canny = os.path.join(
                path_edge_histogram, 'canny')
            path_edge_histogram_laplacian = os.path.join(
                path_edge_histogram, 'laplacian')
            path_edge_histogram_log = os.path.join(
                path_edge_histogram, 'log')
            path_edge_histogram_prewitt = os.path.join(
                path_edge_histogram, 'prewitt')
            path_edge_histogram_roberts = os.path.join(
                path_edge_histogram, 'roberts')
            path_edge_histogram_sobel = os.path.join(
                path_edge_histogram, 'sobel')
            excels_edge_histogram = os.path.join(
                CUTIMG_ABS_PATH, 'excels_save', 'edge_histogram')
            path_edge_histogram1, excel_path = edge_histogram.myEdgeHistogram(path_cutimg, path_edge, path_coner_ORB, path_coner_FAST, path_coner_SURF, path_coner_SIFT,
                                                                  path_coner_BRISKF, path_coner_KAZE, path_edge_histogram, path_edge_histogram_canny, path_edge_histogram_laplacian, path_edge_histogram_log, path_edge_histogram_prewitt, path_edge_histogram_roberts, path_edge_histogram_sobel, excels_edge_histogram)
            data['path_edge_histogram'] = get_server_ip_and_port(
                get_server_file_path(path_edge_histogram1))
            data['excel_path'] = get_server_ip_and_port(
                get_server_file_path(excel_path))
            print(data)
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})


@fea2_ns.route('/myEdgeHistogram2/<myEdgeHistogramName>')
@fea2_ns.param('myEdgeHistogramName', '边缘特征名称')
class rt_myEdgeHistogram2(Resource):
    def get(self, myEdgeHistogramName):
        '''myEdger'''
        try:
            data = {}
            path_cutimg = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_cutimg')
            path_edge = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_edge')
            path_coner_ORB = os.path.join(
                path_edge, 'canny')
            path_coner_FAST = os.path.join(
                path_edge, 'laplacian')
            path_coner_SURF = os.path.join(
                path_edge, 'log')
            path_coner_SIFT = os.path.join(
                path_edge, 'prewitt')
            path_coner_BRISKF = os.path.join(
                path_edge, 'roberts')
            path_coner_KAZE = os.path.join(
                path_edge, 'sobel')
            path_edge_histogram = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_edge_histogram')
            path_edge_histogram_canny = os.path.join(
                path_edge_histogram, 'canny')
            path_edge_histogram_laplacian = os.path.join(
                path_edge_histogram, 'laplacian')
            path_edge_histogram_log = os.path.join(
                path_edge_histogram, 'log')
            path_edge_histogram_prewitt = os.path.join(
                path_edge_histogram, 'prewitt')
            path_edge_histogram_roberts = os.path.join(
                path_edge_histogram, 'roberts')
            path_edge_histogram_sobel = os.path.join(
                path_edge_histogram, 'sobel')
            excels_edge_histogram = os.path.join(
                CUTIMG_ABS_PATH, 'excels_save', 'edge_histogram')

            result_path_edge_histogram = '';
            result_excels_edge_histogram = '';
            if(myEdgeHistogramName == 'VisibleLight_Canny_EDH'):
                result_path_edge_histogram , result_excels_edge_histogram = myEdgeHistogramCanny(path_cutimg, path_coner_ORB, path_edge_histogram_canny, excels_edge_histogram)
            elif (myEdgeHistogramName == 'VisibleLight_Sobel_EDH'):
                result_path_edge_histogram, result_excels_edge_histogram = myEdgeHistogramCanny(path_cutimg, path_coner_KAZE,
                                                                                                path_edge_histogram_sobel,
                                                                                                excels_edge_histogram)
            elif (myEdgeHistogramName == 'VisibleLight_Roberts_EDH'):
                result_path_edge_histogram, result_excels_edge_histogram = myEdgeHistogramCanny(path_cutimg, path_coner_BRISKF,
                                                                                                path_edge_histogram_roberts,
                                                                                                excels_edge_histogram)
            elif (myEdgeHistogramName == 'VisibleLight_Prewitt_EDH'):
                result_path_edge_histogram, result_excels_edge_histogram = myEdgeHistogramCanny(path_cutimg, path_coner_SIFT,
                                                                                                path_edge_histogram_prewitt,
                                                                                                excels_edge_histogram)
            elif (myEdgeHistogramName == 'VisibleLight_Laplacian_EDH'):
                result_path_edge_histogram, result_excels_edge_histogram = myEdgeHistogramCanny(path_cutimg, path_coner_FAST,
                                                                                                path_edge_histogram_laplacian,
                                                                                                excels_edge_histogram)
            elif (myEdgeHistogramName == 'VisibleLight_LoG_EDH'):
                result_path_edge_histogram, result_excels_edge_histogram = myEdgeHistogramCanny(path_cutimg, path_coner_SURF,
                                                                                                path_edge_histogram_log,
                                                                                                excels_edge_histogram)
            print(result_path_edge_histogram, result_excels_edge_histogram)
            data['path_edge_histogram'] = get_server_ip_and_port(
                get_server_file_path(result_path_edge_histogram))
            data['excel_path'] = get_server_ip_and_port(
                get_server_file_path(result_excels_edge_histogram))
            print(data)
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})


# ---------------------------------------------纹理特征--------------------------------------------------------

@fea2_ns.route('/myGLCM')
class rt_myGLCM(Resource):
    def get(self):
        '''myGLCM'''
        try:
            data = {}
            path = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_cutimg')
            arr = texture_GLCM.myGLCM(
                path)
            data['arr'] = arr.tolist()
            path_excel_save = os.path.join(
                CUTIMG_ABS_PATH, 'excels_save', 'color_main_color')
            excel_path = texture_GLCM_excelSave.myGLCM_excelSave(
                path, path_excel_save)
            data['excel_path'] = get_server_ip_and_port(
                get_server_file_path(excel_path))
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})


@fea2_ns.route('/myGGCM')
class rt_myGGCM(Resource):
    def get(self):
        '''myGGCM'''
        try:
            data = {}
            path = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_cutimg')
            arr = texture_GGCM.myGGCM(
                path)
            data['arr'] = arr.tolist()
            path_excel_save = os.path.join(
                CUTIMG_ABS_PATH, 'excels_save', 'texture_GGCM')
            excel_path = texture_GGCM_excelSave.myGGCM_excelSave(
                path, path_excel_save)
            data['excel_path'] = get_server_ip_and_port(
                get_server_file_path(excel_path))
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})


@fea2_ns.route('/myGLDS')
class rt_myGLDS(Resource):
    def get(self):
        '''myGLDS'''
        try:
            data = {}
            path = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_cutimg')
            arr = texture_GLDS.myGLDS(
                path)
            data['arr'] = arr
            path_excel_save = os.path.join(
                CUTIMG_ABS_PATH, 'excels_save', 'texture_GLDS')
            excel_path = texture_GLDS_excelSave.myGLDS_excelSave(
                path, path_excel_save)
            data['excel_path'] = get_server_ip_and_port(
                get_server_file_path(excel_path))
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})


@fea2_ns.route('/texture_Tamura')
class rt_texture_Tamura(Resource):
    def get(self):
        '''texture_Tamura'''
        try:
            data = {}
            path = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_cutimg')
            arr = texture_Tamura.myTamura(
                path)
            data['arr'] = arr.tolist()
            path_excel_save = os.path.join(
                CUTIMG_ABS_PATH, 'excels_save', 'texture_Tamura')
            excel_path = texture_Tamura_excelSave.myTamura_excelSave(
                path, path_excel_save)
            data['excel_path'] = get_server_ip_and_port(
                get_server_file_path(excel_path))
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})


@fea2_ns.route('/texture_Tamura')
class rt_texture_Tamura(Resource):
    def get(self):
        '''texture_Tamura'''
        try:
            data = {}
            path = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_cutimg')
            arr = texture_Tamura.myTamura(
                path)
            data['arr'] = arr.tolist()
            path_excel_save = os.path.join(
                CUTIMG_ABS_PATH, 'excels_save', 'texture_Tamura')
            excel_path = texture_Tamura_excelSave.myTamura_excelSave(
                path, path_excel_save)
            data['excel_path'] = get_server_ip_and_port(
                get_server_file_path(excel_path))
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})


@fea2_ns.route('/texture_LBP_excelSave')
class rt_texture_LBP_excelSave(Resource):
    def get(self):
        '''texture_LBP_excelSave'''
        try:
            data = {}
            path = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_cutimg')
            path_excel_save = os.path.join(
                CUTIMG_ABS_PATH, 'excels_save', 'texture_LBP')
            excel_path = texture_LBP_excelSave.myLBP_excelSave(
                path, path_excel_save)
            data['excel_path'] = get_server_ip_and_port(
                get_server_file_path(excel_path))
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})

# ---------------------------------------------斑块特征--------------------------------------------------------


@fea2_ns.route('/Blob_Kmeans')
class rt_Blob_Kmeans(Resource):
    def get(self):
        '''texture_LBP_excelSave'''
        try:
            data = {}
            path = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_cutimg')
            path_excel_save = os.path.join(
                CUTIMG_ABS_PATH, 'excels_save', 'blob_Kmeans')
            path_save_blob = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_blob')
            excel_path = Blob_Kmeans.myBlob_excelSave(
                path, path_excel_save, path_save_blob)
            data['url'] = path_save_blob = os.path.join(
                CUTIMG_SERVER_PATH, 'img_save_blob', 'blob14.jpg')
            data['excel_path'] = get_server_ip_and_port(
                get_server_file_path(excel_path))
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})


# ---------------------------------------------角点特征--------------------------------------------------------
@fea2_ns.route('/coner_coner')
class rt_coner_coner(Resource):
    def get(self):
        '''coner_coner'''
        try:
            data = {}
            path_cutimg = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_cutimg')
            path_coner = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_coner')
            path_coner_ORB = os.path.join(
                path_coner, 'ORB')
            path_coner_FAST = os.path.join(
                path_coner, 'FAST')
            path_coner_SURF = os.path.join(
                path_coner, 'SURF')
            path_coner_SIFT = os.path.join(
                path_coner, 'SIFT')
            path_coner_BRISKF = os.path.join(
                path_coner, 'BRISKF')
            path_coner_KAZE = os.path.join(
                path_coner, 'KAZE')
            urls = coner_coner.myConer(path_cutimg, path_coner, path_coner_ORB, path_coner_FAST, path_coner_SURF, path_coner_SIFT,
                                       path_coner_BRISKF, path_coner_KAZE)
            server_urls = []
            for url in urls:
                server_urls.append(get_server_file_path(url))
            data['urls'] = server_urls
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})


@fea2_ns.route('/myConer_excelSave')
class rt_myConer_excelSave(Resource):
    def get(self):
        '''texture_LBP_excelSave'''
        try:
            data = {}
            path = os.path.join(
                CUTIMG_ABS_PATH, 'img_save_cutimg')
            path_excel_save = os.path.join(
                CUTIMG_ABS_PATH, 'excels_save', 'texture_Tamura')
            data['excel_path'] = coner_coner_excelsSave.myConer_excelSave(
                path, path_excel_save)
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})
