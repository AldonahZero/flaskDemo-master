import base64
import os
import uuid
from datetime import datetime
from io import BytesIO

import pandas as pd
from flask import request, flash, jsonify, send_from_directory
from flask_restx import Resource, Namespace
from flask_restx.reqparse import RequestParser
from werkzeug.datastructures import FileStorage

import matplotlib.pyplot as plt


from algorithm.HSI.showPseudoColor import show_image
from algorithm.HSI.FeatureExtraction.edge_feature import canny_edge_f
from algorithm.HSI.FeatureExtraction import HSI_NDWI_f, HSI_NDVI_f, Harris_points_f
from algorithm.HSI.HSI_grabcut import Hsi_grabcut_f
from algorithm.HSI.FeatureExtraction.gray_feature import gray_mean_dif_f, gray_var_dif_f, gray_histogram_dif_f
from algorithm.HSI.band_Selection import ECA_f
from common.mysql_operate import db_session, HSIPictureFile, HSIResultFile

hsi_ns = Namespace('hsi', description='高光谱算法')

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'raw'}
HSI_UPLOAD_FOLDER = 'algorithm/HSI/static/upload/'
HSI_RESULT_FOLDER = 'algorithm/HSI/static/result/'
CUT_RESULT_PATH = 'algorithm/HSI/static/cut_result/'
EXCEL_SAVE_PATH = 'algorithm/HSI/static/excel_result/'

# 文件上传格式
parser: RequestParser = hsi_ns.parser()
parser.add_argument('file', location='files', type=FileStorage, required=True)


@hsi_ns.route('/upload', doc={"description": "上传伪彩图片,成功返回调用文件的key,通过此key可以调用后面的功能"})
class Upload(Resource):
    @hsi_ns.expect(parser, validate=True)
    def post(self):
        '''上传图片'''
        if 'file' not in request.files:
            flash('No file part')
            return {'message': 'No file part'}, 201
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return {'message': 'No selected file'}, 201
        if file and allowed_file(file.filename):
            fid = str(uuid.uuid1())
            # 原始文件
            filename = fid + '.' + file.filename.rsplit('.', 1)[1]
            save_path = HSI_UPLOAD_FOLDER + filename
            file.save(save_path)
            # 伪彩图片
            rel_out_path = HSI_RESULT_FOLDER + fid + '.jpg'
            abs_out_path = os.path.abspath(rel_out_path)
            show_image(save_path, abs_out_path)
            # 存放到数据库
            session = db_session()
            picture_file = HSIPictureFile(pid=fid, file_path=save_path, picture_path=rel_out_path, create_time=datetime.now())
            session.add(picture_file)
            session.commit()
            session.close()
            return jsonify({'message': 'success', 'key': fid})
        return jsonify({'code': 400, 'message': "file not allow"})


def allowed_file(filename):
    """判断文件是否允许上传"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_obj(key):
    """按key查找文件"""
    session = db_session()
    arr = session.query(HSIPictureFile).filter(HSIPictureFile.pid == key).all()
    session.close()
    return arr[0]

def get_result_obj(key):
    """按key查找文件"""
    session = db_session()
    arr = session.query(HSIResultFile).filter(HSIResultFile.fid == key).all()
    session.close()
    return arr[0]

#
# @hsi_ns.route('/showPseudoColor/<string:file_path>', doc={"description": "显示伪彩图片：接受文件的服务器存放名称，成功返回图片存放路径"})
# @hsi_ns.param('key', '文件上传后返回的文件名称')
# class showPseudoColor(Resource):
#     @hsi_ns.doc('显示伪彩图片')
#     def get(self, key):
#         '''显示伪彩图片'''
#         save_path = get_file_obj(key).file_path
#         file_name = str(uuid.uuid1())
#         rel_out_path = HSI_RESULT_FOLDER + file_name + '.jpg'
#         abs_out_path = os.path.abspath(rel_out_path)
#         try:
#             show_image(save_path, abs_out_path)
#             session = db_session()
#             pic = HSIPictureFile(pid=file_name, path=rel_out_path, create_time=datetime.now())
#             session.add(pic)
#             session.commit()
#             session.close()
#         except BaseException as e:
#             return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
#         else:
#             return jsonify({'code': 201, 'message': 'success', 'url': rel_out_path})


@hsi_ns.route('/HSI_grabcut/<key>',  doc={"description": "图像分割返回結果：掩膜图像：target.jpg,标定背景的掩膜图像： back.jpg,"
                                                         "九宫格切割结果的目标可视化显示: cut_result.jpg"})
@hsi_ns.param('key', '上传时返回的key')
class HSI_grabcut(Resource):
    def get(self, key):
        '''图像分割 选择特征前需要先运行此方法'''
        save_path = get_file_obj(key).file_path
        try:
            out_path = Hsi_grabcut_f(save_path)
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'target_path': out_path + 'target.jpg',
                            'back_path': out_path + 'back.jpg', 'cut_result': out_path + 'cut_result.jpg'})


@hsi_ns.route('/gray_mean/<key>', doc={"description": "返回结果 result：灰度均值数组  src: 折线图base64编码"})
@hsi_ns.param('key', '上传时返回的key')
class gray_mean(Resource):
    def get(self, key):
        '''灰度均值'''
        save_path = get_file_obj(key).file_path
        data_path = CUT_RESULT_PATH + key + "/"
        try:
            result = gray_mean_dif_f(save_path, data_path, EXCEL_SAVE_PATH)
            plt.plot(result)
            sio = BytesIO()
            plt.savefig(sio, format='png')
            data = base64.encodebytes(sio.getvalue()).decode()
            src = 'data:image/png;base64,' + str(data).replace('\n', '')
            plt.close()
        except BaseException as e:
            print(e)
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': result.tolist(), 'src': src})


@hsi_ns.route('/gray_diff/<key>', doc={"description": "返回结果 result：灰度方差数组  src: 折线图base64编码"})
@hsi_ns.param('key', '上传时返回的key')
class gray_diff(Resource):
    def get(self, key):
        '''灰度方差'''
        save_path = get_file_obj(key).file_path
        data_path = CUT_RESULT_PATH + key + "/"
        try:
            result = gray_var_dif_f(save_path, data_path, EXCEL_SAVE_PATH)
            plt.plot(result)
            sio = BytesIO()
            plt.savefig(sio, format='png')
            data = base64.encodebytes(sio.getvalue()).decode()
            src = 'data:image/png;base64,' + str(data).replace('\n', '')
            plt.close()
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': result.tolist(), 'src': src})


@hsi_ns.route('/gray_histogram_diff/<key>/<int:band_index>')
@hsi_ns.param('key', '上传时返回的key')
@hsi_ns.param('band_index', '波段索引(1到176的数字)')
class gray_histogram_dif(Resource):
    def get(self, key,band_index):
        '''目标背景各波段灰度直方图协方差系数'''
        save_path = get_file_obj(key).file_path
        data_path = CUT_RESULT_PATH + key + "/"
        try:
            result = gray_histogram_dif_f(save_path,band_index, data_path, EXCEL_SAVE_PATH)
            plt.plot(result)
            sio = BytesIO()
            plt.savefig(sio, format='png')
            data = base64.encodebytes(sio.getvalue()).decode()
            src = 'data:image/png;base64,' + str(data).replace('\n', '')
            plt.close()
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': result.tolist(), 'src': src})


@hsi_ns.route('/HSI_NDWI/<key>')
@hsi_ns.param('key', '上传时返回的key')
class HSI_NDWI(Resource):
    def get(self, key):
        '''归一化水体指数'''
        save_path = get_file_obj(key).file_path
        out_path = HSI_RESULT_FOLDER + key + '_NDWI_result.jpg'
        try:
            result = HSI_NDWI_f(save_path, out_path)
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': result})


@hsi_ns.route('/HSI_NDVI/<key>')
@hsi_ns.param('key', '上传时返回的key')
class HSI_NDVI(Resource):
    def get(self, key):
        '''归一化植被指数'''
        save_path = get_file_obj(key).file_path
        out_path = HSI_RESULT_FOLDER + key + '_NDVI_result.jpg'
        try:
            result = HSI_NDVI_f(save_path, out_path)
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': result})


@hsi_ns.route('/HSI_SAM/<key>')
@hsi_ns.param('key', '上传时返回的key')
class HSI_SAM(Resource):
    def get(self, key):
        '''光谱余弦距离'''
        save_path = get_file_obj(key).file_path
        out_path = HSI_RESULT_FOLDER + key + '_SAM_result.jpg'
        try:
            result = HSI_NDVI_f(save_path, out_path)
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': result})


@hsi_ns.route('/canny_edge/<key>/<int:index>')
@hsi_ns.param('key', '上传时返回的key')
@hsi_ns.param('index', '数字所选波段索引')
class canny_edge(Resource):
    def get(self, key, index):
        '''边缘检测'''
        cut_path = CUT_RESULT_PATH + key + '/'
        out_path = HSI_RESULT_FOLDER + key + '_edge_canny_result.jpg'
        try:
            result = canny_edge_f(cut_path, index, out_path)
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': result})


@hsi_ns.route('/ECA/<key>/<int:k_num>')
@hsi_ns.param('key', '上传时返回的key')
@hsi_ns.param('k_num', '一个整数数字')
class ECA(Resource):
    def get(self, key, k_num):
        '''波段选择算法'''
        save_path = get_file_obj(key).file_path
        try:
            result = ECA_f(save_path, k_num)
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': result.tolist()})


@hsi_ns.route('/Harris_points/<key>')
@hsi_ns.param('key', '上传时返回的key')
class Harris_points(Resource):
    def get(self, key):
        '''角点检测'''
        save_path = get_file_obj(key).file_path
        out_path = HSI_RESULT_FOLDER + key + '_points_Harris.jpg'
        try:
            result = Harris_points_f(save_path, out_path)
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': result})

#
# @hsi_ns.route('/download/<result_key>')
# @hsi_ns.param('result_key', '36位的excel文件key')
# class download_result(Resource):
#     def get(self, result_key):
#         '''结果数据下载'''
#         save_path = get_result_obj(result_key).path
#         return send_from_directory(save_path, filename="123", as_attachment=True)
#         #return jsonify({'code': 201, 'message': 'success'})