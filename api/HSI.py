import os
import uuid

from flask import request, flash, jsonify
from flask_restx import Resource, Namespace
from flask_restx.reqparse import RequestParser
from werkzeug.datastructures import FileStorage

from algorithm.HSI.showPseudoColor import show_image
from algorithm.HSI.FeatureExtraction.edge_feature import canny_edge_f
from algorithm.HSI.FeatureExtraction import HSI_NDWI_f, HSI_NDVI_f, Harris_points_f
from algorithm.HSI.HSI_grabcut import Hsi_grabcut_f
from algorithm.HSI.FeatureExtraction.gray_feature import gray_mean_dif_f, gray_var_dif_f, gray_histogram_dif_f
from algorithm.HSI.band_Selection import ECA_f

hsi_ns = Namespace('hsi', description='高光谱部分算法')

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'raw'}
HSI_UPLOAD_FOLDER = 'algorithm/HSI/static/upload/'
HSI_RESULT_FOLDER = 'algorithm/HSI/static/result/'

# 文件上传格式
parser: RequestParser = hsi_ns.parser()
parser.add_argument('file', location='files', type=FileStorage, required=True)


@hsi_ns.route('/upload', doc={"description": "上传伪彩图片,成功返回图片存放路径"})
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
            filename = str(uuid.uuid1()) + '.' + file.filename.rsplit('.', 1)[1]
            save_path = HSI_UPLOAD_FOLDER + filename
            file.save(save_path)
            return jsonify({'message': 'success', 'url': save_path})
        return jsonify({'code': 400, 'message': "file not allow"})


def allowed_file(filename):
    """判断文件是否允许上传"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@hsi_ns.route('/showPseudoColor/<string:file_path>', doc={"description": "显示伪彩图片：接受文件的服务器存放名称，成功返回图片存放路径"})
@hsi_ns.param('file_path', '文件上传后返回的文件名称')
class showPseudoColor(Resource):
    @hsi_ns.doc('示伪彩图片')
    def get(self, file_path):
        '''显示伪彩图片'''
        save_path = HSI_UPLOAD_FOLDER + file_path
        rel_out_path = HSI_RESULT_FOLDER + str(uuid.uuid1()) + '.jpg'
        abs_out_path = os.path.abspath(rel_out_path)
        try:
            show_image(save_path, abs_out_path)
        except BaseException as e:
            return jsonify({'code': 400, 'message': '查找failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'url': rel_out_path})


@hsi_ns.route('/HSI_grabcut/<file_path>',  doc={"description": "图像分割"})
@hsi_ns.param('file_path', '上传时返回的文件名称')
class HSI_grabcut(Resource):
    def get(self, file_path):
        save_path = HSI_UPLOAD_FOLDER + file_path
        try:
            out_path = Hsi_grabcut_f(save_path)
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'target_path': out_path + 'target.jpg',
                            'back_path': out_path + 'back.jpg'})


@hsi_ns.route('/gray_mean/<file_path>')
@hsi_ns.param('file_path', '上传时返回的文件名称')
class gray_mean(Resource):
    def get(self, file_path):
        '''灰度均值'''
        save_path = HSI_UPLOAD_FOLDER + file_path
        try:
            result = gray_mean_dif_f(save_path)
            # print(result)
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': result.tolist()})


@hsi_ns.route('/gray_diff/<file_path>')
@hsi_ns.param('file_path', '上传时返回的文件名称')
class gray_diff(Resource):
    def get(self, file_path):
        '''灰度方差'''
        save_path = HSI_UPLOAD_FOLDER + file_path
        try:
            result = gray_var_dif_f(save_path)
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': result.tolist()})


@hsi_ns.route('/gray_histogram_diff/<file_path>')
@hsi_ns.param('file_path', '上传时返回的文件名称')
class gray_histogram_dif(Resource):
    def get(self, file_path):
        '''目标背景各波段灰度直方图协方差系数'''
        save_path = HSI_UPLOAD_FOLDER + file_path
        try:
            result = gray_histogram_dif_f(save_path)
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': result.tolist()})


@hsi_ns.route('/HSI_NDWI/<file_path>')
@hsi_ns.param('file_path', '上传时返回的文件名称')
class HSI_NDWI(Resource):
    def get(self, file_path):
        '''归一化水体指数'''
        save_path = HSI_UPLOAD_FOLDER + file_path
        out_path = HSI_RESULT_FOLDER + file_path[0:file_path.rindex('.')] + '_NDWI_result.jpg'
        try:
            result = HSI_NDWI_f(save_path, out_path)
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': result})


@hsi_ns.route('/HSI_NDVI/<file_path>')
@hsi_ns.param('file_path', '上传时返回的文件名称')
class HSI_NDVI(Resource):
    def get(self, file_path):
        '''归一化植被指数'''
        save_path = HSI_UPLOAD_FOLDER + file_path
        out_path = HSI_RESULT_FOLDER + file_path[0:file_path.rindex('.')] + '_NDVI_result.jpg'
        try:
            result = HSI_NDVI_f(save_path, out_path)
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': result})


@hsi_ns.route('/HSI_SAM/<file_path>')
@hsi_ns.param('file_path', '上传时返回的文件名称')
class HSI_SAM(Resource):
    def get(self, file_path):
        '''光谱余弦距离'''
        save_path = HSI_UPLOAD_FOLDER + file_path
        out_path = HSI_RESULT_FOLDER + file_path[0:file_path.rindex('.')] + '_SAM_result.jpg'
        try:
            result = HSI_NDVI_f(save_path, out_path)
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': result})


@hsi_ns.route('/canny_edge/<file_path>/<int:index>')
@hsi_ns.param('file_path', '上传时返回的文件名称')
@hsi_ns.param('index', '数字所选波段索引')
class canny_edge(Resource):
    def get(self, file_path, index):
        '''边缘检测'''
        save_path = HSI_UPLOAD_FOLDER + file_path
        out_path = HSI_RESULT_FOLDER + file_path[0:file_path.rindex('.')] + '_edge_canny_result.jpg'
        try:
            result = canny_edge_f(save_path, index, out_path)
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': result})


@hsi_ns.route('/ECA/<file_path>/<int:k_num>')
@hsi_ns.param('file_path', '上传时返回的文件名称')
@hsi_ns.param('k_num', '一个整数数字')
class ECA(Resource):
    def get(self, file_path, k_num):
        '''波段选择算法'''
        save_path = HSI_UPLOAD_FOLDER + file_path
        try:
            result = ECA_f(save_path, k_num)
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': result.tolist()})


@hsi_ns.route('/Harris_points/<file_path>')
@hsi_ns.param('file_path', '上传时返回的文件名称')
class Harris_points(Resource):
    def get(self, file_path):
        '''角点检测'''
        save_path = HSI_UPLOAD_FOLDER + file_path
        out_path = HSI_RESULT_FOLDER + file_path[0:file_path.rindex('.')] + '_points_Harris.jpg'
        try:
            result = Harris_points_f(save_path, out_path)
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': result})
