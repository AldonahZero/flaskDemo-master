# coding:utf-8
import os
import uuid
from algorithm.SAR import fengzhi, fractal, GGCM, GLCM, HOG_feature, LBP, Moment_Seven, RCS, zernike_moment, \
    process_pre, image_stitching, get_input, geometric_feature
from flask import flash, jsonify, request, Blueprint, current_app
from flask_restx import Api, Resource, fields, Namespace
from flask_restx.reqparse import RequestParser

# print(os.getcwd())

sar = Blueprint('sar', __name__)
sar_ns = Namespace('SAR', description='SAR图像处理')
UPLOAD_FOLDER = os.path.join('algorithm', 'SAR', 'uploads')
RESULT_FOLDER = os.path.join('algorithm', 'SAR', 'result')
# 上传图片路径
IMG_UPLOAD = os.path.join(UPLOAD_FOLDER, 'SAR')
# static/uploads/SAR

# 处理结果图片路径
IMG_RESULT = os.path.join(RESULT_FOLDER, 'SAR')
# static/result/SAR

# print(IMG_UPLOAD)
print(IMG_RESULT)
ALLOWED_EXTENSIONS = {'jpg', 'tiff', 'tif', 'png', 'PNG'}


@sar_ns.route('/upload')
class Upload(Resource):
    @sar_ns.param('file', '文件')
    def post(self):
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return {'message': 'No file part'}, 201
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return {'message': 'No selected file'}, 201
        if file and allowed_file(file.filename):
            filename = str(uuid.uuid1()) + '.' + \
                file.filename.rsplit('.', 1)[1]
            filename = 'image_input.png'
            save_path = os.path.join(IMG_UPLOAD, filename)
            file.save(save_path)
            return {'message': 'success', 'url': save_path}
        return {'message': "file not allow"}, 201


def allowed_file(filename):
    """判断文件是否允许上传"""
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in ALLOWED_EXTENSIONS


@sar_ns.route('/filter/<file_path>')
class image_filter(Resource):
    def get(self, file_path):
        '''图像滤波'''
        if file_path == 'none':
            # image_path = RESULT_FOLDER + '/SAR/image_input.png'
            image_path = os.path.join(RESULT_FOLDER, 'SAR', 'image_input.png')
        else:
            # image_path = IMG_UPLOAD + file_path
            image_path = os.path.join(IMG_UPLOAD, file_path)
        try:
            result, _ = process_pre.nsst_dec(image_path)
        except BaseException as e:
            return {'status': 'failed', 'message': str(e)}, 201
        else:
            return {'status': 'success', 'url': result}, 201


@sar_ns.route('/binary/<file_path>')
class image_binary(Resource):
    def get(self, file_path):
        '''图像二值化（分割）'''
        if file_path == 'none':
            # image_path = RESULT_FOLDER + '/SAR/image_input.png'
            image_path = os.path.join(RESULT_FOLDER, 'SAR', 'image_input.png')
        else:
            # image_path = IMG_UPLOAD + file_path
            image_path = os.path.join(IMG_UPLOAD, file_path)
        try:
            result, _ = process_pre.otsu_2d(image_path)
        except BaseException as e:
            return {'status': 'failed', 'message': str(e)}, 201
        else:
            return {'status': 'success', 'url': result}, 201


@sar_ns.route('/value_peak/<file_path>')
class value_peak(Resource):
    def get(self, file_path):
        '''图像峰值点'''
        if file_path == 'none':
            # image_path = RESULT_FOLDER + '/SAR/image_input.png'
            image_path = os.path.join(RESULT_FOLDER, 'SAR', 'image_input.png')
        else:
            # image_path = IMG_UPLOAD + file_path
            image_path = os.path.join(IMG_UPLOAD, file_path)
        try:
            result = fengzhi.peak_value(image_path)
        except BaseException as e:
            return {'status': 'failed', 'message': str(e)}, 201
        else:
            return {'status': 'success', 'url': result}, 201


@sar_ns.route('/fractal/<file_path>')
class Fractal(Resource):
    def get(self, file_path):
        '''分形维数特征'''
        if file_path == 'none':
            # image_path = RESULT_FOLDER + '/SAR/image_input.png'
            image_path = os.path.join(RESULT_FOLDER, 'SAR', 'image_input.png')
        else:
            # image_path = IMG_UPLOAD + file_path
            image_path = os.path.join(IMG_UPLOAD, file_path)
        try:
            result_n, result_s = fractal.Simple_DBC(image_path)
        except BaseException as e:
            return {'status': 'failed', 'message': str(e)}, 201
        else:
            return {'status': 'success',
                    'url': result_n, 'data': result_s}, 201


@sar_ns.route('/GGCM/<file_path>')
class ggcm(Resource):
    def get(self, file_path):
        '''灰度梯度共生矩阵特征'''
        if file_path == 'none':
            # image_path = RESULT_FOLDER + '/SAR/image_input.png'
            image_path = os.path.join(RESULT_FOLDER, 'SAR', 'image_input.png')
        else:
            # image_path = IMG_UPLOAD + file_path
            image_path = os.path.join(IMG_UPLOAD, file_path)
        try:
            result, result_s = GGCM.get_ggcm_features(image_path)
            res = result_s
        except BaseException as e:
            return {'status': 'failed', 'message': str(e)}, 201
        else:
            return {'status': 'success', 'url': result, 'data': res}, 201


@sar_ns.route('/GLCM/<file_path>')
class glcm(Resource):
    def get(self, file_path):
        '''灰度共生矩阵特征'''
        if file_path == 'none':
            # image_path = RESULT_FOLDER + '/SAR/image_input.png'
            image_path = os.path.join(RESULT_FOLDER, 'SAR', 'image_input.png')
        else:
            # image_path = IMG_UPLOAD + file_path
            image_path = os.path.join(IMG_UPLOAD, file_path)
        try:
            result, result_s = GLCM.get_glcm_features(image_path)
            res = result_s.tolist()
        except BaseException as e:
            return {'status': 'failed', 'message': str(e)}, 201
        else:
            return {'status': 'success', 'url': result, 'data': res}, 201


@sar_ns.route('/HOG/<file_path>')
class hog(Resource):
    def get(self, file_path):
        '''方向梯度直方图特征'''
        if file_path == 'none':
            # image_path = RESULT_FOLDER + '/SAR/image_input.png'
            image_path = os.path.join(RESULT_FOLDER, 'SAR', 'image_input.png')
        else:
            # image_path = IMG_UPLOAD + file_path
            image_path = os.path.join(IMG_UPLOAD, file_path)
        try:
            result, result_s = HOG_feature.HOG_feature(image_path)
        except BaseException as e:
            return {'status': 'failed', 'message': str(e)}, 201
        else:
            return {'status': 'success', 'url': result, 'data': result_s}, 201


@sar_ns.route('/LBP/<file_path>')
class lbp(Resource):
    def get(self, file_path):
        '''局部二值模式特征'''
        if file_path == 'none':
            # image_path = RESULT_FOLDER + '/SAR/image_input.png'
            image_path = os.path.join(RESULT_FOLDER, 'SAR', 'image_input.png')
        else:
            # image_path,res = IMG_UPLOAD + file_path
            image_path = os.path.join(IMG_UPLOAD, file_path)
        try:
            result, res = LBP.getCircularLBPFeature(image_path)
        except BaseException as e:
            return {'status': 'failed', 'message': str(e)}, 201
        else:
            return {'status': 'success', 'url': result, 'data': res}, 201


@sar_ns.route('/Hu/<file_path>')
class hu(Resource):
    def get(self, file_path):
        '''Hu矩特征'''
        if file_path == 'none':
            # image_path = RESULT_FOLDER + '/SAR/image_input.png'
            image_path = os.path.join(RESULT_FOLDER, 'SAR', 'image_input.png')
        else:
            # image_path = IMG_UPLOAD + file_path
            image_path = os.path.join(IMG_UPLOAD, file_path)
        try:
            result, result_s = Moment_Seven.MomentSeven(image_path)
            res = result_s.tolist()
        except BaseException as e:
            return {'status': 'failed', 'message': str(e)}, 201
        else:
            return {'status': 'success', 'url': result, 'data': res}, 201


@sar_ns.route('/Genmetric features/<file_path>')
class geometric(Resource):
    def get(self, file_path):
        '''相关几何特征'''
        if file_path == 'none':
            # image_path = RESULT_FOLDER + '/SAR/image_input.png'
            image_path = os.path.join(RESULT_FOLDER, 'SAR', 'image_input.png')
        else:
            # image_path = IMG_UPLOAD + file_path
            image_path = os.path.join(IMG_UPLOAD, file_path)
        try:
            result, result_s = geometric_feature.get_geometric_feature(
                image_path)
            res = result_s.tolist()
        except BaseException as e:
            return {'status': 'failed', 'message': str(e)}, 201
        else:
            return {'status': 'success', 'url': result, 'data': res}, 201


@sar_ns.route('/Zernike/<file_path>')
class zernik(Resource):
    def get(self, file_path):
        '''Zernike矩特征'''
        if file_path == 'none':
            # image_path = RESULT_FOLDER + '/SAR/image_input.png'
            image_path = os.path.join(RESULT_FOLDER, 'SAR', 'image_input.png')
        else:
            # image_path = IMG_UPLOAD + file_path
            image_path = os.path.join(IMG_UPLOAD, file_path)
        try:
            result, result_s = zernike_moment.get_zernike(image_path)
        except BaseException as e:
            return {'status': 'failed', 'message': str(e)}, 201
        else:
            return {'status': 'success', 'url': result, 'data': result_s}, 201


@sar_ns.route('/RCS/<file_path>')
class rcs(Resource):
    def get(self, file_path):
        '''雷达散射截面'''
        # image_path = IMG_UPLOAD + file_path
        if file_path == 'none':
            # image_path = RESULT_FOLDER + '/SAR/image_input.png'
            image_path = os.path.join(RESULT_FOLDER, 'SAR', 'image_input.png')
        else:
            # image_path = IMG_UPLOAD + file_path
            image_path = os.path.join(IMG_UPLOAD, file_path)
        try:
            result, result_s = RCS.RCS(image_path)
        except BaseException as e:
            return {'status': 'failed', 'message': str(e)}, 201
        else:
            return {'status': 'success', 'url': result, 'data': result_s}, 201


# ImgDataParser: RequestParser = sar_ns.parser()
# ImgDataParser.add_argument('points',type=str, required=True,location="json" )
# ImgDataParser.add_argument('image_path',type=str, required=True,location="json" )
#
# @sar_ns.route('/image_seleect')
# class select(Resource):
#     @sar_ns.expect(ImgDataParser)
#     def post(self):
#         '''截取特征提取图像区域(特征提取前操作)'''
#         try:
#             params = ImgDataParser.parse_args()
#             image_points = params["points"]
#             img_name = params["image_path"]
#
#             path = IMG_UPLOAD + img_name
#             points = eval(image_points)
#             result = get_input.get_image(points, path)
#         except BaseException as e:
#             return {'status': 'failed', 'message': str(e)}, 201
#         else:
#             return {'status': 'success', 'url': result}, 201


# json body
StitchingImgDataParser: RequestParser = sar_ns.parser()
StitchingImgDataParser.add_argument(
    'image1_path',
    type=str,
    required=True,
    location="json")
StitchingImgDataParser.add_argument(
    'image2_path',
    type=str,
    required=True,
    location="json")
StitchingImgDataParser.add_argument(
    'RIGHT_LEFT',
    type=str,
    required=True,
    location="json")


@sar_ns.route('/image_stitching')
class Image_Stitching(Resource):
    @sar_ns.expect(StitchingImgDataParser)
    def post(self):
        '''图像拼接'''
        try:
            params = StitchingImgDataParser.parse_args()
            img1_name = params["image1_path"]
            img2_name = params["image2_path"]
            RIGHT_LEFT = params["RIGHT_LEFT"]

            # path1 = IMG_UPLOAD + img1_name
            # path2 = IMG_UPLOAD + img2_name
            path1 = os.path.join(IMG_UPLOAD, img1_name)
            path2 = os.path.join(IMG_UPLOAD, img2_name)
            result = image_stitching.image_stitching(path1, path2, RIGHT_LEFT)
        except BaseException as e:
            return {'status': 'failed', 'message': str(e)}, 201
        else:
            return {'status': 'success', 'url': result}, 201
