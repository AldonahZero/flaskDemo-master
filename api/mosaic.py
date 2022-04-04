import os
import traceback
import uuid
from datetime import datetime
from os import listdir
from os.path import isfile, join

from flask import request, flash, jsonify, current_app
from flask_restx import Resource, Namespace
from flask_restx.reqparse import RequestParser
from werkzeug.datastructures import FileStorage

from algorithm.HSI.showPseudoColor import show_image
from algorithm.HSI.FeatureExtraction.edge_feature import canny_edge_f
from algorithm.HSI.FeatureExtraction import HSI_NDWI_f, HSI_NDVI_f, Harris_points_f
from algorithm.HSI.HSI_grabcut import Hsi_grabcut_f
from algorithm.HSI.FeatureExtraction.gray_feature import gray_mean_dif_f, gray_var_dif_f, gray_histogram_dif_f
from algorithm.HSI.band_Selection import ECA_f
from common.file_tools import unzip_file
from common.getUploadLocation import get_upload_location
from common.mysql_operate import db_session, HSIPictureFile, MOSPictureFile, MOSPictureFolder, MOSResult
from common.remove_file_dir import remove_file_dir
from algorithm.mosaic.resizeKeepGPS import mosaic

mos_ns = Namespace('mos', description='图片拼接')

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}
MOS_UPLOAD_FOLDER = 'algorithm/mosaic/static/upload/'
MOS_RESULT_FOLDER = 'algorithm/mosaic/static/result/'

# 上传图片路径
UPLOAD_PATH = "algorithm/mosaic/static/images"

# 文件上传格式
parser: RequestParser = mos_ns.parser()
parser.add_argument('file', location='files', type=FileStorage, required=True)


def allowed_file(filename):
    """判断文件是否允许上传"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@mos_ns.route("/zipfile", doc={"description": "上传图片压缩包"})
class UploadHandler(Resource):
    def get(self):
        '''查看所有上传的压缩文件'''
        try:
            session = db_session()
            folders = session.query(MOSPictureFolder).all()
            data = []
            for folder in folders:
                data.append(folder.to_json())
            session.close();
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 201, 'message': '查找成功', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})

    @mos_ns.expect(parser, validate=True)
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
            unzip_file(path, os.path.join(UPLOAD_PATH, save_filename))

            # 改名
            # unzip_file_loaction = os.path.join(UPLOAD_PATH, file.filename)[0:-4]
            # unzip_file_uid_loaction = os.path.join(UPLOAD_PATH, save_filename)
            # os.rename(unzip_file_loaction, unzip_file_uid_loaction)
            # 删除压缩包
            remove_file_dir(path)

            # 前端路径
            proLoadPath = os.path.join('algorithm/mosaic/static/images', save_filename) + "/"
            realProLoadPath = os.path.join(UPLOAD_PATH, save_filename) + "/"
            print(realProLoadPath)
            filenames = [f for f in listdir(realProLoadPath) if isfile(join(proLoadPath, f))]

            session = db_session()
            print(save_filename)
            folder = MOSPictureFolder(fid=save_filename, path=proLoadPath, create_time=datetime.now())
            session.add(folder)
            session.commit()
            for filename in filenames:
                pic = MOSPictureFile(pid=str(uuid.uuid1()), path=proLoadPath + filename, fid=save_filename, create_time=datetime.now())
                session.add(pic)
                session.commit()
            session.close()
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 201, 'message': '查找成功', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '上传压缩包成功', 'file_name': save_filename})


@mos_ns.route('/mosaic/<file_name>')
@mos_ns.param('file_name', '上传时返回的36位随机文件名称')
class MOSAIC(Resource):
    def get(self, file_name):
        '''获取拼接后图片'''
        try:
            result = mosaic(file_name)
            save_path = MOS_RESULT_FOLDER + file_name + "/" + result + ".png";
            session = db_session()
            folder = MOSResult(pid=str(uuid.uuid1()), fid=file_name, path=save_path, create_time=datetime.now())
            session.add(folder)
            session.commit()
            session.close()
        except BaseException as e:
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': save_path})