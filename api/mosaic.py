import os
import shutil
import traceback
import uuid
from datetime import datetime
from os import listdir
from os.path import isfile, join

import numpy as np
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
from algorithm.mosaic.c import image_map
from common.file_tools import unzip_file, del_file, zip_file
from common.getUploadLocation import get_upload_location
from common.get_server_file_path import get_server_file_path
from common.get_server_ip_and_port import get_server_ip_and_port
from common.mysql_operate import db_session, HSIPictureFile, MOSPictureFile, MOSPictureFolder, MOSResult
from common.remove_file_dir import remove_file_dir
from algorithm.mosaic.resizeKeepGPS import mosaic

mos_ns = Namespace('mos', description='图片拼接')

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}
MOS_UPLOAD_FOLDER = 'algorithm/mosaic/static/upload/'
MOS_RESULT_FOLDER = 'algorithm/mosaic/static/result/'
MATCH_RESULT_PATH = 'algorithm/mosaic/static/matchresult/'
P2_ORIGIN_PATH = 'algorithm/mosaic/static/P2.txt'  # P2原始文件位置
P2_NEW_FOLDER_PATH = 'algorithm/mosaic/static/P2/'  # P2新的文件夹存放位置
GGP_FOLDER = 'algorithm/mosaic/static/matchfile/ggp/'
KJG_FOLDER = 'algorithm/mosaic/static/matchfile/kjg/'
HW_FOLDER = 'algorithm/mosaic/static/matchfile/hw/'

# 上传图片路径
UPLOAD_PATH = "algorithm/mosaic/static/images"

# 文件上传格式
parser: RequestParser = mos_ns.parser()
parser.add_argument('file', location='files', type=FileStorage, required=True)

# 带参文件上传格式
arg_parser: RequestParser = mos_ns.parser()
arg_parser.add_argument('file', location='files', type=FileStorage, required=True)
arg_parser.add_argument('key', type=str, required=True)

# 局部匹配参数格式
CutImgDataParser: RequestParser = mos_ns.parser()
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


def allowed_file(filename):
    """判断文件是否允许上传"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_mosaic_result(pid):
    """按pid查询拼接结果"""
    session = db_session()
    result = session.query(MOSResult).filter(MOSResult.pid == pid).all()
    return result[0]

def get_file_name(path):
    """查找文件夹所有文件"""
    file_name = []
    for root, dirs, files in os.walk(path):
        for i in range(len(files)):
            file_name.append(files[i])
    return file_name

def file_add_path(path, file_names):
    file_name_new = []
    for file_name in file_names:
        file_name_new.append(path + file_name)
    return file_name_new


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
            session.close()
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
                pic = MOSPictureFile(pid=str(uuid.uuid1()), path=proLoadPath + filename, fid=save_filename,
                                     create_time=datetime.now())
                session.add(pic)
                session.commit()
            session.close()
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '查找成功', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '上传压缩包成功', 'file_name': save_filename})


def remove_file(old_path, new_path):
    shutil.move(old_path, new_path)


@mos_ns.route('/mosaic/<file_name>')
@mos_ns.param('file_name', '上传时返回的36位随机文件名称')
class MOSAIC(Resource):
    def get(self, file_name):
        '''获取拼接后图片'''
        try:
            result = mosaic(file_name)
            # 因为调用exe输出限制，所以采用拼接方式保存结果记录
            save_path = MOS_RESULT_FOLDER + file_name + "/" + result + ".png"
            remove_file(P2_ORIGIN_PATH, P2_NEW_FOLDER_PATH + file_name + '.txt')
            session = db_session()
            result_id = str(uuid.uuid1())
            folder = MOSResult(pid=result_id, fid=file_name, path=save_path, create_time=datetime.now())
            session.add(folder)
            session.commit()
            session.close()
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': 'failed', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': 'success', 'result': save_path, 'result_id': result_id})


@mos_ns.route("/pics/<file_name>", doc={"description": "查看上传的图片列表"})
class getPics(Resource):
    def get(self, file_name):
        '''查看所有上传的压缩文件'''
        try:
            session = db_session()
            pics = session.query(MOSPictureFile).filter(MOSPictureFile.fid == file_name).all()
            data = []
            for pic in pics:
                data.append(pic.to_json())
            session.close()
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 201, 'message': '查找成功', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})


@mos_ns.route("/mosaic_result_all", doc={"description": "查看所有拼接结果"})
class getPics(Resource):
    def get(self):
        '''查看所有拼接结果'''
        try:
            session = db_session()
            pics = session.query(MOSResult).all()
            data = []
            for pic in pics:
                data.append(pic.to_json())
            session.close()
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 201, 'message': '查找成功', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})


@mos_ns.route("/mosaic_result/<pid>", doc={"description": "按id查找拼接结果"})
class getPic(Resource):
    def get(self, pid):
        '''查看所有拼接结果'''
        try:
            session = db_session()
            pics = session.query(MOSResult).filter(MOSResult.pid == pid).all()
            data = []
            for pic in pics:
                data.append(pic.to_json())
            session.close()
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 201, 'message': '查找成功', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'data': data})


@mos_ns.route("/ggp_upload", doc={"description": "高光谱压缩文件上传"})
class ggpFileUpload(Resource):
    @mos_ns.expect(arg_parser, validate=True)
    def post(self):
        '''高光谱压缩文件上传'''
        try:
            # 普通参数获取
            # 获取文件对象
            file = request.files.get('file')
            # args = arg_parser.parse_args()
            save_folder = request.form.get('fid')
            save_path = GGP_FOLDER + save_folder

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # 删除之前上传的文件
            del_file(save_path)
            path = os.path.join(UPLOAD_PATH, file.filename)
            # 保存压缩包
            file.save(path)
            # 解压缩
            unzip_file(path, save_path)
            # 删除压缩包
            remove_file_dir(path)

        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '上传失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '上传文件成功'})


@mos_ns.route("/kjg_upload", doc={"description": "可见光压缩文件上传"})
class kjgFileUpload(Resource):
    @mos_ns.expect(arg_parser, validate=True)
    def post(self):
        '''可见光压缩文件上传'''
        try:
            # 普通参数获取
            # 获取文件对象
            file = request.files.get('file')
            # args = arg_parser.parse_args()
            save_folder = request.form.get('fid')
            save_path = KJG_FOLDER + save_folder

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # 删除之前上传的文件
            del_file(save_path)
            path = os.path.join(UPLOAD_PATH, file.filename)
            # 保存压缩包
            file.save(path)
            # 解压缩
            unzip_file(path, save_path)
            # 删除压缩包
            remove_file_dir(path)

        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '上传失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '上传文件成功'})


@mos_ns.route("/hw_upload", doc={"description": "红外光压缩文件上传"})
class hwFileUpload(Resource):
    @mos_ns.expect(arg_parser, validate=True)
    def post(self):
        '''红外光压缩文件上传'''
        try:
            # 普通参数获取
            # 获取文件对象
            file = request.files.get('file')
            # args = arg_parser.parse_args()
            save_folder = request.form.get('fid')
            save_path = HW_FOLDER + save_folder

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # 删除之前上传的文件
            del_file(save_path)
            path = os.path.join(UPLOAD_PATH, file.filename)
            # 保存压缩包
            file.save(path)
            # 解压缩
            unzip_file(path, save_path)
            # 删除压缩包
            remove_file_dir(path)

        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '上传失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '上传文件成功'})


@mos_ns.route("/match", doc={"description": "局部匹配;返回高光谱压缩文件，可见光和红外为图片文件"})
class MATCH(Resource):
    @mos_ns.expect(CutImgDataParser, validate=True)
    def post(self):
        '''局部匹配'''
        try:
            params = CutImgDataParser.parse_args()
            cutposs = params["cutposs"]
            cutimg_pid = params["cutimg_pid"]
            cutposs_data = np.asfarray(eval(cutposs))
            mosaic_result = get_mosaic_result(cutimg_pid)
            print(cutposs_data)
            fid = mosaic_result.fid
            image_map(mosaic_result.path, HW_FOLDER+fid, KJG_FOLDER+fid, GGP_FOLDER+fid, fid, cutposs_data)
            path_result_ggp = MATCH_RESULT_PATH + 'result_ggp/' + fid
            path_result_kjg = MATCH_RESULT_PATH + 'result_kjg/' + fid
            path_result_hw = MATCH_RESULT_PATH + 'result_hw/' + fid
            path_result_ggp_zip = MATCH_RESULT_PATH + 'result_ggp_zip/' + fid + '.zip'
            # zip_file(path_result_ggp,path_result_ggp_zip)
            ggp_result = get_server_ip_and_port(get_server_file_path(os.path.abspath(path_result_ggp_zip)))
        except BaseException as e:
            current_app.logger.error(traceback.format_exc())
            return jsonify({'code': 400, 'message': '失败', 'data': str(e)})
        else:
            return jsonify({'code': 201, 'message': '查找成功', 'ggp_result': ggp_result,
                            'hw_result': file_add_path(path_result_hw + '/', get_file_name(path_result_hw)),
                            'kjg_result': file_add_path(path_result_kjg + '/', get_file_name(path_result_kjg))})


