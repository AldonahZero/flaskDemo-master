import os
import uuid
from flask import jsonify, request, Blueprint, render_template, redirect, make_response, flash, url_for
from flask_restx import Api, Resource, fields, Namespace
from werkzeug.datastructures import FileStorage
from config.setting import UPLOAD_FOLDER
from config.setting import RESULT_FOLDER
from algorithm.HSI.showPseudoColor import show_image

hsi_ns = Namespace('hsi', description='高光谱部分算法')

upload_parser = hsi_ns.parser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'raw'}


@hsi_ns.route('/upload/')
class Upload(Resource):
    @hsi_ns.param('file', '文件')
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
            filename = file.filename
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            return {'message': 'success', 'url': os.path.join(UPLOAD_FOLDER, filename)}
        return {'message': "file not allow"}, 201


def allowed_file(filename):
    """判断文件是否允许上传"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@hsi_ns.route('/showPseudoColor/')
class showPseudoColor(Resource):
    @hsi_ns.param('file', '文件')
    def post(self):
        if 'file' not in request.files:
            flash('No file part')
            return {'message': 'No file part'}, 201
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return {'message': 'No selected file'}, 201
        if file and allowed_file(file.filename):
            filename = str(uuid.uuid1()) + '.' + file.filename.rsplit('.', 1)[1]
            save_path = os.path.join(UPLOAD_FOLDER + '/hsi', filename)
            file.save(save_path)
            rel_out_path = os.path.join(RESULT_FOLDER + '/hsi', str(uuid.uuid1()) + '.jpg')
            abs_out_path = os.path.abspath(rel_out_path)
            try:
                show_image(save_path, abs_out_path)
            except BaseException as e:
                return {'status': 'failed', 'message': str(e)}, 201
            else:
                return {'status': 'success', 'url': rel_out_path}, 201
        return {'message': "file not allow"}, 201
