# coding:utf-8
# from flask import Flask, request, jsonify
from flask import jsonify, request,Blueprint, render_template, redirect
import os
# import redis
from os.path import isfile, join
from os import listdir
# import json
#
# from util import unzip_file
# from A import grey_compare
from common.mysql_operate import db_session, Pic
from common.file_tools import unzip_file
# from MyEncoder import MyEncoder

from algorithm.multiplePerspectives.A import grey_compare, canny_compare

mul = Blueprint('mul',__name__)


UPLOAD_PATH = os.path.join(os.path.dirname(__file__), 'static/images')
# 上传文件
@mul.route('/upload', methods=['POST'])
def upload():
    # 普通参数获取
    # 获取pichead文件对象
    file = request.files.get('file')
    path = os.path.join(UPLOAD_PATH, file.filename)
    print(path)
    file.save(path)

    # 解压缩

    unzip_file(path,UPLOAD_PATH)

    # 前端路径
    loadpath = ('static/images/%s' % file.filename)[0:-4]+"/"
    filenames = [f for f in listdir(loadpath) if isfile(join(loadpath, f))]

    session = db_session()
    for filename in filenames:
        new_file = Pic(url=loadpath+filename)
        session.add(new_file)
        session.commit()
    session.close()
    return jsonify({'code': 1, 'message': '上传成功'})


UPLOAD_PATH = os.path.join(os.path.dirname(__file__), 'static/images')

# 返回全部图片以供选装
@mul.route('/getAllImage')
def getAllImage():
    # 普通参数获取
    session = db_session()
    pics = session.query(Pic).all()
    # json.dumps(data, cls=MyEncoder)
    # print(pics.__dict__)
    data = []
    for pic in pics:
        data.append(pic.to_json())
    # print(data)
    return jsonify({'code': 1, 'message': '上传成功', 'data': data})

# 根据图片调用算法
@mul.route('/lhy')
def lhy():
    # 普通参数获取
    pids_str = request.values.get('pids').split(',')
    pids =  list(map(int, pids_str))
    session = db_session()
    pics = Pic.query.filter(Pic.pid.in_(pids)).all()
    # pics = session.query(Pic).filter(Pic.pid.in_(pids)).all()

    data = grey_compare(pics)
    return jsonify({'code': 1, 'message': '上传成功', 'data': data })