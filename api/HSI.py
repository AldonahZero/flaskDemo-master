from flask import jsonify, request,Blueprint, render_template, redirect
from common.mysql_operate import db
from common.redis_operate import redis_db
from common.md5_operate import get_md5
import re, time

hsi = Blueprint('hsi',__name__)


@hsi.route('/hsi')
def hello_world():
    return 'Hello Worldaa!'