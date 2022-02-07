import os, sys
from flask import Flask,request,render_template
from config.setting import SERVER_PORT, API_V1
from api.featureExtraction import fea
from api.HSI import hsi

# 项目根路径
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)  # 将项目根路径临时加入环境变量，程序退出后失效

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # jsonify返回的中文正常显示
app.register_blueprint(fea, url_prefix=API_V1 + '/fea')
app.register_blueprint(hsi, url_prefix=API_V1 + '/hsi')

if __name__ == '__main__':
    # host为主机ip地址，port指定访问端口号，debug=True设置调试模式打开
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=True)
