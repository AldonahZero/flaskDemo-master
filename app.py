import os, sys
from flask import Flask, request, render_template, Blueprint
from config.setting import SERVER_PORT
from config.setting import SECRET_KEY
# from flask_apidoc.commands import GenerateApiDoc
# from flask_script import Manager
from werkzeug.middleware.proxy_fix import ProxyFix

def create_app():
    # 创建Flask对象
    # 项目根路径
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, BASE_PATH)  # 将项目根路径临时加入环境变量，程序退出后失效
    app = Flask(__name__)
    app.secret_key = SECRET_KEY
    # app.wsgi_app = ProxyFix(app.wsgi_app)
    app.config["JSON_AS_ASCII"] = False  # jsonify返回的中文正常显示

    # 注册蓝图
    from api.__init__ import api_v1
    app.url_map.strict_slashes = False

    app.register_blueprint(api_v1)
    # from api.swagger_demo import api_v1
    # from api.featureExtraction import fea_v1
    # from api.HSI import hsi

    # from api.multiplePerspectives import mul
    # app.register_blueprint(api_v1)
    # app.register_blueprint(fea_v1)

    # app.register_blueprint(hsi, url_prefix=API_V1 + '/hsi')
    # app.register_blueprint(mul, url_prefix=API_V1 + '/mul')
    return app

if __name__ == '__main__':
    app = create_app()
    # host为主机ip地址，port指定访问端口号，debug=True设置调试模式打开
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=True)
