import os, sys
import logging
import logging.config
from flask import Flask, request, render_template, Blueprint,Response
from config.setting import SERVER_PORT
from config.setting import SECRET_KEY
from werkzeug.datastructures import Headers
from werkzeug.middleware.proxy_fix import ProxyFix

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)  # 将项目根路径临时加入环境变量，程序退出后失效
# 修改默认存储路径
app = Flask(__name__,static_folder="algorithm", static_url_path="/algorithm")
app.secret_key = SECRET_KEY
# app.wsgi_app = ProxyFix(app.wsgi_app)
app.config["JSON_AS_ASCII"] = False  # jsonify返回的中文正常显示

# 注册蓝图
from api.__init__ import api_v1, cors_headers

app.url_map.strict_slashes = False

app.register_blueprint(api_v1)


handler = logging.FileHandler(filename="test.log", encoding='utf-8')
handler.setLevel("DEBUG")
format_ ="%(asctime)s[%(name)s][%(levelname)s] :%(levelno)s: %(message)s"
formatter = logging.Formatter(format_)
handler.setFormatter(formatter)
app.logger.addHandler(handler)
# debug : 打印全部的日志,详细的信息,通常只出现在诊断问题上
# info : 打印info,warning,error,critical级别的日志,确认一切按预期运行
# warning : 打印warning,error,critical级别的日志,一个迹象表明,一些意想不到的事情发生了,或表明一些问题在不久的将来(例如。磁盘空间低”),这个软件还能按预期工作
# error : 打印error,critical级别的日志,更严重的问题,软件没能执行一些功能
# critical : 打印critical级别,一个严重的错误,这表明程序本身可能无法继续运行
# 日志级别：CRITICAL >ERROR> WARNING > INFO> DEBUG> NOTSET
# current_app.logger.info("this is info")
# current_app.logger.debug("this is debug")
# current_app.logger.warning("this is warning")
# current_app.logger.error("this is error")
# current_app.logger.critical("this is critical")

@app.after_request
def after_request(response: Response):
    headers = dict(response.headers)
    headers["Cache-Control"] = "no-transform"
    headers.update(**cors_headers)
    response.headers = Headers(headers)

    path = request.path
    if path.startswith("/api/v") and path.endswith("/") and path.count("/") == 3:
        body = response.get_data().replace(b"<head>", b"<head><style>.models {display: none !important}</style>")
        return Response(body, response.status_code, response.headers)
    return response


if __name__ == '__main__':
    # host为主机ip地址，port指定访问端口号，debug=True设置调试模式打开
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=True)
