import os, sys
from flask import Flask, request, render_template, Blueprint,Response
from config.setting import SERVER_PORT
from config.setting import SECRET_KEY
from werkzeug.datastructures import Headers
from werkzeug.middleware.proxy_fix import ProxyFix

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)  # 将项目根路径临时加入环境变量，程序退出后失效
app = Flask(__name__)
app.secret_key = SECRET_KEY
# app.wsgi_app = ProxyFix(app.wsgi_app)
app.config["JSON_AS_ASCII"] = False  # jsonify返回的中文正常显示

# 注册蓝图
from api.__init__ import api_v1, cors_headers

app.url_map.strict_slashes = False

app.register_blueprint(api_v1)

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
