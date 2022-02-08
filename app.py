import os, sys
from config.setting import SERVER_PORT
from flask import Blueprint,Flask
from api.user import user
from api.cutting import cutting


app = Flask(__name__)

# 注册蓝图
app.register_blueprint(user,url_prefix='/user')
app.register_blueprint(cutting,url_prefix='/cutting')

app.config["JSON_AS_ASCII"] = False  # jsonify返回的中文正常显示
# 项目根路径
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)  # 将项目根路径临时加入环境变量，程序退出后失效

if __name__ == '__main__':
    # host为主机ip地址，port指定访问端口号，debug=True设置调试模式打开
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=True)
