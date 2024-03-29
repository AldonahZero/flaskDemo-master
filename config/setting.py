# 服务端口配置
SERVER_PORT = 5001

# MySQL配置
MYSQL_HOST = "47.96.111.114"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWD = "980506"
MYSQL_DB = "flask_demo"

# Redis配置
REDIS_HOST = "192.168.0.1"
REDIS_PORT = 6379
REDIS_PASSWD = ""
# token过期时间(单位：秒)
EXPIRE_TIME = 600

# MD5加密盐值
MD5_SALT = "test2020#%*"

# url配置
API_VERSION = "/api/v1"

# matplotlib.imshow
# MATPLOTLIB_INSHOW = True
MATPLOTLIB_INSHOW = False

# 文件上传位置
UPLOAD_FOLDER = 'static/uploads'
# 文件上传位置
RESULT_FOLDER = 'static/result'
# session secret_key
SECRET_KEY = 'back_dev_flask'

# 上传文件最大大小
MAX_FILE_SIZE = 1024 * 1024 * 50
