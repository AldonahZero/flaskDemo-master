import pymysql
from config.setting import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWD, MYSQL_DB
from sqlalchemy import create_engine, ForeignKey, Column, Integer, String, Text, DateTime, \
    and_, or_, SmallInteger, Float, DECIMAL, desc, asc, Table, join, event
from sqlalchemy.orm import relationship, backref, sessionmaker, scoped_session, aliased, mapper
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method
from sqlalchemy.orm.collections import attribute_mapped_collection
import datetime

# 数据库url
# '数据库类型+数据库驱动名称://用户名:口令@机器地址:端口号/数据库名'
# MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWD, MYSQL_DB
engine = create_engine(
    "mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8".format(MYSQL_USER, MYSQL_PASSWD, MYSQL_HOST, MYSQL_PORT, MYSQL_DB),
    pool_recycle=7200)

Base = declarative_base()

db_session = scoped_session(sessionmaker(autocommit=False,
                                         autoflush=False,
                                         bind=engine))

Base.query = db_session.query_property()


class User(Base):
    __tablename__ = 'user'

    uid = Column('uid', Integer, primary_key=True)
    phone_number = Column('phone_number', String(11), index=True)
    password = Column('password', String(30))
    nickname = Column('nickname', String(30), index=True, nullable=True)
    register_time = Column('register_time', DateTime, index=True, default=datetime.datetime.now)


class Task(Base):
    __tablename__ = 'task'

    tuid = Column('tuid', Integer, primary_key=True)
    uid = Column('uid', Integer, index=True)
    puid = Column('puid', Integer)
    resultUrl = Column('resultUrl', String(30), index=True, nullable=True)


class PictureFile(Base):
    __tablename__ = 'picturefile'

    puid = Column('puid', Integer, primary_key=True)
    uid = Column('uid', Integer, index=True)
    path = Column('path', String(30))
    dataType = Column('dataType', String(30), index=True, nullable=True)
    pictureLabels = Column('pictureLabels', String(30))


class HSIPictureFile(Base):
    __tablename__ = 'hsi_picture'

    pid = Column('pid', String(36), primary_key=True)
    uid = Column('uid', String(36), index=True)
    file_path = Column('file_path', String(128))
    picture_path = Column('picture_path', String(128))
    create_time = Column('create_time', DateTime)



class HSIResultFile(Base):
    __tablename__ = 'hsi_result_file'

    fid = Column('fid', String(36), primary_key=True)
    pid = Column('pid', String(36), index=True)
    type = Column('type', String(10))
    path = Column('path', String(128))
    create_time = Column('create_time', DateTime)

class FEAPictureFile(Base):
    __tablename__ = 'fea_picture'

    pid = Column('pid', Integer, primary_key=True)
    url = Column('url', String(128))
    create_time = Column('create_time', DateTime)

    def to_json(self):
        dict = self.__dict__
        if "_sa_instance_state" in dict:
            del dict["_sa_instance_state"]
        return dict

    def __repr__(self):
        return '<FeaPic: %s %s >' % (self.pid, self.url)

class MOSPictureFolder(Base):
    __tablename__ = 'mos_picture_folder'

    fid = Column('fid', String(36), primary_key=True)
    path = Column('path', String(128))
    create_time = Column('create_time', DateTime)

    def to_json(self):
        dict = self.__dict__
        if "_sa_instance_state" in dict:
            del dict["_sa_instance_state"]
        return dict


class MOSPictureFile(Base):
    __tablename__ = 'mos_picture_file'

    pid = Column('pid', String(36), primary_key=True)
    fid = Column('fid', String(36))
    path = Column('path', String(128))
    create_time = Column('create_time', DateTime)


class MOSResult(Base):
    __tablename__ = 'mos_result'

    pid = Column('pid', String(36), primary_key=True)
    fid = Column('fid', String(36))
    path = Column('path', String(128))
    create_time = Column('create_time', DateTime)


class Pic(Base):
    __tablename__ = 'pic'

    pid = Column('pid', Integer, primary_key=True)
    url = Column('url', String(30))

    def to_json(self):
        dict = self.__dict__
        if "_sa_instance_state" in dict:
            del dict["_sa_instance_state"]
        return dict

    def __repr__(self):
        return '<Pic: %s %s >' % (self.pid, self.url)


if __name__ == '__main__':
    Base.metadata.create_all(engine)

# class MysqlDb():
#
#     def __init__(self, host, port, user, passwd, db):
#         # 建立数据库连接
#         self.conn = pymysql.connect(
#             host=host,
#             port=port,
#             user=user,
#             passwd=passwd,
#             db=db,
#             autocommit=True
#         )
#         # 通过 cursor() 创建游标对象，并让查询结果以字典格式输出
#         self.cur = self.conn.cursor(cursor=pymysql.cursors.DictCursor)
#
#     def __del__(self): # 对象资源被释放时触发，在对象即将被删除时的最后操作
#         # 关闭游标
#         self.cur.close()
#         # 关闭数据库连接
#         self.conn.close()
#
#     def select_db(self, sql):
#         """查询"""
#         # 检查连接是否断开，如果断开就进行重连
#         self.conn.ping(reconnect=True)
#         # 使用 execute() 执行sql
#         self.cur.execute(sql)
#         # 使用 fetchall() 获取查询结果
#         data = self.cur.fetchall()
#         return data
#
#     def execute_db(self, sql):
#         """更新/新增/删除"""
#         try:
#             # 检查连接是否断开，如果断开就进行重连
#             self.conn.ping(reconnect=True)
#             # 使用 execute() 执行sql
#             self.cur.execute(sql)
#             # 提交事务
#             self.conn.commit()
#         except Exception as e:
#             print("操作出现错误：{}".format(e))
#             # 回滚所有更改
#             self.conn.rollback()
#
# db = MysqlDb(MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWD, MYSQL_DB)
