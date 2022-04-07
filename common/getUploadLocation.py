import os
from os.path import isfile, join

from numpy.core.defchararray import rfind

STATIC_FOILD = 'algorithm'
project_lo = os.path.abspath(os.path.dirname(__file__))
indexof = rfind(project_lo ,os.path.sep, start=0 ,end=len(project_lo))
project_lo_alg = os.path.join(project_lo[0:indexof], STATIC_FOILD)

def get_upload_location(datalocation):
    path = os.path.join(project_lo_alg, datalocation)
    # print('path1' +path)
    return path

def get_server_location(datalocation):
    path = os.path.join(STATIC_FOILD, datalocation)
    # print('path2' + path)
    return path