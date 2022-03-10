import os
from os.path import isfile, join

from numpy.core.defchararray import rfind

STATIC_FOILD = '/algorithm'
project_lo = os.path.abspath(os.path.dirname(__file__))
indexof = rfind(project_lo ,'/', start=0 ,end=len(project_lo))
project_lo_alg = project_lo[0:indexof] + STATIC_FOILD

def get_upload_location(datalocation):
    return project_lo_alg + datalocation

def get_server_location(datalocation):
    return STATIC_FOILD + datalocation