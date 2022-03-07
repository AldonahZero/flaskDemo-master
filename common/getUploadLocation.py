import os
from os.path import isfile, join

from numpy.core.defchararray import rfind

project_lo = os.path.abspath(os.path.dirname(__file__))
indexof = rfind(project_lo ,'/', start=0 ,end=len(project_lo))
project_lo_alg = project_lo[0:indexof] + '/algorithm'

def get_upload_location(datalocation):

    return project_lo_alg + datalocation