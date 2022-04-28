import os
import zipfile

def unzip_file(input, output):
    """Unzip file.
    Args:
        input (str): Zip file.
        output (str): Target directory.
    """
    with zipfile.ZipFile(input, 'r') as zip:
        zip.extractall(output)

def zip_file(directory, output):
    """Create a zip file.
    Args:
        directory (str): Directory to be compressed.
        output (str): Output file .zip.
    """
    relroot = os.path.abspath(os.path.join(directory, os.pardir))
    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED) as zip:
        for root, dirs, files in os.walk(directory):
            zip.write(root, os.path.relpath(root, relroot))
            for file in files:
                filename = os.path.join(root, file)
                if os.path.isfile(filename):
                    arcname = os.path.join(os.path.relpath(root, relroot), file)
                    zip.write(filename, arcname)


def zip_file_stored(directory, output):
    """Create a zip file.
    Args:
        directory (str): Directory to be compressed.
        output (str): Output file .zip.
    """
    relroot = os.path.abspath(os.path.join(directory, os.pardir))
    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED) as zip:
        for root, dirs, files in os.walk(directory):
            zip.write(root, os.path.relpath(root, relroot))
            for file in files:
                filename = os.path.join(root, file)
                if os.path.isfile(filename):
                    arcname = os.path.join(os.path.relpath(root, relroot), file)
                    zip.write(filename, arcname)


def del_file(path_data):
    """删除文件夹下所有文件"""
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = os.path.join(path_data, i)  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data):  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)

