from config.setting import SERVER_PORT
import socket


def extract_ip():
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        st.connect(('10.255.255.255', 1))
        IP = st.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        st.close()
    return IP


def get_server_ip_and_port(server_file_path):
    return 'http://' + extract_ip() + ':' + str(SERVER_PORT) + '/' + server_file_path


# /Users/aldno/Downloads/flaskDemo-master/algorithm/cutimg2/static/excels_save/color_gray_mean/excel_color_gray_mean.xls
if __name__ == '__main__':
    import get_server_file_path
    print(get_server_ip_and_port(get_server_file_path.get_server_file_path(
        '/Users/aldno/Downloads/flaskDemo-master/algorithm/cutimg2/static/excels_save/color_gray_mean/excel_color_gray_mean.xls')))
