#自动驼峰
import subprocess

def write_to_clipboard(output):
    process = subprocess.Popen(
        'pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
    process.communicate(output.encode('utf-8'))

res = ""
while True:
    a = input()
    astrs = a.split(' ')
    for astr in astrs:
        if astr.islower():
            new_astr = astr.capitalize()
            res += new_astr
        else:
            res =  res + astr[0].lower()+astr[1:]

    print(res)
    write_to_clipboard(res)
    res = ""
