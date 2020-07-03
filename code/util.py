import re
import os


def split(*argv):
    length = len(argv)
    if length == 2:
        s = argv[0]
        delim = argv[1]
        return s.split(delim)
    elif length ==1:
        s = argv[0]
        return s.split()


def splitOffDigits(string):
    # find first number
    num = re.findall(r"\d", string)
    # split
    if string[0] == num[0]:
        return [string, []]
    else:
        split_string = string.split(num[0], 1)
        split_string[1] = num[0]+split_string[1]
        return split_string


def endsWith(str, key):
    key_word = re.findall(re.compile(r""+key+"$"), str)
    if key_word:
        return True
    else:
        return False


def intToString(number, minLength=0):
    string = str(number)
    while len(string) < minLength:
        string = '0'+string
    return string


def floatToString(number):
    return str(number)


def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


def getSubPaths(basePath):
    dirpath = []
    if os.path.exists(basePath):
        for name in os.listdir(basePath):
            dirpath.append(os.path.join(basePath, name))
    else:
        print('Could not open directory: '+basePath)
    return dirpath


def getFiles(basePath, ext, silent = False):
    dirfile = []
    path_list = os.listdir(basePath)
    path_list.sort()
    if os.path.exists(basePath):
        for files in path_list:
            if not os.path.isdir(files):
                if endsWith(files, ext):
                    dirfile.append(os.path.join(basePath, files))
    else:
        if not silent:
            print('Could not open directory: ' + basePath)
    return dirfile

