import numpy as np
import os


def write_value(file_name,  b):
    file = open('E:/DSAC/' + file_name + '.txt', 'w')
    file.write(b)
    file.close()


def read_value(file_name):    # 最后返回一个读取出来的参数b，所以read函数除了file_name没有其他输入
    file = open('E:/DSAC/' + file_name + '.txt', 'r')
    b = file.read()
    file.close()
    return b


def write_vector(file_name, v):
    file = open('E:/DSAC/' + file_name + '.txt', 'w')
    values = v
    for value in values:
        file.write(value)      # write 只能一个一个value write


def read_vector(file_name):
    file = open('E:/DSAC/' + file_name + '.txt', 'r')
    values = file.read()
    v = values
    file.close()
    return v

    # def write_map(self, file, m):

    # def read_map(self, file, m):  不清楚这里的map对应的是什么类型的变量


def write_cvmat(file_name, m):
    # m here should be a opencv matrix, but i use np.array first
    file = open('E:/DSAC/' + file_name + '.txt', 'w')
    lines = m
    for line in lines:
        for element in line:
            file.write(element)


def read_cvmat(file_name):
    # m 这里是返回值，有很多行array的一个矩阵
    f = open('E:/DSAC/' + file_name + '.txt', 'r')
    lines = f.readlines()
    m = np.array(lines)
    return m

