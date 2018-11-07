import tensorflow as tf
import numpy as np


class ConvLayer:
    # isPoolLayer- True表示卷积层中包含pool层
    # input - 网络的输入
    # ConvFilterKsize - 卷积核
    # ConvStrides - 卷积步长
    # ConvPadding - 卷积层是否padding
    # PoolKsize - 池化核大小
    # PoolStrides - 池化步长
    # PoolPadding - 池化层是否padding
    # name-该层网络的名称
    def __init__(self, isPoolLayer, input, ConvFilterKsize, ConvStrides, ConvPadding, PoolKsize, PoolStrides,
                 PoolPadding, name):

        self.ReLUOutput = None

        with tf.name_scope(name):
            self.ConvLayerWithPool(isPoolLayer, input, ConvFilterKsize, PoolKsize, ConvStrides, PoolStrides,
                                   ConvPadding, PoolPadding)

    # 卷积层中含pool层
    def ConvLayerWithPool(self, isPoolLayer, input, Conksize, Poolksize, ConvStrides, PoolStrides, ConvPadding,
                          PoolPadding):  # input-卷积输入，
        w = self.weight_variable(Conksize, 'W')
        bias = self.bias_variable([Conksize[len(Conksize) - 1]], "Bias")
        ConvOutput = self.conv2d(input, w, ConvStrides, ConvPadding, 'ConvOutput') + bias
        if isPoolLayer == True:
            PoolOutput = self.max_pool(ConvOutput, Poolksize, PoolStrides, PoolPadding, 'PoolOutput')
            self.ReLUOutput = tf.nn.relu(PoolOutput, "ReLu")
        else:
            self.ReLUOutput = tf.nn.relu(ConvOutput, "ReLu")

    # 给权重制造一些随机的噪声来打破完全对称
    def weight_variable(self, shape, name):
        with tf.name_scope(name):
            initial = tf.truncated_normal(shape, stddev=0.1)  # 截断的正态分布噪声，标准差为0.1
            return tf.Variable(initial)

    # 偏置函数
    def bias_variable(self, shape, name):
        with tf.name_scope(name):
            initial = tf.constant(0.1, shape=shape)  # 偏置增加一些小的正值（0.1）用来避免死亡节点（dead neurons）
            return tf.Variable(initial)

    # 卷积层函数
    def conv2d(self, x, W, strides, padding, name):
        with tf.name_scope(name):
            return tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],
                                padding=padding)  # tf.nn.conv2是tensorflow的二维卷积函数

    # 池化层函数
    def max_pool(self, x, ksize, strides, padding, name):
        with tf.name_scope(name):
            return tf.nn.max_pool(x, ksize=ksize, strides=[1, strides, strides, 1], padding=padding)


# 残差块
class ResBlock:
    # isPoolLayer- True表示卷积层中包含pool层
    # input - 网络的输入
    # ConvFilterKsize - 卷积核
    # ConvStrides - 卷积步长
    # ConvPadding - 卷积层是否padding
    # PoolKsize - 池化核大小
    # PoolStrides - 池化步长
    # PoolPadding - 池化层是否padding
    # LayerNReLUOutput - 哪一层跳过来的
    # res_kerne1_shape - 为了与该层线性输出的维度相同，在跳过来的那一层非线性输出a[l]上乘以一个系数阵
    # Ws_Padding - 乘以系数的那个阵是否需要padding
    # name -该层网络的名称

    def __init__(self, isPoolLayer, input, ConvFilterKsize, ConvStrides, ConvPadding, PoolKsize, PoolStrides,
                 PoolPadding, LayerNReLUOutput, res_kerne1_shape, Ws_Padding, name):
        self.ReLUOutput = None

        with tf.name_scope(name):
            self.ResLayer(isPoolLayer, input, ConvFilterKsize, PoolKsize, ConvStrides, PoolStrides,
                          ConvPadding, PoolPadding, LayerNReLUOutput, res_kerne1_shape, Ws_Padding)

    def ResLayer(self, isPoolLayer, input, Conksize, Poolksize, ConvStrides, PoolStrides, ConvPadding,
                 PoolPadding, LayerNReLUOutput, res_kerne1_shape, Ws_Padding):  # input-卷积输入，
        w = self.weight_variable(Conksize, 'W')
        bias = self.bias_variable([Conksize[len(Conksize) - 1]], "Bias")
        ConvOutput = self.conv2d(input, w, ConvStrides, ConvPadding, 'ConvOutput') + bias

        if isPoolLayer == True:
            PoolOutput = self.max_pool(ConvOutput, Poolksize, PoolStrides, PoolPadding, 'PoolOutput')
            res_kerne1 = self.weight_variable(res_kerne1_shape, 'res_kerne1')
            self.ReLUOutput = tf.nn.relu(PoolOutput +
                                         self.conv2d(LayerNReLUOutput, res_kerne1, 1, Ws_Padding, 'resblock'), "ReLU")
        else:
            res_kerne1 = self.weight_variable(res_kerne1_shape, 'res_kerne1')
            self.ReLUOutput = tf.nn.relu(
                ConvOutput + self.conv2d(LayerNReLUOutput, res_kerne1, 1, Ws_Padding, 'resblock'), "ReLU")

    # 给权重制造一些随机的噪声来打破完全对称
    def weight_variable(self, shape, name):
        with tf.name_scope(name):
            initial = tf.truncated_normal(shape, stddev=0.1)  # 截断的正态分布噪声，标准差为0.1
            return tf.Variable(initial)

    # 偏置函数
    def bias_variable(self, shape, name):
        with tf.name_scope(name):
            initial = tf.constant(0.1, shape=shape)  # 偏置增加一些小的正值（0.1）用来避免死亡节点（dead neurons）
            return tf.Variable(initial)

    # 卷积层函数
    def conv2d(self, x, W, strides, padding, name):
        with tf.name_scope(name):
            return tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],
                                padding=padding)  # tf.nn.conv2是tensorflow的二维卷积函数

    # 池化层函数
    def max_pool(self, x, ksize, strides, padding, name):
        with tf.name_scope(name):
            return tf.nn.max_pool(x, ksize=ksize, strides=[1, strides, strides, 1], padding=padding)


# 全连接层
class FullConnectLayer:
    # input - 网络的输入
    # Wsize - 权重
    # name-该层网络的名称

    def __init__(self, input, Wsize, name):
        self.FullOutput = None
        self.ReLUOutput = None
        with tf.name_scope(name):
            self.FullLayer(input, Wsize)

    def FullLayer(self, input, Wsize):
        w = self.weight_variable(Wsize, 'W')
        bias = self.bias_variable([Wsize[len(Wsize) - 1]], "Bias")
        self.FullOutput = tf.matmul(input, w, name='xXw') + bias
        self.ReLUOutput = tf.nn.relu(self.FullOutput, "ReLU")

    # 给权重制造一些随机的噪声来打破完全对称
    def weight_variable(self, shape, name):
        with tf.name_scope(name):
            initial = tf.truncated_normal(shape, stddev=0.1)  # 截断的正态分布噪声，标准差为0.1
            return tf.Variable(initial)

    # 偏置函数
    def bias_variable(self, shape, name):
        with tf.name_scope(name):
            initial = tf.constant(0.1, shape=shape)  # 偏置增加一些小的正值（0.1）用来避免死亡节点（dead neurons）
            return tf.Variable(initial)


# 输出层
class OutputLayer:
    # input - 网络的输入
    # Wsize - 权重
    # name-该层网络的名称
    def __init__(self, input, Wsize, name):
        self.Output = None
        with tf.name_scope(name):
            self.OutputLayer(input, Wsize)

    def OutputLayer(self, input, Wsize):
        w = self.weight_variable(Wsize, 'W')
        bias = self.bias_variable([Wsize[len(Wsize) - 1]], "Bias")
        FullOutput = tf.matmul(input, w, name='xXw') + bias
        self.Output = tf.nn.softmax(FullOutput, name='Output')

    # 给权重制造一些随机的噪声来打破完全对称
    def weight_variable(self, shape, name):
        with tf.name_scope(name):
            initial = tf.truncated_normal(shape, stddev=0.1)  # 截断的正态分布噪声，标准差为0.1
            return tf.Variable(initial)

    # 偏置函数
    def bias_variable(self, shape, name):
        with tf.name_scope(name):
            initial = tf.constant(0.1, shape=shape)  # 偏置增加一些小的正值（0.1）用来避免死亡节点（dead neurons）
            return tf.Variable(initial)
