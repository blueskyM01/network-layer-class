import tensorflow as tf
import numpy as np
from m4_ConvClass import ConvLayer,FullConnectLayer,ResBlock,OutputLayer
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # 用自带的input_data.read_data_sets()方法加载mnist数据集
    session = tf.InteractiveSession()
    # 图片和标签占位符
    with tf.name_scope('Input_Org'):
        x = tf.placeholder(tf.float32, [None, 784], 'x')  # 图像
        y_ = tf.placeholder(tf.float32, [None, 10], 'y_')  # 标签

    # 将拍平的图像变成28X28
    with tf.name_scope('Image_Reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

    LayerOut1 = ConvLayer(isPoolLayer=True, input=x_image, ConvFilterKsize=[5, 5, 1, 32], ConvStrides=1, ConvPadding='VALID',
                     PoolKsize=[1, 2, 2, 1], PoolStrides=2, PoolPadding="VALID", name='layer1').ReLUOutput #12x12x32

    LayerOut2 = ConvLayer(isPoolLayer=True, input=LayerOut1, ConvFilterKsize=[3, 3, 32, 64], ConvStrides=1,ConvPadding='VALID',
                          PoolKsize=[1, 2, 2, 1], PoolStrides=2, PoolPadding="VALID", name='layer2').ReLUOutput  # 5x5x64

    LayerOut3 = ResBlock(isPoolLayer=False, input=LayerOut2, ConvFilterKsize=[3, 3, 64, 128], ConvStrides=1,ConvPadding='SAME',
                          PoolKsize=[1, 2, 2, 1], PoolStrides=2, PoolPadding="VALID",
                         LayerNReLUOutput=LayerOut2,res_kerne1_shape=[3,3,64,128],Ws_Padding='SAME',
                         name='layer3').ReLUOutput  # 5x5x128
    # LayerOut3 = ConvLayer(isPoolLayer=False, input=LayerOut2, ConvFilterKsize=[3, 3, 64, 128], ConvStrides=1,ConvPadding='SAME',
    #                       PoolKsize=[1, 2, 2, 1], PoolStrides=2, PoolPadding="VALID",name='layer3').ReLUOutput  # 5x5x128

    LayerOut4 = ConvLayer(isPoolLayer=False, input=LayerOut3, ConvFilterKsize=[3, 3, 128, 256], ConvStrides=1,ConvPadding='VALID',
                          PoolKsize=[1, 2, 2, 1], PoolStrides=2, PoolPadding="VALID",name='layer4').ReLUOutput  # 3x3x256

    convtofc = tf.reshape(LayerOut4,[-1,3*3*256],'convtofc')

    FullLayerOutput1 = FullConnectLayer(convtofc,[3*3*256,512],'FullLayerOutput1').ReLUOutput

    Output = OutputLayer(FullLayerOutput1,[512,10],'OutputLayer').Output

    with tf.name_scope('LossFunction'):
        loss = tf.reduce_mean(tf.square(Output - y_),reduction_indices=[1],name='loss')

    with tf.name_scope('Train'):
        TrainStep = tf.train.AdamOptimizer(1e-4).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(Output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    writer = tf.summary.FileWriter('logs', session.graph)
    writer.close()

    op_init = tf.global_variables_initializer().run()

    for i in range(1000000):
        batch = mnist.train.next_batch(50)  # 每一批取50张图片，batch[0]为图片，batch[1]为标签labels
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
            print('step %d,training accuracy %g' % (i, train_accuracy))

            TrainStep.run(feed_dict={x: batch[0], y_: batch[1]})

    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
