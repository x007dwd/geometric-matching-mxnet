"""References:

Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for
large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
"""
import mxnet as mx

def get_symbol(num_classes, **kwargs):
    ## define VGG-ccn geometric
    data_left = mx.symbol.Variable(name="data")
    # group 1
    conv1_1a = mx.symbol.Convolution(data=data_left, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1a = mx.symbol.Activation(data=conv1_1a, act_type="relu", name="relu1_1a")
    pool1a = mx.symbol.Pooling(
        data=relu1_1a, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1a")
    # group 2
    conv2_1a = mx.symbol.Convolution(
        data=pool1a, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1a")
    relu2_1a = mx.symbol.Activation(data=conv2_1a, act_type="relu", name="relu2_1a")
    pool2a = mx.symbol.Pooling(
        data=relu2_1a, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2a")
    # group 3
    conv3_1a = mx.symbol.Convolution(
        data=pool2a, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1a")
    relu3_1a = mx.symbol.Activation(data=conv3_1a, act_type="relu", name="relu3_1a")
    conv3_2a = mx.symbol.Convolution(
        data=relu3_1a, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2a")
    relu3_2a = mx.symbol.Activation(data=conv3_2a, act_type="relu", name="relu3_2a")
    pool3a = mx.symbol.Pooling(
        data=relu3_2a, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3a")
    # group 4
    conv4_1a = mx.symbol.Convolution(
        data=pool3a, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1a")
    relu4_1a = mx.symbol.Activation(data=conv4_1a, act_type="relu", name="relu4_1a")
    conv4_2a = mx.symbol.Convolution(
        data=relu4_1a, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2a")
    relu4_2a = mx.symbol.Activation(data=conv4_2a, act_type="relu", name="relu4_2a")
    pool4a = mx.symbol.Pooling(
        data=relu4_2a, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4a")
    # group 5
    conv5_1a = mx.symbol.Convolution(
        data=pool4a, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1a")
    relu5_1a = mx.symbol.Activation(data=conv5_1a, act_type="relu", name="relu5_1a")
    conv5_2a = mx.symbol.Convolution(
        data=relu5_1a, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2a")
    relu5_2a = mx.symbol.Activation(data=conv5_2a, act_type="relu", name="relu5_2a")
    pool5a = mx.symbol.Pooling(
        data=relu5_2a, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5a")
    normal_a = mx.symbol.L2Normalization(data=pool5a, mode='instance', mame="l2norma")


    data_right = mx.symbol.Variable(name="data")
    # group 1
    conv1_1b = mx.symbol.Convolution(data=data_right, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1b")
    relu1_1b = mx.symbol.Activation(data=conv1_1b, act_type="relu", name="relu1_1b")
    pool1b = mx.symbol.Pooling(
        data=relu1_1b, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1b")
    # group 2
    conv2_1b = mx.symbol.Convolution(
        data=pool1b, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1b")
    relu2_1b = mx.symbol.Activation(data=conv2_1b, act_type="relu", name="relu2_1b")
    pool2b = mx.symbol.Pooling(
        data=relu2_1b, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2b")
    # group 3
    conv3_1b = mx.symbol.Convolution(
        data=pool2b, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1b")
    relu3_1b = mx.symbol.Activation(data=conv3_1b, act_type="relu", name="relu3_1b")
    conv3_2b = mx.symbol.Convolution(
        data=relu3_1b, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2b")
    relu3_2b = mx.symbol.Activation(data=conv3_2b, act_type="relu", name="relu3_2b")
    pool3b = mx.symbol.Pooling(
        data=relu3_2b, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3b")
    # group 4
    conv4_1b = mx.symbol.Convolution(
        data=pool3b, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1b")
    relu4_1b = mx.symbol.Activation(data=conv4_1b, act_type="relu", name="relu4_1b")
    conv4_2b = mx.symbol.Convolution(
        data=relu4_1b, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2b")
    relu4_2b = mx.symbol.Activation(data=conv4_2b, act_type="relu", name="relu4_2b")
    pool4b = mx.symbol.Pooling(
        data=relu4_2b, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4b")
    # group 5
    conv5_1b = mx.symbol.Convolution(
        data=pool4b, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1b")
    relu5_1b = mx.symbol.Activation(data=conv5_1b, act_type="relu", name="relu5_1b")
    conv5_2b = mx.symbol.Convolution(
        data=relu5_1b, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2b")
    relu5_2b = mx.symbol.Activation(data=conv5_2b, act_type="relu", name="relu5_2b")
    pool5b = mx.symbol.Pooling(
        data=relu5_2b, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool5b")
    normal_b = mx.symbol.L2Normalization(data=pool5b, mode='instance', name="l2normb")

    corr_1 = mxnet.symbol.Correlation(data1=normal_a, data2=normal_b)
    # Need to do normalization at the output of the CorrLayer
    relu6_1 = mx.symbol.Activation(data=corr_1, act_type="relu", name="relu6_1")
    norm_2 = mx.symbol.L2Normalization(data=relu6_1, mode='instance')


    conv6_1 = mx.symbol.Convolution(
        data=norm_2, kernel=(7, 7), pad=(1, 1), num_filter=128, name="conv6_1")
    relu6_1 = mx.symbol.Activation(data=conv6_1, act_type="relu", name="relu6_1")
    pool6 = mx.symbol.Pooling(
        data=relu6_1, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool6")
    batchnorm1 = mx.symbol.BatchNorm(pool6, name="batchnorm1")

    conv7_1 = mx.symbol.Convolution(
        data=batchnorm1, kernel=(5, 5), pad=(1, 1), num_filter=64, name="conv7_1")
    relu7_1 = mx.symbol.Activation(data=conv7_1, act_type="relu", name="relu7_1")
    pool7 = mx.symbol.Pooling(
        data=relu7_1, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool7")
    batchnorm2 = mx.symbol.BatchNorm(pool7, name="batchnorm2")

    fc1 = mx.symbol.FullyConnected(data=batchnorm2, num_hidden=6, name="fc1")
    relu8 = mx.symbol.Activation(data=fc1, act_type="linear", name="relu8")
    trans = mx.symbol.Dropout(data=relu8, p=0.5, name="drop8")

    return trans
