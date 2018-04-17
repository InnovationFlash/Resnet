import collections
import tensorflow as tf
slim = tf.contrib.slim
import tensorflow as tf
import time
import math
from datetime import datetime
tf.reset_default_graph()
# 我们使用collections.namedtuple设计ResNet基本Block模块的named tuple，并用它创建Block的类，
# 但只包含数据结构，不包含具体方法。
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    #创建一个Block，其中包含三大属性；
    #scope:名称
    #bootleneck:残差学习单元
    #args:维护一个列表，其中每一个元素对应一个bottleneck残差学习单元,每个元素都为三元tuple（depth,depth_bottleneck,stride）
    #残差单元包含三个卷积层
    #残差学习单元结构[(1*1/s1,depth_bottleneck),(stride*stride/s2,depth_bottlenck),(1*1/s1,depth)]
    'A named tuple describing a ResNet block'

# 下面定义一个降采样subsample的方法
def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)
    #使用slim库自行进行最大池化操作

# 再定义一个conv2d_same函数创建卷积层。
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    if stride == 1:
        #步长为1的卷积
        return slim.conv2d(inputs, num_outputs, kernel_size,
                           stride=1, padding='SAME', scope=scope)
    else:
        #如果不唯一，显式pad zero,pad总数是kernel数目-1，剩余部分直接在补0后猜词cov2d
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size,
                           stride=stride, padding='VALID', scope=scope)

# 接下来定义堆叠Blocks的函数，参数中的net即为输入，blocks是之前定义的Block的class的列表，
# 而outputs_collections则是用来收集各个end_points的collections。
@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None):
    for block in blocks:
        #逐个block循环
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                #逐个Residual Unit堆叠
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    #残差学习单元命名
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net,
                                        depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name,
                                                   net)
            #将输出net添加到collection中
    return net
    #返回net

# 这里创建ResNet通用的arg_scope,用来定义某些函数的参数默认值。
def resnet_arg_scope(is_training=True,#是否训练
                     weight_decay=0.0001,#权重衰减速率
                     batch_norm_decay=0.997,#BN衰减速率
                     batch_norm_epsilon=1e-5,#BN epsilon默认为1e-5
                     batch_norm_scale=True):#BN scale默认为Ture
                     #事先设置好BN各项参数
    batch_norm_parms = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    with slim.arg_scope(
        #将slim.conv2d默认参数设置好
        [slim.conv2d],
        weights_regularizer = slim.l2_regularizer(weight_decay),
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn = tf.nn.relu,#激活函数为ReLU
        normalizer_fn = slim.batch_norm,
        normalizer_params = batch_norm_parms
    ):
        with slim.arg_scope([slim.batch_norm], **batch_norm_parms):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc

# 接下来定义核心的bottlneck残差学习单元
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections=None, scope=None):
    #outputs_collections:收集end_points的collection
    #scope:unit名称
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        #获取获取输入的最后一个维度：通道数。min_rank限定最少维度数
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        #对输入进行Batch Normalization，并且使用ReLU函数进行于激活Preactivate
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
            #定义shorcut，即直连的x；
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')
            #输入输出通道不同，使用步长为stride的1*1卷积改变通道数，来使得通道数一致
        #三层residual，进行残差定义
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        #1*1尺寸 步长为1，输出通道数为depth_bottleneck
        residual = conv2d_same(residual, depth_bottleneck, 3, stride, scope='conv2')
        #3*3尺寸 步长stride,输出通道数为depth_bottleneck
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None,
                               activation_fn=None, scope='conv3')
        #1*1卷积 步长为1 输出通道数为depth（没有正则项和激活函数）
        output = shortcut + residual
        #相加得到最后的output
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)
        #并且将结果填入collection返回output

# 下面定义生成ResNet V2的主函数，我们只要预先定义好网络的残差学习模块组blocks，它就可以生成对应
# 的完整的ResNet
        
#num_classes:最后输出的类数
#reuse标志代表是否重用
def resnet_v2(inputs, blocks, num_classes=None, global_pool=True,
              include_root_block=True, reuse=None, scope=None):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_point_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense],
                            outputs_collections=end_point_collection):
            #将(slim.con2d\bottleneck\stack_block_dense)三个函数参数outputs_collections默认设为end_points_collection
            net = inputs
            if include_root_block:
                with slim.arg_scope([slim.conv2d], activation_fn=None,
                                    normalizer_fn=None):
                    net = conv2d_same(net,64, 7, stride=2, scope='conv1')
                    #创建ResNet最前面的64输出通道的步长为2的7*7卷积，在紧接着一个步长为2的3*3最大池化
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = stack_blocks_dense(net, blocks)
            #使用stack_blocks_dense组装残差学习模块。
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')
            end_points = slim.utils.convert_collection_to_dict(end_point_collection)
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope='predictions')
            #添加Softmax层输出网络结果
            return net, end_points

# 至此，我们就将ResNet的生成函数定义好了。下面根据几个不同深度的ResNet网络配置，来设计层数
# 分别为50， 101， 152， 200的ResNet。
def resnet_v2_50(inputs, num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_50'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True,
                     reuse=reuse, scope=scope)

def resnet_v2_101(inputs, num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_101'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True,
                     reuse=reuse, scope=scope)

def resnet_v2_152(inputs, num_classes=None, global_pool=True,
                  reuse=None, scope='resnet_v2_152'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True,
                     reuse=reuse, scope=scope)

def resnet_v2_200(inputs, num_classes=None, global_pool=True,
                  reuse=None, scope='resnet_v2_200'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True,
                     reuse=reuse, scope=scope)
    
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %(datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration

    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %(datetime.now(), info_string, num_batches, mn, sd))


batch_size = 32
height, width = 224, 224
inputs = tf.random_uniform((batch_size, height, width, 3))
with slim.arg_scope(resnet_arg_scope(is_training=False)):
    net, end_points = resnet_v2_152(inputs, 1000)
    
init = tf.global_variables_initializer()
with tf.Session() as sess: 
    sess.run(init)
    num_batches = 200
    time_tensorflow_run(sess, net, "Forward")
     
