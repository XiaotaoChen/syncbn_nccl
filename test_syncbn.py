import ctypes
_=ctypes.CDLL("/mnt/truenas/scratch/xiaotao.chen/Repositories/mx_ops/additional.so")
import mxnet as mx
import numpy as np
import pickle as pkl
from numpy.testing import assert_allclose
from mxnet.test_utils import assert_almost_equal

np.random.seed(5)
batch_size = 16

data_shape = (batch_size, 2,4,4)
data = np.random.uniform(size=data_shape)
for i in range(batch_size):
    data[i] += i
label_shape = (batch_size,)
label = np.ones(shape=label_shape)


np_gamma = np.ones((2,))
np_beta = np.zeros((2,))
# gamma = np.ones(shape=(1,2,1,1))
# beta = np.zeros(shape=(1,2,1,1))
eps=1e-5


data_names = ['data']
label_names = ["softmax_label"]
mx_data_shape = [('data', data_shape)]
mx_label_shape = [("softmax_label", label_shape)]

mx_data_batch = mx.io.DataBatch(data=[mx.nd.array(data)], provide_data=mx_data_shape, label=[mx.nd.array(label)], provide_label=mx_label_shape)

def get_all_internals(sym):
    internal_syms = sym.get_internals()
    internal_sym_names = internal_syms.list_outputs()
    all_output_names = []
    all_outputs = []
    for internal_name in internal_sym_names:
        if internal_name.endswith("output"):
            all_output_names.append(internal_name)
            all_outputs.append(internal_syms[internal_name])
    return all_outputs, all_output_names

def get_infer_shape(sym, data_shape=(1, 3, 224, 224)):
    arg_shape, _, aux_shape = sym.infer_shape(data=data_shape)
    _, out_shape, _ = sym.get_internals().infer_shape(data=data_shape)
    return arg_shape, out_shape, aux_shape

def load_checkpoint(prefix, epoch):
    print('load %s-%04d.params' % (prefix, epoch))
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params

def bn_sym():
    data = mx.sym.Variable(name="data")
    bn = mx.sym.BatchNorm_v1(name="bn1", data=data, eps=eps, fix_gamma=False)
    bn = mx.sym.BatchNorm_v1(name="bn2", data=bn, eps=eps, fix_gamma=False)
    flat = mx.sym.Flatten(data=bn, name="flatten1")
    sym = mx.sym.SoftmaxOutput(data=flat, name="softmax")

    internal_syms = sym.get_internals()
    # 'bn1_gamma', 'bn1_beta', 'bn1_moving_mean', 'bn1_moving_var', 'bn1_output'
    bn_out = internal_syms["bn1_output"]
    bn2_out = internal_syms["bn2_output"]
    return mx.sym.Group([bn_out, bn2_out, sym])

def syncbn_sym(ndev=1):
    data = mx.sym.Variable(name="data")
    # bn = mx.sym.contrib.SyncBatchNorm(name="bn1", data=data, eps=eps, fix_gamma=False, key="syncbn1", ndev=ndev)
    # bn = mx.sym.contrib.SyncBatchNorm(name="bn2", data=bn, eps=eps, fix_gamma=False, key="syncbn2", ndev=ndev)
    bn = mx.sym.contrib.SyncBatchNormV3(name="bn1", data=data, eps=eps, fix_gamma=False, key="syncbn1", ndev=ndev)
    bn = mx.sym.contrib.SyncBatchNormV3(name="bn2", data=bn, eps=eps, fix_gamma=False, key="syncbn2", ndev=ndev)
    flat = mx.sym.Flatten(data=bn, name="flatten1")
    sym = mx.sym.SoftmaxOutput(data=flat, name="softmax")

    internal_syms = sym.get_internals()
    # 'bn1_gamma', 'bn1_beta', 'bn1_moving_mean', 'bn1_moving_var', 'bn1_output'
    bn_out = internal_syms["bn1_output"]
    bn2_out = internal_syms["bn2_output"]
    return mx.sym.Group([bn_out, bn2_out, sym])


def test_symbol(bnop, ndev=1, bn_type="local", debug=False):
    input_data = mx.sym.Variable(name="data")
    conv1 = mx.sym.Convolution(data=input_data, num_filter=16, kernel=(3, 3), stride=(1, 1),
                                   pad=(1, 1), no_bias=True, name = 'conv1')
    if bn_type=="local":   
        bn1 = bnop(data=conv1, fix_gamma=False, eps=1e-5, momentum=0.9, name='bn1')
    else:
        bn1 = bnop(data=conv1, fix_gamma=False, eps=1e-5, momentum=0.9, name='bn1', key="bn1", ndev=ndev, debug=debug)
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')

    conv2 = mx.sym.Convolution(data=act1, num_filter=8, kernel=(3, 3), stride=(1, 1),
                                   pad=(1, 1), no_bias=True, name = 'conv2')
    if bn_type == "local":
        bn2 = bnop(data=conv2, fix_gamma=False, eps=1e-5, momentum=0.9, name='bn2')
    else:
        bn2 = bnop(data=conv2, fix_gamma=False, eps=1e-5, momentum=0.9, name='bn2', key="bn2", ndev=ndev, debug=debug)
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name='relu2')

    flatten = mx.sym.Flatten(data=act2)

    fc1 = mx.sym.FullyConnected(name='fc1', data=flatten, num_hidden=10, no_bias=1)
    softmax = mx.sym.SoftmaxOutput(data=fc1, name='softmax')
    return softmax
    # internal_syms = softmax.get_internals()
    # bn1_out = internal_syms["bn1_output"]
    # bn2_out = internal_syms["bn2_output"]
    
    # _, out_shape, _ = internal_syms.infer_shape(data=data_shape)
    # out_shape_dict = list(zip(internal_syms.list_outputs(), out_shape))
    # print(out_shape_dict)

    # return mx.sym.Group([bn1_out, bn2_out, softmax])


def mx_bn(data):
    ndev = 8
    ctx = [mx.gpu(i) for i in range(ndev)]
    # sym = syncbn_sym(ndev)
    # ctx = mx.gpu()
    # sym = bn_sym()
    bn_type = "sync"
    debug = False
    if bn_type == "local":
        bnop = mx.sym.BatchNorm
        # bnop = mx.sym.BatchNorm_v1
    else:
        # bnop = mx.sym.contrib.SyncBatchNorm
        bnop = mx.sym.contrib.SyncBatchNormV3
    sym = test_symbol(bnop=bnop, ndev=ndev, bn_type=bn_type, debug=debug)

    mod = mx.mod.Module(symbol=sym, context=ctx, data_names=data_names)
    mod.bind(for_training=True, data_shapes=mx_data_shape, label_shapes=mx_label_shape)

    # mod.init_params()
    arg_params, aux_params = load_checkpoint(prefix="tmp_file/testsym", epoch=0)
    mod.init_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True)

    # arg_params, aux_params = mod.get_params()
    # mx.model.save_checkpoint("bnsym", 0, sym, arg_params, aux_params)
    print("bn op:{}, ndev:{}".format(bnop, ndev))
    mod.init_optimizer()
    for idx in range(10):
        print("forward idx:{}".format(idx))
        mod.forward(mx_data_batch)
        mod.backward()
        mod.update()
        mx.nd.waitall()
        outputs = mod.get_outputs()
        out = outputs[0].asnumpy()
        print("forward idx:{}, output:{}".format(idx, out[0]))
    # mx.nd.waitall()
    # arg_params, aux_params = mod.get_params()
    # mx.model.save_checkpoint("testsym_syncbnv28d_10iter", 0, sym, arg_params, aux_params)

    # bn_out = outputs[0].asnumpy()
    # print(bn_out[0,0])
    # bn2_out = outputs[1].asnumpy()
    # print(bn2_out[1,0])
    # with open("tmp_file/testsym_syncbn_v2_8dev.pkl", "wb") as f:
    #     pkl.dump(bn_out, f)

def check_correct():
    # test.pkl, np_bn_output.pkl, bn_output.pkl, syncbn_ndev_1_output.pkl, syncbn_ndev_8_output.pkl
    src_path = "tmp_file/testsym_syncbn.pkl"
    dst_path = "tmp_file/testsym_syncbn_v2_8dev.pkl"
    with open(src_path, 'rb') as f:
        src_data = pkl.load(f)
    with open(dst_path, 'rb') as f:
        dst_data = pkl.load(f)

    print("check correnctness shape:{}  {} vs {}".format(src_data.shape, src_path, dst_path))
    np.testing.assert_almost_equal(src_data, dst_data, decimal=5)

if __name__ == "__main__":
    mx_bn(data)
    # check_correct()
