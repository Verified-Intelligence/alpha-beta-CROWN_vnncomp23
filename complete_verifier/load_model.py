import os
import collections
import gzip
from ast import literal_eval
import torch
import numpy as np

import onnx2pytorch
import onnx
import onnxruntime as ort
import onnxoptimizer
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from onnx_opt import compress_onnx

import warnings
import importlib
from functools import partial

import arguments

from model_defs import *
from utils import expand_path


def Customized(def_file, callable_name, *args, **kwargs):
    """Fully customized model or dataloader."""
    if def_file.endswith('.py'):
        # Use relatively path w.r.t. to the configuration file
        if arguments.Config['general']['root_path']:
            path = os.path.join(arguments.Config['general']['root_path'], def_file)
        elif arguments.Config.file:
            path = os.path.join(os.path.dirname(arguments.Config.file), def_file)
        else:
            path = def_file
        spec = importlib.util.spec_from_file_location('customized', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(f'custom.{def_file}')
    # Load model from a specified file.
    model_func = getattr(module, callable_name)
    customized_func = partial(model_func, *args, **kwargs)
    # We need to return a Callable which returns the model.
    return customized_func


def deep_update(d, u):
    """Update a dictionary based another dictionary, recursively.

    (https://stackoverflow.com/a/3233356).
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def unzip_and_optimize_onnx(path, onnx_optimization_flags='none'):
    if onnx_optimization_flags == 'none':
        if path.endswith('.gz'):
            onnx_model = onnx.load(gzip.GzipFile(path))
        else:
            onnx_model = onnx.load(path)
        return onnx_model
    else:
        print(f'Onnx optimization with flag: {onnx_optimization_flags}')
        npath = path + '.optimized'
        if os.path.exists(npath):
            print(f'Found existed optimized onnx model at {npath}')
            return onnx.load(npath)
        else:
            print(f'Generate optimized onnx model to {npath}')
            if path.endswith('.gz'):
                onnx_model = onnx.load(gzip.GzipFile(path))
            else:
                onnx_model = onnx.load(path)
            return compress_onnx(onnx_model, path, npath, onnx_optimization_flags, debug=True)


def inference_onnx(path, input):
    # Workaround for onnx bug, see issue #150
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    sess = ort.InferenceSession(unzip_and_optimize_onnx(path).SerializeToString(),
                                sess_options=options)
    assert len(sess.get_inputs()) == len(sess.get_outputs()) == 1
    res = sess.run(None, {sess.get_inputs()[0].name: input})[0]
    return res


@torch.no_grad()
def load_model_onnx(path, quirks=None, x=None):
    onnx_optimization_flags = arguments.Config['model']['onnx_optimization_flags']
    if arguments.Config['model']['cache_onnx_conversion']:
        path_cache = f'{path}.cache'
        if os.path.exists(path_cache):
            print(f'Loading converted model from {path_cache}')
            return torch.load(path_cache)
    quirks = {} if quirks is None else quirks
    if arguments.Config['model']['onnx_quirks']:
        try:
            config_quirks = literal_eval(arguments.Config['model']['onnx_quirks'])
        except ValueError:
            print('ERROR: onnx_quirks '
                  f'{arguments.Config["model"]["onnx_quirks"]}'
                  'cannot be parsed!')
            raise
        assert isinstance(config_quirks, dict)
        deep_update(quirks, config_quirks)
    print(f'Loading onnx {path} wih quirks {quirks}')

    onnx_model = unzip_and_optimize_onnx(path, onnx_optimization_flags)

    if arguments.Config["model"]["input_shape"] is None:
        # find the input shape from onnx_model generally
        # https://github.com/onnx/onnx/issues/2657
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [node.name for node in onnx_model.graph.initializer]
        net_feed_input = list(set(input_all) - set(input_initializer))
        net_feed_input = [node for node in onnx_model.graph.input if node.name in net_feed_input]

        if len(net_feed_input) != 1:
            # in some rare case, we use the following way to find input shape
            # but this is not always true (collins-rul-cnn)
            net_feed_input = [onnx_model.graph.input[0]]

        onnx_input_dims = net_feed_input[0].type.tensor_type.shape.dim
        onnx_shape = tuple(d.dim_value for d in onnx_input_dims)
    else:
        # User specify input_shape
        onnx_shape = arguments.Config['model']['input_shape']

    # remove batch information
    # for nn4sys pensieve parallel, the first dimension of the input size is not batch, do not remove
    if onnx_shape[0] <= 1:
        onnx_shape = onnx_shape[1:]

    pytorch_model = onnx2pytorch.ConvertModel(
        onnx_model, experimental=True, quirks=quirks)
    pytorch_model.eval()
    pytorch_model.to(dtype=torch.get_default_dtype())

    conversion_check_result = True
    try:
        # check conversion correctness
        # FIXME dtype of dummy may not match the onnx model, which can cause runtime error
        if x is not None:
            x = x.reshape(1, *onnx_shape)
        else:
            x = torch.randn([1, *onnx_shape])
        output_pytorch = pytorch_model(x).numpy()
        try:
            if arguments.Config['model']['check_optmized']:
                output_onnx = inference_onnx(path+'.optimized', x.numpy())
            else:  
                output_onnx = inference_onnx(path, x.numpy())
        except ort.capi.onnxruntime_pybind11_state.InvalidArgument:
            # ONNX model might have shape problems. Remove the batch dimension and try again.
            output_onnx = inference_onnx(path, x.numpy().squeeze(0))
        if 'remove_relu_in_last_layer' in onnx_optimization_flags:
            output_pytorch = output_pytorch.clip(min=0)
        conversion_check_result = np.allclose(
            output_pytorch, output_onnx, 1e-4, 1e-5)
    except:  # pylint: disable=broad-except
        warnings.warn('Not able to check model\'s conversion correctness')
        print('\n*************Error traceback*************')
        import traceback; print(traceback.format_exc())
        print('*****************************************\n')
    if not conversion_check_result:
        print('\n**************************')
        print('Model might not be converted correctly. Please check onnx conversion carefully.')
        print('Output by pytorch:', output_pytorch)
        print('Output by onnx:', output_onnx)
        print('Max error:', torch.tensor(output_pytorch - output_onnx).abs().max())
        print('**************************\n')
        if arguments.Config["model"]["debug_onnx"]:
            debug_onnx(onnx_model, pytorch_model, x.numpy())

    if arguments.Config["model"]["cache_onnx_conversion"]:
        torch.save((pytorch_model, onnx_shape), path_cache)

    # TODO merge into the unzip_and_optimize_onnx()
    if arguments.Config["model"]["flatten_final_output"]:
        pytorch_model = nn.Sequential(pytorch_model, nn.Flatten())

    return pytorch_model, onnx_shape


def debug_onnx(onnx_model, pytorch_model, dummy_input):
    path_tmp = '/tmp/debug.onnx'

    output_onnx = {}
    for node in enumerate_model_node_outputs(onnx_model):
        print('Inferencing onnx node:', node)
        save_onnx_model(select_model_inputs_outputs(onnx_model, node), path_tmp)
        optimized_model = onnxoptimizer.optimize(
            onnx.load(path_tmp),
            ["extract_constant_to_initializer",
             "eliminate_unused_initializer"])
        sess = ort.InferenceSession(optimized_model.SerializeToString())
        output_onnx[node] = torch.tensor(sess.run(
            None, {sess.get_inputs()[0].name: dummy_input})[0])

    print('Inferencing the pytorch model')
    output_pytorch = pytorch_model(
        torch.tensor(dummy_input), return_all_nodes=True)

    for k in output_pytorch:
        if k == sess.get_inputs()[0].name:
            continue
        print(k, output_onnx[k].shape)
        close = torch.allclose(output_onnx[k], output_pytorch[k])
        print('  close?', close)
        if not close:
            print('  max error', (output_onnx[k] - output_pytorch[k]).abs().max())

    import pdb; pdb.set_trace()


def load_model(weights_loaded=True):
    """
    Load the model architectures and weights
    """

    assert arguments.Config["model"]["name"] is None or arguments.Config["model"]["onnx_path"] is None, (
        "Conflict detected! User should specify model path by either --model or --onnx_path! "
        "The cannot be both specified.")

    assert arguments.Config["model"]["name"] is not None or arguments.Config["model"]["onnx_path"] is not None, (
        "No model is loaded, please set --model <modelname> for pytorch model or --onnx_path <filename> for onnx model.")

    if arguments.Config['model']['name'] is not None:
        # You can customize this function to load your own model based on model name.
        try:
            model_ori = eval(arguments.Config['model']['name'])()  # pylint: disable=eval-used
        except Exception:  # pylint: disable=broad-except
            print(f'Cannot load pytorch model definition "{arguments.Config["model"]["name"]}()". '
                  f'"{arguments.Config["model"]["name"]}()" must be a callable that returns a torch.nn.Module object.')
            import traceback
            traceback.print_exc()
            exit()
        model_ori.eval()

        if not weights_loaded:
            return model_ori

        if arguments.Config["model"]["path"] is not None:
            # Load pytorch model
            # You can customize this function to load your own model based on model name.
            sd = torch.load(expand_path(arguments.Config["model"]["path"]),
                            map_location=torch.device('cpu'))
            if 'state_dict' in sd:
                sd = sd['state_dict']
            if isinstance(sd, list):
                sd = sd[0]
            if not isinstance(sd, dict):
                raise NotImplementedError("Unknown model format, please modify model loader yourself.")
            try:
                model_ori.load_state_dict(sd)
            except RuntimeError:
                print('Failed to load the model')
                print('Keys in the state_dict of model_ori:')
                print(list(model_ori.state_dict().keys()))
                print('Keys in the state_dict trying to load:')
                print(list(sd.keys()))
                raise

    elif arguments.Config["model"]["onnx_path"] is not None:
        # Load onnx model
        model_ori, _ = load_model_onnx(expand_path(
            arguments.Config["model"]["onnx_path"]))

    else:
        print("Warning: pretrained model path is not given!")

    print(model_ori)
    print('Parameters:')
    for p in model_ori.named_parameters():
        print(f'  {p[0]}: shape {p[1].shape}')

    return model_ori
