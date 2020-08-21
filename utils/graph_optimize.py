# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# bought from https://github.com/TuSimple/simpledet/blob/master/utils/graph_optimize.py

import json
import logging
import mxnet as mx

FLOAT32_DTYPE = 0
INIT_ZERO = '[\"zero\", {}]'
MINMAX_SUFFIX = "_minmax"

def convert_class_to_dict(obj):
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not callable(value):
            pr[name] = str(value)
    return pr

def merge_bn(symbol, args, auxs, symbol_only=False):
    """
    Adapted from https://github.com/dmlc/tvm/blob/master/python/tvm/relay/frontend/mxnet.py
    Instead of translating nnvm graph into TVM relay graph, we adapt the script to translate
    it back to mxnet graph.
    """
    assert symbol is not None
    jgraph = json.loads(symbol.tojson())
    jnodes = jgraph["nodes"]
    node_map = {}
    node_op_map = {}

    for nid, node in enumerate(jnodes):
        # edges are [which_node, which_output, type(? not sure)]
        # mx.symbol has an attribute of __getitem__. sym[1] gives the second output
        children = [node_map[e[0]][e[1]] for e in node["inputs"]]
        attrs = node.get("attrs", {})
        node_name = node["name"]
        op_name = node["op"]
        if op_name == "null":
            print(node_name)
            attrs = dict({k:v for k, v in attrs.items() if k.startswith("__")})
            node_map[nid] = mx.sym.var(node_name, **attrs)
            node_op_map[nid] = ["Variable"]
        elif op_name == "BatchNorm":
            e = node["inputs"][0]
            _, gamma, beta, mmean, mvar = children
            gamma_name, beta_name, mmean_name, mvar_name = gamma.name, beta.name, mmean.name, mvar.name
            assert "gamma" in gamma_name
            assert "beta" in beta_name
            assert "moving_mean" in mmean_name or "running_mean" in mmean_name
            assert "moving_var" in mvar_name or "running_var" in mvar_name
            eps = float(attrs["eps"])
            if attrs["use_global_stats"] == "True" and node_op_map[e[0]][e[1]] in["Convolution", "DeformableConvolution"]:
                logging.info("Merging {}".format(node_name))
                # modify beta before gamma since gamma is not depend on beta
                args[beta_name] -= args[gamma_name] * auxs[mmean_name] / mx.nd.sqrt(eps + auxs[mvar_name])
                args[gamma_name] /= mx.nd.sqrt(eps + auxs[mvar_name])
                # expand for broadcasting
                if args[gamma_name].ndim == 1:
                    args[gamma_name] = args[gamma_name].expand_dims(axis=0).expand_dims(axis=-1).expand_dims(axis=-1)
                    args[beta_name] = args[beta_name].expand_dims(axis=0).expand_dims(axis=-1).expand_dims(axis=-1)
                args["fused_" + node_name + "_gamma"] = args[gamma_name]
                args["fused_" + node_name + "_beta"] = args[beta_name]

                # BroadcastScale is needed
                gamma = mx.sym.var("fused_" + node_name + "_gamma", shape=args[node_name.replace('_fwd', "") + "_gamma"].shape)
                beta = mx.sym.var("fused_" + node_name + "_beta", shape=args[node_name.replace('_fwd', "") + "_beta"].shape)
                res = mx.sym.broadcast_add(mx.sym.broadcast_mul(children[0], gamma), beta)

                # delete mmean and mvar to identity to avoid fusing more than once in weight sharing
                del auxs[mmean_name]
                del auxs[mvar_name]
                del args[gamma_name]
                del args[beta_name]

            else:
                assert False
                res = mx.sym.BatchNorm(*children, **attrs, name=node_name)
            node_map[nid] = res
            node_op_map[nid] = ["XXXX"]
        else:
            if op_name.startswith("_contrib_"):
                op_name = op_name.replace("_contrib_", "")
                operator = eval("mx.sym.contrib." + op_name)
            elif op_name.startswith("_"):
                operator = eval("mx.sym._internal." + op_name)
            else:
                operator = eval("mx.sym." + op_name)
            res = operator(*children, **attrs, name=node_name)
            node_map[nid] = res
            node_op_map[nid] = [op_name]

    outputs = [node_map[e[0]][e[1]] for e in jgraph["heads"]]
    outputs = outputs[0] if len(outputs) == 1 else mx.sym.Group(outputs)

    return outputs, args, auxs


def merge_gluon_hybrid_block_bn(net, data_shape=(1, 3, 368, 368)):
    logging.info("Trying to replace batchnorm by scaling and adding.")
    params = net.collect_params()
    params_auxs = {}
    params_args = {}
    for k in params.keys():
        if params[k]._data is not None:
            if "moving" in k or "running" in k:
                params_auxs[k] = params[k].data()
            else:
                params_args[k] = params[k].data()

    var_data = mx.symbol.var(name="data", shape=data_shape)
    sym = net(var_data)
    if isinstance(sym, list):
        sym = mx.sym.Group(sym)
    sym, args, auxs = merge_bn(sym, params_args, params_auxs)
    net = mx.gluon.SymbolBlock(sym, var_data)
    params_merged = net.collect_params()
    for k in params_merged:
        if k in args:
            params_merged[k]._load_init(args[k], ctx=mx.cpu())
            logging.info("loaded {} from merged bn.".format(k))
        else:
            print("Escaping {} when loading".format(k))

    logging.info("Merging batchnorm finished.")
    return net


if __name__ == "__main__":
    sym = mx.sym.load("source.json")
    sym1, _, _ = merge_bn(sym, None, None, True)