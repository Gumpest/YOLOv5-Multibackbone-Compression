import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import yaml

from models.yolo import Detect
from models.common import *
from models.experimental import *
from utils.general import make_divisible


def parse_module_defs(d):
    CBL_idx = []
    ignore_idx  =[]

    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    fromlayer = []  # last module bn layer name
    from_to_map = {}

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        named_m_base = "model.{}".format(i)
        if m is Conv:
            named_m_bn = named_m_base+'.bn'
            CBL_idx.append(named_m_bn)
            if i > 0:
                from_to_map[named_m_bn] = fromlayer[f]
            fromlayer.append(named_m_bn)
        elif m is C3:
            named_m_cv1_bn = named_m_base + ".cv1.bn"
            named_m_cv2_bn = named_m_base + ".cv2.bn"
            named_m_cv3_bn = named_m_base + ".cv3.bn"
            from_to_map[named_m_cv1_bn] = fromlayer[f]
            from_to_map[named_m_cv2_bn] = fromlayer[f]
            fromlayer.append(named_m_cv3_bn)
            c3fromlayer = [named_m_cv1_bn]

            # CBL_idx.append(named_m_cv1_bn)
            ignore_idx.append(named_m_cv1_bn)
            CBL_idx.append(named_m_cv2_bn)
            CBL_idx.append(named_m_cv3_bn)
            for j in range(n):
                named_m_bottle_cv1_bn = named_m_base + ".m.{}.cv1.bn".format(j)
                named_m_bottle_cv2_bn = named_m_base + ".m.{}.cv2.bn".format(j)
                from_to_map[named_m_bottle_cv1_bn] = c3fromlayer[j]
                from_to_map[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn
                c3fromlayer.append(named_m_bottle_cv2_bn)
                CBL_idx.append(named_m_bottle_cv1_bn)
                # ignore_idx.append(named_m_bottle_cv1_bn)
                ignore_idx.append(named_m_bottle_cv2_bn)  # not prune shortcut
            from_to_map[named_m_cv3_bn] = [c3fromlayer[-1], named_m_cv2_bn]
        elif m is Focus:
            named_m_bn = named_m_base+'.conv.bn'
            CBL_idx.append(named_m_bn)
            fromlayer.append(named_m_bn)
        elif m is SPP:
            named_m_cv1_bn = named_m_base+'.cv1.bn'
            named_m_cv2_bn = named_m_base+'.cv2.bn'
            CBL_idx.append(named_m_cv1_bn)
            ignore_idx.append(named_m_cv2_bn)
            from_to_map[named_m_cv1_bn] = fromlayer[f]
            from_to_map[named_m_cv2_bn] = [named_m_cv1_bn]*4
            fromlayer.append(named_m_cv2_bn)
        elif m is SPPF:
            named_m_cv1_bn = named_m_base+'.cv1.bn'
            named_m_cv2_bn = named_m_base+'.cv2.bn'
            CBL_idx.append(named_m_cv1_bn)
            ignore_idx.append(named_m_cv2_bn)
            from_to_map[named_m_cv1_bn] = fromlayer[f]
            from_to_map[named_m_cv2_bn] = [named_m_cv1_bn]*4
            fromlayer.append(named_m_cv2_bn)
        elif m is Concat:
            inputtmp = [fromlayer[x] for x in f]
            fromlayer.append(inputtmp)
        elif m is Detect:
            for j in range(3):
                ignore_idx.append(named_m_base + ".m.{}".format(j))
                from_to_map[named_m_base + ".m.{}".format(j)] = fromlayer[f[j]]
        else:
            fromlayer.append(fromlayer[-1])

    return CBL_idx, ignore_idx, from_to_map

def obtain_filtermask_l1(conv_module, rand_remain_ratio):
    w_copy = conv_module.weight.data.abs().clone()
    w_copy = torch.sum(w_copy, dim=(1,2,3))
    length = w_copy.size()[0]
    num_retain = int(length*rand_remain_ratio)
    if num_retain<2:
        num_retain=2
    _,y = torch.topk(w_copy,num_retain)
    mask = torch.zeros(length)
    mask[y.cpu()] = 1

    return mask

def weights_inheritance(model, compact_model, from_to_map, maskbndict):
    modelstate = model.state_dict()
    pruned_model_state = compact_model.state_dict()
    assert pruned_model_state.keys() == modelstate.keys()
    for ((layername, layer),(pruned_layername, pruned_layer)) in zip(model.named_modules(), compact_model.named_modules()):
        assert layername == pruned_layername
        if isinstance(layer, nn.Conv2d) and not layername.startswith("model.24"):
            convname = layername[:-4] + "bn"
            if convname in from_to_map.keys():
                former = from_to_map[convname]
                if isinstance(former, str):
                    out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                    in_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[former].cpu().numpy())))
                    w = layer.weight.data[:, in_idx, :, :].clone()
                    w = w[out_idx, :, :, :].clone()
                    if len(w.shape) ==3:
                        w = w.unsqueeze(0)
                    pruned_layer.weight.data = w
                if isinstance(former, list):
                    orignin = [modelstate[i+".weight"].shape[0] for i in former]
                    formerin = []
                    for it in range(len(former)):
                        name = former[it]
                        tmp = [i for i in range(maskbndict[name].shape[0]) if maskbndict[name][i] == 1]
                        if it > 0:
                            tmp = [k + sum(orignin[:it]) for k in tmp]
                        formerin.extend(tmp)
                    out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                    w = layer.weight.data[out_idx, :, :, :].clone()
                    pruned_layer.weight.data = w[:,formerin, :, :]
            else:
                out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                w = layer.weight.data[out_idx, :, :, :].clone()
                assert len(w.shape) == 4
                pruned_layer.weight.data = w

        if isinstance(layer,nn.BatchNorm2d):
            out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername].cpu().numpy())))
            pruned_layer.weight.data = layer.weight.data[out_idx].clone()
            pruned_layer.bias.data = layer.bias.data[out_idx].clone()
            pruned_layer.running_mean = layer.running_mean[out_idx].clone()
            pruned_layer.running_var = layer.running_var[out_idx].clone()

        if isinstance(layer, nn.Conv2d) and layername.startswith("model.24"):
            former = from_to_map[layername]
            in_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[former].cpu().numpy())))
            pruned_layer.weight.data = layer.weight.data[:, in_idx, :, :]
            pruned_layer.bias.data = layer.bias.data


def update_yaml_loop(d, name, maskconvdict):
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    ch = [3]
    c2 = ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m  # eval strings
        # for j, a in enumerate(args):
        #     try:
        #         args[j] = eval(a) if isinstance(a, str) else a  # eval strings
        #     except:
        #         pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        named_m_base = "model.{}".format(i)
        if m is Conv:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                if isinstance(args[-1],float):
                    c2 = c2 * args[-1]
                c2 = make_divisible(c2 * gw, 8)
            named_m_conv = named_m_base+'.conv'
            if name == named_m_conv:
                args[-1] = maskconvdict[named_m_conv].sum().item() / c2
        elif m is Focus:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                if isinstance(args[-1],float):
                    c2 = c2 * args[-1]
                c2 = make_divisible(c2 * gw, 8)
            named_m_conv = named_m_base+'.conv.conv'
            if name == named_m_conv:
                args[-1] = maskconvdict[named_m_conv].sum().item() / c2
        elif m is SPP or m is SPPF:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                if isinstance(args[-1],float):
                    c2 = c2 * args[-1]
                c2 = make_divisible(c2 * gw, 8)
            named_m_cv1_conv = named_m_base+'.cv1.conv'
            if name == named_m_cv1_conv:
                args[-1] = 0.5 * maskconvdict[named_m_cv1_conv].sum().item() / c2
        elif m is C3:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2_ = make_divisible(c2 * gw, 8)
                if isinstance(args[-1],float):
                    c2 = c2 * args[-1]
                c2 = make_divisible(c2 * gw, 8)
            named_m_cv1_conv = named_m_base + ".cv1.conv"
            named_m_cv2_conv = named_m_base + ".cv2.conv"
            named_m_cv3_conv = named_m_base + ".cv3.conv"
            if name == named_m_cv1_conv:
                # args[-3][0] = 0.5 * maskconvdict[named_m_cv1_conv].sum().item() / (c2*0.5)
                continue
            elif name == named_m_cv2_conv:
                args[-3][1] = 0.5 * maskconvdict[named_m_cv2_conv].sum().item() / (c2*0.5)
                continue
            elif name == named_m_cv3_conv:
                args[-1] = maskconvdict[named_m_cv3_conv].sum().item() / c2
                continue
            for j in range(n):
                named_m_bottle_cv1_conv = named_m_base + ".m.{}.cv1.conv".format(j)
                if name == named_m_bottle_cv1_conv:
                    args[-2][j] = maskconvdict[named_m_bottle_cv1_conv].sum().item() / (c2_*0.5)
                    continue


def update_yaml(pruned_yaml, model, ignore_conv_idx, maskdict, opt):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if name not in ignore_conv_idx:
                update_yaml_loop(pruned_yaml,name,maskdict)
    
    return pruned_yaml