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
    CBL_idx = []  # Conv + BN + LeakyReLU
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
    w_copy = torch.sum(w_copy, dim=(1,2,3))  # each kernel owns one
    length = w_copy.size()[0]
    # num_retain = int(length*rand_remain_ratio)
    num_retain = max(make_divisible(int(length*rand_remain_ratio), 8), 8)
    _, indice = torch.topk(w_copy, num_retain)  # tensor([6, 1, 3, 2], device='cuda:0')
    mask = torch.zeros(length)
    mask[indice.cpu()] = 1

    return mask

def obtain_filtermask_bn(bn_module, thresh):
    w_copy = bn_module.weight.data.abs().clone()

    num_retain_init = int(sum(w_copy.gt(thresh).float()))

    length = w_copy.shape[0]
    num_retain = max(make_divisible(num_retain_init, 8), 8)
    _, index = torch.topk(w_copy, num_retain)
    mask = torch.zeros(length)
    mask[index.cpu()] = 1

    return mask

def weights_inheritance(model, compact_model, from_to_map, maskbndict):
    modelstate = model.state_dict()
    pruned_model_state = compact_model.state_dict()
    assert pruned_model_state.keys() == modelstate.keys()
    
    last_idx = 0  # find last layer index
    for (layername, layer) in model.named_modules():
        try:
            last_idx = max(last_idx, int(layername.split('.')[1]))  # 0 in model.0.conv 
        except:
            pass

    for ((layername, layer),(pruned_layername, pruned_layer)) in zip(model.named_modules(), compact_model.named_modules()):
        assert layername == pruned_layername
        if isinstance(layer, nn.Conv2d) and not layername.startswith(f"model.{last_idx}"):  # --------------------------------
            convname = layername[:-4] + "bn"

            # Clone in_idx and out_idx changed layer
            if convname in from_to_map.keys():
                former = from_to_map[convname]
                if isinstance(former, str):
                    out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                    in_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[former].cpu().numpy())))  # [1 2 3 6]
                    w = layer.weight.data[:, in_idx, :, :].clone()
                    w = w[out_idx, :, :, :].clone()
                    if len(w.shape) ==3:
                        w = w.unsqueeze(0)
                    pruned_layer.weight.data = w
                if isinstance(former, list):
                # ['model.2.m.0.cv2.bn', 'model.2.cv2.bn']
                    out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))

                    former_kernel_num = [modelstate[former_item + ".weight"].shape[0] for former_item in former] # Compute former_kernel_num for accumlate index
                    in_idx = []
                    for i in range(len(former)):
                        former_item = former[i]
                        in_idx_each = np.squeeze(np.argwhere(np.asarray(maskbndict[former_item].cpu().numpy())))

                        if i > 0:
                            in_idx_each = [k + sum(former_kernel_num[:i]) for k in in_idx_each]
                        in_idx.extend(in_idx_each)
   
                    w = layer.weight.data[:, in_idx, :, :].clone()
                    w = w[out_idx, :, :, :].clone()
                    if len(w.shape) ==3:
                        w = w.unsqueeze(0)
                    pruned_layer.weight.data = w

            # Clone out_idx changed layer
            else:
                out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                w = layer.weight.data[out_idx, :, :, :].clone()
                if len(w.shape) == 3:
                    w = w.unsqueeze(0)
                pruned_layer.weight.data = w

        # Clone in_idx changed layer
        if isinstance(layer, nn.Conv2d) and layername.startswith(f"model.{last_idx}"):  # --------------------------------
            former = from_to_map[layername]
            in_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[former].cpu().numpy())))
            pruned_layer.weight.data = layer.weight.data[:, in_idx, :, :].clone()
            pruned_layer.bias.data = layer.bias.data.clone()

        # Clone BatchNorm2d Layer
        if isinstance(layer, nn.BatchNorm2d):
            out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername].cpu().numpy())))
            pruned_layer.weight.data = layer.weight.data[out_idx].clone()
            pruned_layer.bias.data = layer.bias.data[out_idx].clone()
            pruned_layer.running_mean = layer.running_mean[out_idx].clone()
            pruned_layer.running_var = layer.running_var[out_idx].clone()

    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
    cm = compact_model.module.model[-1] if hasattr(compact_model, 'module') else compact_model.model[-1]
    cm.anchors = m.anchors.clone()


def update_yaml_loop(d, name, maskconvdict):
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    ch = [3]
    c2 = ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings

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

class BatchNormSparser():

    @staticmethod
    def updateBN(model, sparse_rate, ignore_idx):
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d) and name not in ignore_idx:
                    module.weight.grad.data.add_(sparse_rate * torch.sign(module.weight.data))  # L1

# gather all the bn weights and put them into a line
def gather_bn_weights(model, ignore_idx):

    bn_size_list = []
    bn_module_list = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and name not in ignore_idx:
            bn_size_list.append(module.weight.data.shape[0])  # torch.Size([128]) --> 128
            bn_module_list.append(module)

    # print(bn_size_list)
    bn_weights = torch.zeros(sum(bn_size_list))
    
    index = 0
    for module, size in zip(bn_module_list, bn_size_list):

        # print(module.weight.data.abs().clone().shape[0], size)
        bn_weights[index:(index + size)] = module.weight.data.abs().clone()
        index = index + size

    return bn_weights 