import paddle
import numpy as np
from copy import deepcopy
import paddle.nn.functional as F

def get_sr_flag(epoch, sr):
    # return epoch >= 5 and sr
    return sr

class BNOptimizer():

    @staticmethod
    def updateBN(sr_flag, module_list, s, prune_idx, epoch, idx2mask=None, opt=None):
        if sr_flag:
            # s = s if epoch <= opt.epochs * 0.5 else s * 0.01
            for idx in prune_idx:
                # Squential(Conv, BN, Lrelu)
                # bn_module = module_list[idx][1]
                # bn_module = module_list[idx][1] if type(
                #     module_list[idx][1]).__name__ == 'BatchNorm2D' else module_list[idx][0]
                bn_module = module_list[idx]['BatchNorm2D']
                bn_module.weight.grad.add_(s * paddle.sign(bn_module.weight))  # L1
            if idx2mask:
                for idx in idx2mask:
                    # bn_module = module_list[idx][1]
                    bn_module = module_list[idx][1] if type(
                        module_list[idx][1]).__name__ == 'BatchNorm2D' else module_list[idx][0]
                    #bn_module.weight.grad.data.add_(0.5 * s * torch.sign(bn_module.weight) * (1 - idx2mask[idx].cuda()))
                    bn_module.weight.grad.sub_(0.99 * s * paddle.sign(bn_module.weight) * idx2mask[idx].cuda())

    @staticmethod
    def updateBN_scaler(sr_flag, module_list, s, prune_idx, epoch,scaler, idx2mask=None, opt=None):
        # print("===> updateBN_scaler", sr_flag, s, prune_idx)

        if use_index:
            size_list = [module_list[idx][1].weight.shape[0] if type(module_list[idx][1]).__name__ == 'BatchNorm2D' else
                         module_list[idx][0].weight.shape[0] for idx in prune_idx]
        else:
            size_list = [module_list[idx]["BatchNorm2D"].weight.shape[0] if type(
                module_list[idx]["BatchNorm2D"]).__name__ == 'BatchNorm2D' else
                         module_list[idx]["Conv2D"].weight.shape[0] for idx in prune_idx]

        if sr_flag:
            # s = s if epoch <= opt.epochs * 0.5 else s * 0.01
            for idx in prune_idx:
                # Squential(Conv, BN, Lrelu)
                # bn_module = module_list[idx][1]
                bn_module = module_list[idx][1] if type(
                    module_list[idx][1]).__name__ == 'BatchNorm2D' else module_list[idx][0]
                # print(bn_module)
                bn_module.weight.grad.add_(scaler.scale(s * paddle.sign(bn_module.weight)))  # L1
            if idx2mask:
                for idx in idx2mask:
                    # bn_module = module_list[idx][1]
                    bn_module = module_list[idx][1] if type(
                        module_list[idx][1]).__name__ == 'BatchNorm2D' else module_list[idx][0]
                    # bn_module.weight.grad.data.add_(0.5 * s * torch.sign(bn_module.weight) * (1 - idx2mask[idx].cuda()))
                    bn_module.weight.grad.sub_(scaler.scale(0.99 * s * paddle.sign(bn_module.weight) * idx2mask[idx].cuda()))

def parse_module_defs(module_defs):

    CBL_idx = []
    Conv_idx = []
    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
            if module_defs[i+1]['type'] == 'maxpool' and module_defs[i+2]['type'] == 'route':
                #spp前一个CBL不剪 区分tiny
                ignore_idx.add(i)
            if module_defs[i+1]['type'] == 'route' and 'groups' in module_defs[i+1]:
                ignore_idx.add(i)
            if module_defs[i+1]['type'] == 'convolutional_nobias':
                ignore_idx.add(i)
            if module_defs[i + 1]['type'] == 'maxpool' and module_defs[i + 2]['type'] == 'maxpool':
                # sppf前一个CBL不剪
                ignore_idx.add(i)
        elif module_def['type'] == 'convolutional_noconv':
            CBL_idx.append(i)
            ignore_idx.add(i)
        elif module_def['type'] == 'shortcut':
            ignore_idx.add(i-1)
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':
                ignore_idx.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                ignore_idx.add(identity_idx - 1)

        elif module_def['type'] == 'upsample':
            #上采样层前的卷积层不裁剪
            ignore_idx.add(i - 1)


    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx


def parse_module_defs2(module_defs):
    CBL_idx = []
    Conv_idx = []
    shortcut_idx = dict()
    shortcut_all = set()
    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
            if module_defs[i + 1]['type'] == 'maxpool' and module_defs[i + 2]['type'] == 'route':
                # spp前一个CBL不剪 区分spp和tiny
                ignore_idx.add(i)
            if module_defs[i + 1]['type'] == 'route' and 'groups' in module_defs[i + 1]:
                ignore_idx.add(i)
            if module_defs[i + 1]['type'] == 'maxpool' and module_defs[i + 2]['type'] == 'maxpool':
                # sppf前一个CBL不剪
                ignore_idx.add(i)

        elif module_def['type'] == 'convolutional_noconv':
            CBL_idx.append(i)

        elif module_def['type'] == 'upsample':
            # 上采样层前的卷积层不裁剪
            ignore_idx.add(i - 1)

        elif module_def['type'] == 'shortcut':
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':

                # ignore_idx.add(identity_idx)
                shortcut_idx[i - 1] = identity_idx
                shortcut_all.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':

                # ignore_idx.add(identity_idx - 1)
                shortcut_idx[i - 1] = identity_idx - 1
                shortcut_all.add(identity_idx - 1)
            shortcut_all.add(i - 1)

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx, shortcut_idx, shortcut_all


def gather_bn_weights(module_list, prune_idx, use_index=True):

    if use_index:
        size_list = [module_list[idx][1].weight.shape[0] if type(module_list[idx][1]).__name__ == 'BatchNorm2D' else module_list[idx][0].weight.shape[0] for idx in prune_idx]
    else:
        size_list = [module_list[idx]["BatchNorm2D"].weight.shape[0] if type(module_list[idx]["BatchNorm2D"]).__name__ == 'BatchNorm2D' else
                     module_list[idx]["Conv2D"].weight.shape[0] for idx in prune_idx]

    bn_weights = paddle.zeros([sum(size_list)])
    index = 0
    for idx, size in zip(prune_idx, size_list):
        if use_index:
            bn_weights[index:(index + size)] = module_list[idx][1].weight.abs().clone() if type(module_list[idx][1]).__name__ == 'BatchNorm2D' else module_list[idx][0].weight.abs().clone()
        else:
            bn_weights[index:(index + size)] = module_list[idx]["BatchNorm2D"].weight.abs().clone() if type(
                module_list[idx]["BatchNorm2D"]).__name__ == 'BatchNorm2D' else module_list[idx]["Conv2D"].weight.abs().clone()

        index += size

    return bn_weights

def write_cfg(cfg_file, module_defs):

    with open(cfg_file, 'w') as f:
        for module_def in module_defs:
            f.write(f"[{module_def['type']}]\n")
            for key, value in module_def.items():
                if key == 'batch_normalize' and value == 0:
                    continue

                if key != 'type':
                    if key == 'anchors':
                        value = ', '.join(','.join(str(int(i)) for i in j) for j in value)
                    f.write(f"{key}={value}\n")
            f.write("\n")
    return cfg_file

def get_input_mask(module_defs, idx, CBLidx2mask):

    if idx == 0:
        return np.ones(3)
    if module_defs[idx - 1]['type'] == 'focus':
        return np.ones(12)
    if module_defs[idx - 1]['type'] == 'convolutional':
        return CBLidx2mask[idx - 1]
    if module_defs[idx - 1]['type'] == 'convolutional_noconv':  #yolov5-v3
        return CBLidx2mask[idx - 1]
    elif module_defs[idx - 1]['type'] == 'shortcut':
        return CBLidx2mask[idx - 2]
    elif module_defs[idx - 1]['type'] == 'route':
        route_in_idxs = []
        for layer_i in module_defs[idx - 1]['layers'].split(","):
            if int(layer_i) < 0:
                route_in_idxs.append(idx - 1 + int(layer_i))
            else:
                route_in_idxs.append(int(layer_i))

        if len(route_in_idxs) == 1:
            if module_defs[route_in_idxs[0]]['type'] == 'route':
                route_in_idxs_tmp = []
                for layer_i in module_defs[route_in_idxs[0]]['layers'].split(","):
                    if int(layer_i) < 0:
                        route_in_idxs_tmp.append(route_in_idxs[0] + int(layer_i))
                    else:
                        route_in_idxs_tmp.append(int(layer_i))
                if module_defs[route_in_idxs_tmp[0]]['type'] == 'upsample':
                    mask1 = CBLidx2mask[route_in_idxs_tmp[0] - 1]
                elif module_defs[route_in_idxs_tmp[0]]['type'] == 'convolutional':
                    mask1 = CBLidx2mask[route_in_idxs_tmp[0]]
                else:
                    assert 0
                if module_defs[route_in_idxs_tmp[1]]['type'] == 'convolutional':
                    mask2 = CBLidx2mask[route_in_idxs_tmp[1]]
                else:
                    assert 0
                return np.concatenate([mask1, mask2])
            mask = CBLidx2mask[route_in_idxs[0]]
            if 'groups' in module_defs[idx - 1]:
                return mask[(mask.shape[0]//2):]
            return mask

        elif len(route_in_idxs) == 2:
            # return np.concatenate([CBLidx2mask[in_idx - 1] for in_idx in route_in_idxs])
            if module_defs[route_in_idxs[0]]['type'] == 'upsample':
                mask1 = CBLidx2mask[route_in_idxs[0] - 1]
            elif module_defs[route_in_idxs[0]]['type'] == 'convolutional':
                mask1 = CBLidx2mask[route_in_idxs[0]]
            elif module_defs[route_in_idxs[0]]['type'] == 'shortcut':  #yolov5-v4
                mask1 = CBLidx2mask[route_in_idxs[0] - 1]
            elif module_defs[route_in_idxs[0]]['type'] == 'convolutional_nobias':  #yolov5-v3
                if module_defs[route_in_idxs[0]-1]['type'] == 'convolutional':
                    mask1 = CBLidx2mask[route_in_idxs[0] - 1]
                else:
                    mask1 = CBLidx2mask[route_in_idxs[0]-2]
            if module_defs[route_in_idxs[1]]['type'] == 'convolutional':
                mask2 = CBLidx2mask[route_in_idxs[1]]
            elif module_defs[route_in_idxs[1] - 1]['type'] == 'route':
                route_in_idxs_tmp = []
                for layer_i in module_defs[route_in_idxs[1] - 1]['layers'].split(","):
                    if int(layer_i) < 0:
                        route_in_idxs_tmp.append(route_in_idxs[1] - 1 + int(layer_i))
                    else:
                        route_in_idxs_tmp.append(int(layer_i))
                if module_defs[route_in_idxs_tmp[0]]['type'] == 'upsample':
                    mask1tmp = CBLidx2mask[route_in_idxs_tmp[0] - 1]
                elif module_defs[route_in_idxs_tmp[0]]['type'] == 'convolutional':
                    mask1tmp = CBLidx2mask[route_in_idxs_tmp[0]]
                else:
                    assert 0
                if module_defs[route_in_idxs_tmp[1]]['type'] == 'convolutional':
                    mask2tmp = CBLidx2mask[route_in_idxs_tmp[1]]
                else:
                    assert 0
                mask2=np.concatenate([mask1tmp, mask2tmp])
            else:
                mask2 = CBLidx2mask[route_in_idxs[1] - 1]
            if module_defs[route_in_idxs[0]]['type'] == 'convolutional_nobias': #yolov5-v3
                return [mask1,mask2]
            return np.concatenate([mask1, mask2])

        elif len(route_in_idxs) == 4:
            #spp结构中最后一个route
            # mask = CBLidx2mask[route_in_idxs[-1]]
            mask = CBLidx2mask[route_in_idxs[0]]
            return np.concatenate([mask, mask, mask, mask])

        else:
            print("Something wrong with route module!")
            raise Exception
    elif module_defs[idx - 1]['type'] == 'maxpool':  #tiny
        if module_defs[idx - 2]['type'] == 'route':  #v4 tiny
            return get_input_mask(module_defs, idx - 1, CBLidx2mask)
        else:
            return CBLidx2mask[idx - 2]  #v3 tiny

def init_weights_from_loose_model(compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask):

    for idx in CBL_idx:
        compact_CBL = compact_model.module_list[idx]
        loose_CBL = loose_model.module_list[idx]
        out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()
        # print(compact_CBL)
        compact_bn, loose_bn         = compact_CBL.sublayers()[1] if type(compact_CBL.sublayers()[1]).__name__ is 'BatchNorm2D' else compact_CBL.sublayers()[0], loose_CBL.sublayers()[1] if type(loose_CBL.sublayers()[1]).__name__ is 'BatchNorm2D' else loose_CBL.sublayers()[0]
        compact_bn.weight.set_value(loose_bn.weight[out_channel_idx].clone())
        compact_bn.bias.set_value(loose_bn.bias[out_channel_idx].clone())
        compact_bn._mean.set_value(loose_bn._mean[out_channel_idx].clone())
        compact_bn._variance.set_value(loose_bn._variance[out_channel_idx].clone())
        compact_bn._epsilon = loose_bn._epsilon
        compact_bn._momentum = loose_bn._momentum

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        if isinstance(input_mask, list):
            in_channel_idx1 = np.argwhere(input_mask[0])[:, 0].tolist()
            in_channel_idx2 = np.argwhere(input_mask[1])[:, 0].tolist()
        else:
            in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        if type(compact_CBL.sublayers()[0]).__name__ is 'BatchNorm2D':
            mdef=compact_model.module_defs[idx - 1]
            layers = [int(x)+idx-1 for x in mdef['layers'].split(',')]
            compact_conv1 = compact_model.module_list[layers[0]][0]
            compact_conv2 = compact_model.module_list[layers[1]][0]
            loose_conv1 = loose_model.module_list[layers[0]][0]
            loose_conv2 = loose_model.module_list[layers[1]][0]
            tmp1=loose_conv1.weight.numpy()[:, in_channel_idx1, :, :].copy()
            tmp2 = loose_conv2.weight.numpy()[:, in_channel_idx2, :, :].copy()
            half_num=int(len(CBLidx2mask[idx])/2)
            out_channel_idx1=np.argwhere(CBLidx2mask[idx][:half_num])[:, 0].tolist()
            out_channel_idx2 = np.argwhere(CBLidx2mask[idx][half_num:])[:, 0].tolist()
            compact_conv1.weight.set_value(tmp1[out_channel_idx1, :, :, :].copy())
            compact_conv2.weight.set_value(tmp2[out_channel_idx2, :, :, :].copy())
        else:
            compact_conv, loose_conv = compact_CBL.sublayers()[0], loose_CBL.sublayers()[0]
            # print("===>", in_channel_idx, loose_conv, loose_conv.weight.shape)
            tmp = loose_conv.weight.numpy()[:, in_channel_idx, :, :].copy()
            compact_conv.weight.set_value(tmp[out_channel_idx, :, :, :].copy())

    # print(compact_model.module_list)
    for idx in Conv_idx:
        # print("--->", idx)
        compact_conv = compact_model.module_list[idx].sublayers()[0]
        loose_conv = loose_model.module_list[idx].sublayers()[0]

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv.weight.set_value(loose_conv.weight.numpy()[:, in_channel_idx, :, :].copy())
        compact_conv.bias.set_value(loose_conv.bias.clone())

def merge_mask_regular(model, CBLidx2mask, CBLidx2filters):
    for i in range(len(model.module_defs) - 1, -1, -1):
        mtype = model.module_defs[i]['type']
        if mtype == 'shortcut':
            if model.module_defs[i]['is_access']:
                continue

            Merge_masks = []
            layer_i = i
            while mtype == 'shortcut':
                model.module_defs[layer_i]['is_access'] = True

                if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                    bn = int(model.module_defs[layer_i - 1]['batch_normalize'])
                    if bn:
                        Merge_masks.append(CBLidx2mask[layer_i - 1].unsqueeze(0))

                layer_i = int(model.module_defs[layer_i]['from']) + layer_i
                mtype = model.module_defs[layer_i]['type']

                if mtype == 'convolutional':
                    bn = int(model.module_defs[layer_i]['batch_normalize'])
                    if bn:
                        Merge_masks.append(CBLidx2mask[layer_i].unsqueeze(0))

            if len(Merge_masks) > 1:
                Merge_masks = paddle.concat(Merge_masks, 0)
                merge_mask = (paddle.sum(Merge_masks, axis=0) > 0).astype("float32")
            else:
                merge_mask = Merge_masks[0].float()

            layer_i = i
            mtype = 'shortcut'

            # regular
            mask_cnt = int(paddle.sum(merge_mask).item())
            if mask_cnt % 8 != 0:
                mask_cnt = int((mask_cnt // 8 + 1) * 8)

            # bn_module = model.module_list[layer_i - 1][1]
            bn_module = model.module_list[layer_i - 1]["BatchNorm2D"]
            this_layer_sort_bn = bn_module.weight.abs().clone()
            sorted_index_weights = paddle.argsort(this_layer_sort_bn, descending=True)
            merge_mask[sorted_index_weights[:mask_cnt]] = 1.

            while mtype == 'shortcut':

                if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                    bn = int(model.module_defs[layer_i - 1]['batch_normalize'])
                    if bn:
                        CBLidx2mask[layer_i - 1] = merge_mask
                        CBLidx2filters[layer_i - 1] = int(paddle.sum(merge_mask).item())

                layer_i = int(model.module_defs[layer_i]['from']) + layer_i
                mtype = model.module_defs[layer_i]['type']

                if mtype == 'convolutional':
                    bn = int(model.module_defs[layer_i]['batch_normalize'])
                    if bn:
                        CBLidx2mask[layer_i] = merge_mask
                        CBLidx2filters[layer_i] = int(paddle.sum(merge_mask).item())


def update_activation(i, pruned_model, activation, CBL_idx):
    next_idx = i + 1
    if pruned_model.module_defs[next_idx]['type'] == 'convolutional':

        # next_conv = pruned_model.module_list[next_idx][0]
        # conv_sum = next_conv.weight.sum(dim=(2, 3))
        # offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)

        next_conv = pruned_model.module_list[next_idx]["Conv2D"]
        conv_sum = next_conv.weight.sum(axis=(2, 3))
        # print("===>shape compare: ", conv_sum, activation)
        # print("===>shape", i, next_idx, activation.shape, conv_sum.shape)

        offset = conv_sum.matmul(activation.reshape([-1, 1])).reshape([-1])
        if next_idx in CBL_idx:
            next_bn = pruned_model.module_list[next_idx]["BatchNorm2D"]
            # next_bn._mean.sub_(offset)
            next_bn._mean.set_value(paddle.subtract(next_bn._mean, offset))
        else:
            next_conv.bias.set_value(paddle.add(next_conv.bias, offset))
            #next_conv.bias.add_(offset)

def prune_model_keep_size2(model, prune_idx, CBL_idx, CBLidx2mask):

    pruned_model = deepcopy(model)
    activations = []
    for i, model_def in enumerate(model.module_defs):

        if model_def['type'] == 'convolutional':
            activation = paddle.zeros([int(model_def['filters'])])
            if i in prune_idx:
                mask = paddle.to_tensor(CBLidx2mask[i])
                # mask = torch.from_numpy(CBLidx2mask[i])
                bn_module = pruned_model.module_list[i]["BatchNorm2D"]
                # bn_module.weight.mul_(maori modelsk)
                bn_module.weight.set_value(paddle.multiply(bn_module.weight, mask))
                if model_def['activation'] == 'leaky':
                    activation = F.leaky_relu((1 - mask) * bn_module.bias, 0.1)
                elif model_def['activation'] == 'mish':
                    activation = (1 - mask) * paddle.multiply(bn_module.bias, F.softplus(bn_module.bias).tanh())
                elif model_def['activation'] == 'SiLU':  #yolov5-v4
                    activation=(1 - mask) * paddle.multiply(bn_module.bias, F.sigmoid(bn_module.bias))
                elif model_def['activation'] == 'Hardswish':
                    activation=(1 - mask) *bn_module.bias.mul(F.hardtanh(bn_module.bias + 3, 0., 6.) / 6.)
                # print(111)
                update_activation(i, pruned_model, activation, CBL_idx)
                # bn_module.bias.mul_(mask)
                bn_module.bias.set_value(paddle.multiply(bn_module.bias, mask))
            activations.append(activation)

        elif model_def['type'] == 'shortcut':
            actv1 = activations[i - 1]
            from_layer = int(model_def['from'])
            actv2 = activations[i + from_layer]
            activation = actv1 + actv2

            # print(222)
            update_activation(i, pruned_model, activation, CBL_idx)
            activations.append(activation)

        elif model_def['type'] == 'route':
            #spp不参与剪枝，其中的route不用更新，仅占位
            from_layers = [int(s) for s in model_def['layers'].split(',')]
            activation = None
            if len(from_layers) == 1:
                activation = activations[i + from_layers[0] if from_layers[0] < 0 else from_layers[0]]
                if 'groups' in model_def:
                    activation = activation[(activation.shape[0]//2):]
                update_activation(i, pruned_model, activation, CBL_idx)
            elif len(from_layers) == 2:
                actv1 = activations[i + from_layers[0]]
                actv2 = activations[i + from_layers[1] if from_layers[1] < 0 else from_layers[1]]

                activation = paddle.concat((actv1, actv2))
                # update_activation(i, pruned_model, activation, CBL_idx)
                #update_activation_nconv
                next_idx = i + 1

                # if i == 54:
                #     print('----->', actv1.shape)
                #     print(actv2.shape)
                #     print(activation.shape)
                # print(i, i + from_layers[0], i + from_layers[1] if from_layers[1] < 0 else from_layers[1], model_def)
                # # print(from_layers[0], model.module_defs[])
                # print(from_layers[1], next_idx, pruned_model.module_defs[next_idx]['type'])


                if pruned_model.module_defs[next_idx]['type'] == 'convolutional_noconv':
                    next_conv1 = pruned_model.module_list[i + from_layers[0]]["Conv2D"]
                    next_conv2 = pruned_model.module_list[i + from_layers[1] if from_layers[1] < 0 else from_layers[1]]["Conv2D"]
                    conv_sum1 = next_conv1.weight.sum(axis=(2, 3))
                    conv_sum2 = next_conv2.weight.sum(axis=(2, 3))
                    offset1 = conv_sum1.matmul(actv1.reshape([-1, 1])).reshape([-1])
                    offset2 = conv_sum2.matmul(actv2.reshape([-1, 1])).reshape([-1])
                    offset=paddle.concat((offset1, offset2))
                    if next_idx in CBL_idx:
                        next_bn = pruned_model.module_list[next_idx]["Conv2D"]
                        next_bn.running_mean.data.sub_(offset)
                else:
                    update_activation(i, pruned_model, activation, CBL_idx)
            activations.append(activation)

        elif model_def['type'] == 'upsample':
            # activation = torch.zeros(int(model.module_defs[i - 1]['filters'])).cuda()
            activations.append(activations[i-1])

        elif model_def['type'] == 'yolo':
            activations.append(None)

        elif model_def['type'] == 'focus':
            activations.append(None)

        elif model_def['type'] == 'convolutional_nobias':
            activations.append(activations[i - 1])
            # activation = torch.zeros(int(model_def['filters'])).cuda()
            # activations.append(activation)

        elif model_def['type'] == 'convolutional_noconv':
            activation = paddle.zeros([int(model_def['filters'])])
            if i in prune_idx:
                mask = paddle.to_tensor(CBLidx2mask[i])
                # mask = torch.from_numpy(CBLidx2mask[i])
                bn_module = pruned_model.module_list[i]["Conv2D"]
                # bn_module.weight.mul_(mask)
                bn_module.weight.set_value(paddle.multiply(bn_module.weight, mask))

                activation = F.leaky_relu((1 - mask) * bn_module.bias, 0.1)
                # if model_def['activation'] == 'leaky':
                #     activation = F.leaky_relu((1 - mask) * bn_module.bias, 0.1)
                # elif model_def['activation'] == 'mish':
                #     activation = (1 - mask) * bn_module.bias.mul(F.softplus(bn_module.bias).tanh())

                update_activation(i, pruned_model, activation, CBL_idx)
                # bn_module.bias.mul_(mask)
                bn_module.bias.set_value(paddle.multiply(bn_module.bias, mask))
            activations.append(activation)

        elif model_def['type'] == 'maxpool':  #区分spp和tiny
            if model.module_defs[i + 1]['type'] == 'route':
                activations.append(None)
            else:
                activation = activations[i-1]
                update_activation(i, pruned_model, activation, CBL_idx)
                activations.append(activation)

    return pruned_model
