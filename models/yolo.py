# YOLOv5 reproduction 🚀 by GuoQuanhao
"""
YOLO-specific layers

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path
import paddle.nn.functional as F
import paddle.nn as nn
import paddle

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import check_version, check_yaml, make_divisible, print_args, LOGGER
from utils.plots import feature_visualization
from utils.paddle_utils import copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, \
    select_device, time_sync

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Layer):
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.onnx_dynamic = False  # ONNX export parameter
        self.stride = None  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [paddle.zeros([1])] * self.nl  # init grid
        self.anchor_grid = [paddle.zeros([1])] * self.nl  # init anchor grid
        self.register_buffer('anchors', paddle.to_tensor(anchors).astype('float32').reshape([self.nl, -1, 2]))  # shape(nl,na,2)
        self.m = nn.LayerList(nn.Conv2D(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # x[i] = x[i].reshape([bs, self.na, self.no, ny, nx]).transpose([0, 1, 3, 4, 2])
            x[i] = x[i].reshape([bs, self.na, self.no, ny * nx]).transpose([0, 1, 3, 2])

            if not self.training:  # inference
                # y = F.sigmoid(x[i]).reshape((bs, self.na, ny * nx * self.no))
                y = F.sigmoid(x[i])
                print("====>y shape: ", y.shape)

                # # y[:, :, :, :, 0:2] = y[:, :, :, :, 0:2] * 2. - 0.5
                # # y[:,:,:,:, 2:4] = (y[:,:,:,:, 2:4] * 2) ** 2
                # # max_val = paddle.max(y[:, :, :, :, 5:], axis=-1)
                # # max_idx = paddle.argmax(y[:, :, :, :, 5:], axis=-1)
                # # y[:, :, :, :, 5] = max_idx.astype(y.dtype)
                # # y[:, :, :, :, 6] = max_val * y[:, :, :, :, 4]
                #
                # y1 = y[:, :, :, :, 0:2] * 2. - 0.5
                # y2 = (y[:,:,:,:, 2:4] * 2) ** 2
                # y3 = paddle.unsqueeze(y[:,:,:,:, 4],  axis=[-1])
                #
                # cls_cols = y[:, :, :, :, 5:]
                # # max_val = paddle.max(cls_cols, axis=-1, keepdim=True).astype(y.dtype)
                # max_idx = paddle.argmax(cls_cols, axis=-1, keepdim=True).astype(y.dtype)
                # max_val = paddle.max(cls_cols, axis=-1, keepdim=True).astype(y.dtype)
                #
                # # print("--===>", y.shape, max_val.shape, max_idx.shape)
                # # y4 = paddle.unsqueeze(max_idx,axis=[0])
                # # # y5 = paddle.unsqueeze(paddle.multiply(max_val, y[:, :, :, :, 4]), axis=[-1])
                # # y5 = paddle.unsqueeze(max_val, axis=[0])
                #
                # # if i == 2 and y.shape[2] > 11:
                # #     print(y.shape)
                # #     print(y3[0, 1, 11, 8], y5[0, 1, 11, 8], max_val[0, 1, 11, 8])
                # #     print("checking...", y[0, 1, 11, 8, 4], paddle.max(y[0, 1, 11, 8, 5:])) # score max_val
                #
                # z.append(paddle.concat([y1, y2, y3, max_idx, max_val], axis=-1))

                z.append(y)

        return x if self.training else z

    def forward_bk2(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].reshape([bs, self.na, self.no, ny, nx]).transpose([0, 1, 3, 4, 2])

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = F.sigmoid(x[i])
                if self.inplace:
                    y[:,:,:,:, 0:2] = (y[:,:,:,:, 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[:,:,:,:, 2:4] = (y[:,:,:,:, 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[:,:,:,:, 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[:,:,:,:, 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = paddle.concat((xy, wh, y[:,:,:,:, 4:]), -1)
                z.append(y.reshape([bs, -1, self.no]))

        return x if self.training else (paddle.concat(z, 1), x)

class Model(nn.Layer):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None, mode='training'):  # model, input channels, number of classes
        # super().__init__()
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for paddle hub
            self.yaml_file = Path(cfg).name
            with open(cfg, errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            print(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            print(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch], mode=mode)  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace

            # x = self.forward(paddle.zeros([1, ch, s, s]))
            # print(x)

            m.stride = paddle.to_tensor([s / x.shape[-2] for x in self.forward(paddle.zeros([1, ch, s, s]))])  # forward
            # m.stride = paddle.to_tensor([x.shape[-2] for x in self.forward(paddle.zeros([1, ch, s, s]))])  # forward
            # print(m.stride)
            # print("m.stride.shape===>", ch, s, m.stride)
            m.anchors /= m.stride.reshape([-1, 1, 1])
            # print(m.anchors)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        print('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        # print("--->ori input: ", x.shape, x)
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return paddle.concat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):

        print(self.save)
        y, dt = [], []  # outputs
        for m in self.model:

            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                print(m.f)

            if profile:
                self._profile_one_layer(m, x, dt)

            if isinstance(m, Detect):
                print(m, m.anchors, m.na)

            x = m(x)

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = paddle.concat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            print(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'layer'}")
        print(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            print(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.model[-1]  # Detect() layer
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.reshape([m.na, -1])  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else paddle.log(cf / cf.sum())  # cls
            mi.bias = paddle.create_parameter(shape=b.flatten().shape, dtype='float32',
                           default_initializer=paddle.nn.initializer.Assign(b.flatten()))

    def _print_biases(self):
        m = self.model[-1]  # Detect() layer
        for mi in m.m:  # from
            b = mi.bias.detach().reshape([m.na, -1]).t()  # conv.bias(255) to (3,85)
            print(
                ('%6g Conv2D.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.sublayers():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (F.sigmoid(m.w.detach()) * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2D() + BatchNorm2D() layers
        print('Fusing layers... ')
        for m in self.model.sublayers():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                # print(m.bn)
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def autoshape(self):  # add AutoShape layer
        print('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch, mode='training'):  # model_dict, input_channels(3)
    if mode == 'training':
        print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'layer':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, layer, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2D:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # layer
        t = str(m)[8:-2].replace('__main__.', '')  # layer type
        np = int(sum(x.numel() for x in m_.parameters()))  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        if mode == 'training':
            print(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg)
    model.train()

    # Profile
    if opt.profile:
        img = paddle.rand([8 if paddle.device.is_compiled_with_cuda() else 1, 3, 640, 640])
        y = model(img, profile=True)

    # LogWriter (https://github.com/PaddlePaddle/VisualDL)
    # from visualdl import LogWriter
    # vl_writer = LogWriter('.')
    # print("Run 'visualdl --logdir=models' to view VisualDL at http://localhost:6006/")
