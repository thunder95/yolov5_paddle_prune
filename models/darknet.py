import paddle
import paddle.nn as nn
import numpy as np
from .common import SiLU

def create_grids(self, img_size=416, ng=(13, 13), type="float32"):
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)

    # build xy offsets
    yv, xv = paddle.meshgrid([paddle.arange(ny), paddle.arange(nx)])
    self.grid_xy = paddle.stack((xv, yv), 2).astype(type).reshape((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors / self.stride
    self.anchor_wh = self.anchor_vec.reshape((1, self.na, 1, 1, 2)).astype(type)
    self.ng = paddle.to_tensor(ng)
    self.nx = nx
    self.ny = ny

class YOLOLayer(nn.Layer):
    def __init__(self, anchors, nc, img_size, yolo_index, arc):
        super(YOLOLayer, self).__init__()

        self.anchors = paddle.to_tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints
        self.arc = arc

    def forward(self, p, img_size, var=None):

        bs, ny, nx = p.shape[0], p.shape[-2], p.shape[-1]
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny), p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.reshape((bs, self.na, self.nc + 5, self.ny, self.nx)).transpose([0, 1, 3, 4, 2])  # prediction

        if self.training:
            return p

        else:  # inference
            # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
            io = p.clone()  # inference output

            grid = self._make_grid(nx, ny)
            anchor_grid=self.anchors.clone().reshape([1, -1, 1, 1, 2])

            y = paddle.nn.functional.sigmoid(io)
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * self.stride  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh
            z=y.reshape((bs, -1, 5 + self.nc))

            return z, p

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = paddle.meshgrid([paddle.arange(ny), paddle.arange(nx)])
        return paddle.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype("float32")

def create_modules(module_defs, img_size, arc):
    # Constructs module list of layer blocks from module configuration in module_defs
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.LayerList()
    routs = []  # list of layers which rout to deeper layes
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            kernel_size = int(mdef['size'])
            pad = (kernel_size - 1) // 2 if int(mdef['pad']) else 0
            # print("---==>filters: ", output_filters[-1], filters)
            modules.add_sublayer('Conv2D', nn.Conv2D(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=kernel_size,
                                                   stride=int(mdef['stride']),
                                                   padding=pad,
                                                   bias_attr=not bn))
            if bn:
                modules.add_sublayer('BatchNorm2D', nn.BatchNorm2D(filters, momentum=0.9))

            if mdef['activation'] == 'leaky':  # TODO: activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_sublayer('activation', nn.LeakyReLU(0.1, inplace=True))
                # modules.add_sublayer('activation', nn.PReLU(num_parameters=1, init=0.10))
                # modules.add_sublayer('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_sublayer('activation', nn.Mish())
            elif mdef['activation'] == 'Hardswish':
                modules.add_sublayer('activation', nn.Hardswish())
            elif mdef['activation']=='SiLU':
                modules.add_sublayer('activation', SiLU())
        elif mdef['type'] == 'convolutional_nobias':
            filters = int(mdef['filters'])
            kernel_size = int(mdef['size'])
            modules.add_sublayer('Conv2D', nn.Conv2D(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=kernel_size,
                                                   stride=int(mdef['stride']),
                                                   bias_attr=False))
        elif mdef['type'] == 'convolutional_noconv':
            filters = int(mdef['filters'])
            modules.add_sublayer('BatchNorm2D', nn.BatchNorm2D(filters, momentum=0.9))
            modules.add_sublayer('activation', nn.LeakyReLU(0.1))

        elif mdef['type'] == 'maxpool':
            kernel_size = int(mdef['size'])
            stride = int(mdef['stride'])
            maxpool = nn.MaxPool2D(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            if kernel_size == 2 and stride == 1:  # yolov3-tiny
                # modules.add_sublayer('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_sublayer('MaxPool2D', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'upsample':
            modules = nn.Upsample(scale_factor=int(mdef['stride']), mode='nearest')

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = [int(x) for x in mdef['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            if 'groups' in mdef:
                filters = filters // 2
            routs.extend([l if l > 0 else l + i for l in layers])
            # if mdef[i+1]['type'] == 'reorg3d':
            #     modules = nn.Upsample(scale_factor=1/float(mdef[i+1]['stride']), mode='nearest')  # reorg3d

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            filters = output_filters[int(mdef['from'])]
            layer = int(mdef['from'])
            routs.extend([i + layer if layer < 0 else layer])

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            # torch.Size([16, 128, 104, 104])
            # torch.Size([16, 64, 208, 208]) <-- # stride 2 interpolate dimensions 2 and 3 to cat with prior layer
            pass

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            mask = [int(x) for x in mdef['mask'].split(',')]  # anchor mask
            modules = YOLOLayer(anchors=mdef['anchors'][mask],  # anchor list
                                nc=int(mdef['classes']),  # number of classes
                                img_size=img_size,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1 or 2
                                arc=arc)  # yolo architecture

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                if arc == 'defaultpw' or arc == 'Fdefaultpw':  # default with positive weights
                    b = [-4, -3.6]  # obj, cls
                elif arc == 'default':  # default no pw (40 cls, 80 obj)
                    b = [-5.5, -4.0]
                elif arc == 'uBCE':  # unified BCE (80 classes)
                    b = [0, -8.5]
                elif arc == 'uCE':  # unified CE (1 background + 80 classes)
                    b = [10, -0.1]
                elif arc == 'Fdefault':  # Focal default no pw (28 cls, 21 obj, no pw)
                    b = [-2.1, -1.8]
                elif arc == 'uFBCE' or arc == 'uFBCEpw':  # unified FocalBCE (5120 obj, 80 classes)
                    b = [0, -6.5]
                elif arc == 'uFCE':  # unified FocalCE (64 cls, 1 background + 80 classes)
                    b = [7.7, -1.1]

                # print(module_list)
                bias = module_list[-1]["Conv2D"].bias.reshape([len(mask), -1])  # 255 to 3x85
                bias[:, 4] += b[0] - bias[:, 4].mean()  # obj
                bias[:, 5:] += b[1] - bias[:, 5:].mean()  # cls
                # bias = torch.load('weights/yolov3-spp.bias.pt')[yolo_index]  # list of tensors [3x85, 3x85, 3x85]


                module_list[-1]["Conv2D"].bias = paddle.create_parameter(shape=bias.flatten().shape, dtype='float32',
                               default_initializer=paddle.nn.initializer.Assign(bias.flatten()))

                # utils.print_model_biases(model)
            except Exception as e:
                print(e)
                print('WARNING: smart bias initialization failure.')
        elif mdef['type'] == 'focus':
            filters = int(mdef['filters'])
        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return module_list, routs, hyperparams

def parse_model_cfg(path):
    # Parses the yolo-v3 layer configuration file and returns module definitions
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    mdefs = []  # module definitions
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].rstrip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if 'anchors' in key:
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # np anchors
            else:
                mdefs[-1][key] = val.strip()

    return mdefs

def get_yolo_layers(model):
    return [i for i, x in enumerate(model.module_defs) if x['type'] == 'yolo']  # [82, 94, 106] for yolov3

class Darknet(nn.Layer):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size=(416, 416), arc='default'):
        super(Darknet, self).__init__()

        if isinstance(cfg, str):
            self.module_defs = parse_model_cfg(cfg)
        elif isinstance(cfg, list):
            self.module_defs = cfg

        self.module_list, self.routs, self.hyperparams = create_modules(self.module_defs, img_size, arc)
        self.yolo_layers = get_yolo_layers(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training

    def forward(self, x, var=None,augment=False):
        img_size = x.shape[-2:]
        layer_outputs = []
        output = []

        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            # print("i=", i, mdef)
            if mtype in ['convolutional', 'upsample', 'maxpool','convolutional_nobias','convolutional_noconv']:
                # print(module)
                x = module(x)
            elif mtype == 'focus':
                x = paddle.concat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
            elif mtype == 'route':
                layers = [int(x) for x in mdef['layers'].split(',')]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                    if 'groups' in mdef:
                        x = x[:, (x.shape[1]//2):]
                else:
                    try:
                        x = paddle.concat([layer_outputs[i] for i in layers], 1)
                    except:  # apply stride 2 for darknet reorg layer
                        layer_outputs[layers[1]] = F.interpolate(layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        x = paddle.concat([layer_outputs[i] for i in layers], 1)
                    # print(''), [print(layer_outputs[i].shape) for i in layers], print(x.shape)
            elif mtype == 'shortcut':
                x = x + layer_outputs[int(mdef['from'])]
            elif mtype == 'yolo':
                x = module(x, img_size)
                output.append(x)
            layer_outputs.append(x if i in self.routs else [])

        if self.training:
            return output

        else:
            io, p = list(zip(*output))  # inference output, training output
            return paddle.concat(io, 1), p

    def fuse(self):
        # todo later
        # Fuse Conv2d + BatchNorm2d layers throughout model
        # fused_list = nn.ModuleList()
        # for a in list(self.children())[0]:
        #     if isinstance(a, nn.Sequential):
        #         for i, b in enumerate(a):
        #             if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
        #                 # fuse this bn layer with the previous conv2d layer
        #                 conv = a[i - 1]
        #                 fused = torch_utils.fuse_conv_and_bn(conv, b)
        #                 a = nn.Sequential(fused, *list(a.children())[i + 1:])
        #                 break
        #     fused_list.append(a)
        # self.module_list = fused_list
        return self
        # model_info(self)  # yolov3-spp reduced from 225 to 152 layers

if __name__ == '__main__':
    model = Darknet("yolov5n_v6_person.cfg", (640, 640))
    print(model)
