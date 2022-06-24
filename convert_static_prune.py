# YOLOv5 reproduction 🚀 by thunder95
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pdparams --img 640
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import paddle
import warnings

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, check_dataset, \
    nms, increment_path, non_max_suppression, print_args, save_one_box, scale_coords, strip_optimizer, xyxy2xywh, LOGGER, check_yaml
from utils.plots import Annotator, colors
from utils.paddle_utils import load_classifier, select_device, time_sync, initialize_weights
from models.yolo import Model
import yaml
from models.darknet import Darknet
from models.copy_weight import copy_weight_v6_reverse

warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='paddle')
warnings.filterwarnings(action='ignore', category=Warning, module='paddle')
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='utils')



@paddle.no_grad()
def run(weights=ROOT / 'yolov5s.pdparams',  # model.pdparams path(s)
        single_cls=False,  # treat as single-class dataset
        data = None,
        cfg = None,
        hyp = None,
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.01,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    device = select_device(device)
    data = check_dataset(data)  # check
    # print("nc num: ", data['nc'])
    # exit()
    nc = 1 if single_cls else int(data['nc'])

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pdparams', '.onnx', '.tflite', '.pb', '']
    # check_suffix(w, suffixes)  # check weights have acceptable suffix
    pdparams, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

    # Load model
    # print("--------------->", weights, w)
    # print(hyp.get('anchors'))
    # check_suffix(weights, '.pdparams')
    # # print("--->cfg: ", cfg)
    # yaml_cfg = "models/yolov5n.yaml"  # 暂时先写死
    # model = Model(yaml_cfg, ch=3, nc=nc, anchors=hyp.get('anchors'))  # create
    # names = {0: "person"}
    # if cfg.endswith(".cfg"):
    #     # print("debug...", weights, cfg)
    #     cfg_model = Darknet(opt.cfg, (opt.imgsz, opt.imgsz))
    #     initialize_weights(cfg_model)
    #     if w.endswith('.pdparams'):
    #         cfg_model.set_state_dict(paddle.load(w)['state_dict'])
    #     copy_weight_v6_reverse(model, cfg_model)
    # model.set_state_dict(paddle.load(w)['state_dict'])
    # model.eval()
    # print(imgsz)
    # fake_input = paddle.ones([1, 3, 640, 640], dtype=paddle.float32)
    # test_out = model(fake_input)
    # # print(test_out)
    # exit()


    if pdparams:
        yaml_cfg = "models/yolov5n.yaml"  # 暂时先写死
        model = Model(yaml_cfg, ch=3, nc=nc, anchors=hyp.get('anchors'))  # create
        if cfg.endswith(".cfg"):
            cfg_model = Darknet(opt.cfg, (opt.imgsz, opt.imgsz))
            initialize_weights(cfg_model)
            cfg_model.set_state_dict(paddle.load(w)['state_dict'])
            copy_weight_v6_reverse(model, cfg_model)

        raw_wgt = paddle.load(w)
        # print(raw_wgt.keys())
        wgt = raw_wgt['state_dict']
        # for k in wgt:
        #     print(k, wgt[k].shape)
            # break

        model.set_state_dict(wgt)
        model.eval()
        stride = int(model.stride.max())  # model stride
        names = {0: "person"}
        # names =paddle.load(w)['names']  # get class names
        # if classify:  # second-stage classifier
        #     modelc = load_classifier(name='resnet50', n=2)  # initialize
        #     modelc.set_state_dict(paddle.load('resnet50.pdparams')).eval()

        # test_out = model(paddle.ones([1, 3, imgsz, imgsz], dtype=paddle.float32))
        # print(test_out)
        # exit()


    elif onnx:
        if dnn:
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        else:
            check_requirements(('onnx', 'onnxruntime-gpu' if paddle.device.is_compiled_with_cuda() else 'onnxruntime'))
            import onnxruntime
            session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # print(imgsz)
    # exit()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pdparams)
        bs = len(dataset)  # batch_size
    else:
        # print("debug: ", imgsz, stride)
        # exit()
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pdparams)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs



    # Run inference
    # if pdparams and 'CUDA' in str(device):
    #     model(paddle.zeros([1, 3, *imgsz]).astype(model.parameters()[0].dtype))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, img, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = paddle.to_tensor(img)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # np.save("fake_img", img.numpy())

        # Inference
        if pdparams:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # print(img)
            pred = model(img, augment=augment, visualize=visualize)[0]
            # print("--->pred", pred)
            # exit()
            # wgt2 = model.state_dict()
            # for k in wgt2:
            #     if k == "model.23.m.0.cv2.conv.weight":
            #         print(k, wgt2[k])
            #     # break
            #     print(k, wgt2[k].shape)
            # exit()
        elif onnx:
            if dnn:
                net.setInput(img)
                pred = paddle.to_tensor(net.forward())
            else:
                pred = paddle.to_tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = paddle.to_tensor(pred)
        t3 = time_sync()
        dt[1] += t3 - t2

        strides = [8, 16, 32]
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]

        wx, hy = 320, 320
        dim = (
        (strides[0] / hy, strides[0] / wx), (strides[1] / hy, strides[1] / wx), (strides[2] / hy, strides[2] / wx))
        xdim = (int(wx / strides[0]), int(wx / strides[1]), int(wx / strides[2]))

        s = hy * wx
        dim_num = (
        int(s / (strides[0] * strides[0])), int(s / (strides[1] * strides[1])), int(s / (strides[2] * strides[2])))

        cls_num = nc
        row_num = cls_num + 5

        if pdparams:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            model.eval()

            # img = paddle.ones([1, 3, 640, 640])
            # print("==============================================")
            pred = model(img, augment=augment, visualize=visualize)
            # for p in pred:
            #     print(p.shape)

            # # 后处理提取boxes
            boxes = []  # xywh, score, cls

            for l in range(len(pred)):
                pred[l] = pred[l].numpy()
                bs, ch, nd, cn = pred[l].shape
                rows = dim_num[l]
                print(l, rows)
                for b in range(bs):
                    for c in range(ch):
                        for r in range(rows):
                            offset = r * row_num
                            max_cls_val = 0
                            max_cls_id = 0
                            score = pred[l][b, c, r, 4]
                            if score < conf_thres:
                                continue

                            for i in range(nc):
                                if pred[l][b, c, r, i + 5] > max_cls_val:
                                    max_cls_val = pred[l][b, c, r, i + 5]
                                    max_cls_id = i

                            score *= max_cls_val
                            if score < conf_thres:
                                continue

                            y = int(r / xdim[l])
                            x = int(r % xdim[l])
                            # print("===>good score: ", b, c, y, x, score, pred[l][b, c, offset],
                            #       pred[l][b, c, offset+1], pred[l][b, c, offset+2], pred[l][b, c, offset+3], y, strides[l])

                            tmp_box = [
                                (pred[l][b, c, r, 0] * 2 - 0.5 + x) * strides[l],
                                (pred[l][b, c, r, 1] * 2 - 0.5 + y) * strides[l],  # i, y, x, 2
                                (pred[l][b, c, r, 2] * 2) ** 2 * anchors[l][c * 2],
                                (pred[l][b, c, r, 3] * 2) ** 2 * anchors[l][c * 2 + 1],
                                max_cls_id,
                                score,  # scores
                            ]
                            # print(tmp_box)

                            tmp_box[0] = tmp_box[0] - tmp_box[2] / 2  # x1
                            tmp_box[1] = tmp_box[1] - tmp_box[3] / 2  # y1
                            tmp_box[2] = tmp_box[0] + tmp_box[2]  # x2
                            tmp_box[3] = tmp_box[1] + tmp_box[3]  # y2

                            boxes.append(tmp_box)

        boxes = np.asarray(boxes)
        # print("===> boxes: ", boxes)
        # print(boxes.shape)

        if len(list(boxes)):
            idx = nms(boxes[:, :4], boxes[:, 5], iou_thres)
            print("after nms: ", idx)

        x_spec = paddle.static.InputSpec(shape=[None, 3, 320, 320], dtype='float32', name='x')
        net = paddle.jit.save(model, path="prune_net", input_spec=[x_spec])  # 动静转换
        exit()

        # NMS
        # conf_thres = 0.001
        # iou_thres = 0.6
        # print(pred.numpy())
        # print(classes)
        # print(agnostic_nms)
        # print(max_det)
        # exit()
        pred = non_max_suppression(pred.numpy(), conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # print(img.shape[2:], type(img.shape[2:]))
            s += '%gx%g ' % tuple(img.shape[2:])  # print string
            gn = paddle.to_tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in np.unique(det[:, -1]):
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(paddle.to_tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            # cv2.imshow(str(p), im0)
            # cv2.waitKey(0)  # 1 millisecond
            # if view_img:
            #     cv2.imshow(str(p), im0)
            #     cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pdparams', help='model path(s)')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[320], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.data, opt.hyp, = check_yaml(opt.data), check_yaml(opt.hyp)  # check YAML
    print_args(FILE.stem, opt)

    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
