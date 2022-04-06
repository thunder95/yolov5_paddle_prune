import paddle

def copy_conv(conv_src,conv_dst):
    name_list = [x[0] for x in conv_dst.named_sublayers()]
    if "Conv2D" in name_list:
        conv_dst["Conv2D"] = conv_src.conv
        conv_dst["BatchNorm2D"] = conv_src.bn
        conv_dst["activation"] = conv_src.act
    else:
        conv_dst[0] = conv_src.conv
        conv_dst[0] = conv_src.bn
        conv_dst[0] = conv_src.act

def copy_conv_reverse(conv_src, conv_dst):
    # print("copy_conv_reverse--->")
    # print(conv_src)
    # print(conv_dst)
    # exit()
    name_list = [x[0] for x in conv_dst.named_sublayers()]
    if "Conv2D" in name_list:
        conv_src.conv = conv_dst["Conv2D"]
        conv_src.bn = conv_dst["BatchNorm2D"]
        conv_src.act = conv_dst["activation"]
    else:
        conv_src.conv = conv_dst[0]
        conv_src.bn = conv_dst[1]
        conv_src.act = conv_dst[2]

def copy_conv_idx(conv_src,conv_dst,idx):
    # print(conv_src)
    # print(conv_dst)
    copy_conv(conv_src,conv_dst)
    return idx+1

def copy_conv_idx_reverse(conv_src,conv_dst,idx):
    copy_conv_reverse(conv_src,conv_dst)
    return idx+1


def copy_c3(c3_src,module_lists,idx,num,shortcut=True):
    # C3-cv2
    idx=copy_conv_idx(c3_src.cv2, module_lists[idx],idx)
    # route
    idx = idx + 1
    # C3-cv1
    idx=copy_conv_idx(c3_src.cv1, module_lists[idx],idx)
    # C3-m
    for i in range(num):
        # Bottleneck-cv1
        idx=copy_conv_idx(c3_src.m[i].cv1, module_lists[idx],idx)
        # Bottleneck-cv2
        idx=copy_conv_idx(c3_src.m[i].cv2, module_lists[idx],idx)
        if shortcut:
            #m-add
            idx = idx + 1
    #C3-cat
    idx = idx + 1
    #C3--cv3
    idx=copy_conv_idx(c3_src.cv3, module_lists[idx],idx)
    return idx

def copy_c3_reverse(c3_src,module_lists,idx,num,shortcut=True):
    # C3-cv2
    idx=copy_conv_idx_reverse(c3_src.cv2, module_lists[idx],idx)
    # route
    idx = idx + 1
    # C3-cv1
    idx=copy_conv_idx_reverse(c3_src.cv1, module_lists[idx],idx)
    # C3-m
    for i in range(num):
        # Bottleneck-cv1
        idx=copy_conv_idx_reverse(c3_src.m[i].cv1, module_lists[idx],idx)
        # Bottleneck-cv2
        idx=copy_conv_idx_reverse(c3_src.m[i].cv2, module_lists[idx],idx)
        if shortcut:
            #m-add
            idx = idx + 1
    #C3-cat
    idx = idx + 1
    #C3--cv3
    idx=copy_conv_idx_reverse(c3_src.cv3, module_lists[idx],idx)
    return idx

def copy_weight_v6(modelyolov5,model):
    idx=0
    depth_multiple=0.33
    # if 'depth_multiple' in modelyolov5.yaml:
    #     depth_multiple=modelyolov5.yaml['depth_multiple']
    conv0 = list(modelyolov5.model.children())[0]
    idx=copy_conv_idx(conv0, model.module_list[idx],idx)
    conv1 = list(modelyolov5.model.children())[1]
    idx=copy_conv_idx(conv1, model.module_list[idx],idx)
    cspnet2 = list(modelyolov5.model.children())[2]
    idx=copy_c3(cspnet2,model.module_list,idx, round(3 * depth_multiple))
    conv3 = list(modelyolov5.model.children())[3]
    idx=copy_conv_idx(conv3, model.module_list[idx],idx)
    cspnet4 = list(modelyolov5.model.children())[4]
    idx = copy_c3(cspnet4, model.module_list,idx, round(6 * depth_multiple))
    conv5 = list(modelyolov5.model.children())[5]
    idx=copy_conv_idx(conv5, model.module_list[idx],idx)
    cspnet6 = list(modelyolov5.model.children())[6]
    idx = copy_c3(cspnet6, model.module_list,idx, round(9 * depth_multiple))
    conv7 = list(modelyolov5.model.children())[7]
    idx=copy_conv_idx(conv7, model.module_list[idx],idx)
    cspnet8 = list(modelyolov5.model.children())[8]
    idx = copy_c3(cspnet8, model.module_list,idx, round(3 * depth_multiple))
    sppf9 = list(modelyolov5.model.children())[9]
    idx=copy_conv_idx(sppf9.cv1, model.module_list[idx],idx)

    # print("idx", idx)
    model.module_list[idx] = sppf9.m
    idx = idx + 1
    model.module_list[idx] = sppf9.m
    idx = idx + 1
    model.module_list[idx] = sppf9.m
    idx = idx + 1
    #route
    idx = idx + 1
    idx=copy_conv_idx(sppf9.cv2, model.module_list[idx],idx)
    conv10 = list(modelyolov5.model.children())[10]
    idx=copy_conv_idx(conv10, model.module_list[idx],idx)
    upsample11 = list(modelyolov5.model.children())[11]
    model.module_list[idx] = upsample11
    idx = idx + 1
    #route
    idx=idx+1
    cspnet13 = list(modelyolov5.model.children())[13]
    idx = copy_c3(cspnet13, model.module_list, idx,round(3 * depth_multiple),False)
    conv14 = list(modelyolov5.model.children())[14]
    idx=copy_conv_idx(conv14, model.module_list[idx],idx)
    upsample15 = list(modelyolov5.model.children())[15]
    model.module_list[idx] = upsample15
    idx = idx + 1
    # route
    idx = idx + 1
    cspnet17 = list(modelyolov5.model.children())[17]
    idx = copy_c3(cspnet17, model.module_list, idx,round(3 * depth_multiple), False)
    #conv
    conv_detect1_idx=idx
    idx=idx+1
    #yolo
    idx=idx+1
    #route
    idx=idx+1
    conv18 = list(modelyolov5.model.children())[18]
    idx=copy_conv_idx(conv18, model.module_list[idx],idx)
    # route
    idx = idx + 1
    cspnet20 = list(modelyolov5.model.children())[20]
    idx = copy_c3(cspnet20, model.module_list,idx, round(3 * depth_multiple), False)
    # conv
    conv_detect2_idx = idx
    idx = idx + 1
    # yolo
    idx = idx + 1
    # route
    idx = idx + 1
    conv21 = list(modelyolov5.model.children())[21]
    idx=copy_conv_idx(conv21, model.module_list[idx],idx)
    # route
    idx = idx + 1
    cspnet23 = list(modelyolov5.model.children())[23]
    idx = copy_c3(cspnet23, model.module_list, idx,round(3 * depth_multiple), False)
    # conv
    conv_detect3_idx = idx
    detect24 = list(modelyolov5.model.children())[24]

    name_list = [x[0] for x in model.module_list[conv_detect1_idx].named_sublayers()]
    if "Conv2D" in name_list:
        model.module_list[conv_detect1_idx]["Conv2D"] = detect24.m[0]
        model.module_list[conv_detect2_idx]["Conv2D"] = detect24.m[1]
        model.module_list[conv_detect3_idx]["Conv2D"] = detect24.m[2]
    else:
        model.module_list[conv_detect1_idx][0] = detect24.m[0]
        model.module_list[conv_detect2_idx][0] = detect24.m[1]
        model.module_list[conv_detect3_idx][0] = detect24.m[2]

def copy_weight_v6x(modelyolov5,model):
    idx=0
    depth_multiple=0.33
    if 'depth_multiple' in modelyolov5.yaml:
        depth_multiple=modelyolov5.yaml['depth_multiple']
    conv0 = list(modelyolov5.model.children())[0]
    idx=copy_conv_idx(conv0, model.module_list[idx],idx)
    conv1 = list(modelyolov5.model.children())[1]
    idx=copy_conv_idx(conv1, model.module_list[idx],idx)
    cspnet2 = list(modelyolov5.model.children())[2]
    idx=copy_c3(cspnet2,model.module_list,idx, round(3 * depth_multiple))
    conv3 = list(modelyolov5.model.children())[3]
    idx=copy_conv_idx(conv3, model.module_list[idx],idx)
    cspnet4 = list(modelyolov5.model.children())[4]
    idx = copy_c3(cspnet4, model.module_list,idx, round(6 * depth_multiple))
    conv5 = list(modelyolov5.model.children())[5]
    idx=copy_conv_idx(conv5, model.module_list[idx],idx)
    cspnet6 = list(modelyolov5.model.children())[6]
    idx = copy_c3(cspnet6, model.module_list,idx, round(9 * depth_multiple))
    conv7 = list(modelyolov5.model.children())[7]
    idx=copy_conv_idx(conv7, model.module_list[idx],idx)
    cspnet8 = list(modelyolov5.model.children())[8]
    idx = copy_c3(cspnet8, model.module_list,idx, round(3 * depth_multiple))
    conv9 = list(modelyolov5.model.children())[9]
    idx = copy_conv_idx(conv9, model.module_list[idx], idx)
    cspnet10 = list(modelyolov5.model.children())[10]
    idx = copy_c3(cspnet10, model.module_list, idx, round(3 * depth_multiple))
    sppf11 = list(modelyolov5.model.children())[11]
    idx=copy_conv_idx(sppf11.cv1, model.module_list[idx],idx)
    model.module_list[idx] = sppf11.m
    idx = idx + 1
    model.module_list[idx] = sppf11.m
    idx = idx + 1
    model.module_list[idx] = sppf11.m
    idx = idx + 1
    #route
    idx = idx + 1
    idx=copy_conv_idx(sppf11.cv2, model.module_list[idx],idx)
    conv12 = list(modelyolov5.model.children())[12]
    idx=copy_conv_idx(conv12, model.module_list[idx],idx)
    upsample13 = list(modelyolov5.model.children())[13]
    model.module_list[idx] = upsample13
    idx = idx + 1
    #route
    idx=idx+1
    cspnet15 = list(modelyolov5.model.children())[15]
    idx = copy_c3(cspnet15, model.module_list, idx,round(3 * depth_multiple),False)
    conv16 = list(modelyolov5.model.children())[16]
    idx=copy_conv_idx(conv16, model.module_list[idx],idx)
    upsample17 = list(modelyolov5.model.children())[17]
    model.module_list[idx] = upsample17
    idx = idx + 1
    # route
    idx = idx + 1
    cspnet19 = list(modelyolov5.model.children())[19]
    idx = copy_c3(cspnet19, model.module_list, idx,round(3 * depth_multiple), False)
    conv20 = list(modelyolov5.model.children())[20]
    idx = copy_conv_idx(conv20, model.module_list[idx], idx)
    upsample21 = list(modelyolov5.model.children())[21]
    model.module_list[idx] = upsample21
    idx = idx + 1
    # route
    idx = idx + 1
    cspnet23 = list(modelyolov5.model.children())[23]
    idx = copy_c3(cspnet23, model.module_list, idx, round(3 * depth_multiple), False)
    #conv
    conv_detect1_idx=idx
    idx=idx+1
    #yolo
    idx=idx+1
    #route
    idx=idx+1
    conv24 = list(modelyolov5.model.children())[24]
    idx=copy_conv_idx(conv24, model.module_list[idx],idx)
    # route
    idx = idx + 1
    cspnet26 = list(modelyolov5.model.children())[26]
    idx = copy_c3(cspnet26, model.module_list,idx, round(3 * depth_multiple), False)
    # conv
    conv_detect2_idx = idx
    idx = idx + 1
    # yolo
    idx = idx + 1
    # route
    idx = idx + 1
    conv27 = list(modelyolov5.model.children())[27]
    idx=copy_conv_idx(conv27, model.module_list[idx],idx)
    # route
    idx = idx + 1
    cspnet29 = list(modelyolov5.model.children())[29]
    idx = copy_c3(cspnet29, model.module_list, idx,round(3 * depth_multiple), False)
    # conv
    conv_detect3_idx = idx
    idx = idx + 1
    # yolo
    idx = idx + 1
    # route
    idx = idx + 1
    conv30 = list(modelyolov5.model.children())[30]
    idx = copy_conv_idx(conv30, model.module_list[idx], idx)
    # route
    idx = idx + 1
    cspnet32 = list(modelyolov5.model.children())[32]
    idx = copy_c3(cspnet32, model.module_list, idx, round(3 * depth_multiple), False)
    # conv
    conv_detect4_idx = idx
    detect33 = list(modelyolov5.model.children())[33]

    name_list = [x[0] for x in model.module_list[conv_detect1_idx].named_sublayers()]
    if "Conv2D" in name_list:
        model.module_list[conv_detect1_idx]["Conv2D"] = detect33.m[0]
        model.module_list[conv_detect2_idx]["Conv2D"] = detect33.m[1]
        model.module_list[conv_detect3_idx]["Conv2D"] = detect33.m[2]
        model.module_list[conv_detect4_idx]["Conv2D"] = detect33.m[3]
    else:
        model.module_list[conv_detect1_idx][0] = detect33.m[0]
        model.module_list[conv_detect2_idx][0] = detect33.m[1]
        model.module_list[conv_detect3_idx][0] = detect33.m[2]
        model.module_list[conv_detect4_idx][0] = detect33.m[3]

def copy_weight_v6_reverse(modelyolov5,model):
    idx=0
    depth_multiple=0.33
    # if 'depth_multiple' in modelyolov5.yaml:
    #     depth_multiple=modelyolov5.yaml['depth_multiple']
    conv0 = list(modelyolov5.model.children())[0]
    idx=copy_conv_idx_reverse(conv0, model.module_list[idx],idx)
    conv1 = list(modelyolov5.model.children())[1]
    idx=copy_conv_idx_reverse(conv1, model.module_list[idx],idx)
    cspnet2 = list(modelyolov5.model.children())[2]
    idx=copy_c3_reverse(cspnet2,model.module_list,idx, round(3 * depth_multiple))
    conv3 = list(modelyolov5.model.children())[3]
    idx=copy_conv_idx_reverse(conv3, model.module_list[idx],idx)
    cspnet4 = list(modelyolov5.model.children())[4]
    idx = copy_c3_reverse(cspnet4, model.module_list,idx, round(6 * depth_multiple))
    conv5 = list(modelyolov5.model.children())[5]
    idx=copy_conv_idx_reverse(conv5, model.module_list[idx],idx)
    cspnet6 = list(modelyolov5.model.children())[6]
    idx = copy_c3_reverse(cspnet6, model.module_list,idx, round(9 * depth_multiple))
    conv7 = list(modelyolov5.model.children())[7]
    idx=copy_conv_idx_reverse(conv7, model.module_list[idx],idx)
    cspnet8 = list(modelyolov5.model.children())[8]
    idx = copy_c3_reverse(cspnet8, model.module_list,idx, round(3 * depth_multiple))
    sppf9 = list(modelyolov5.model.children())[9]
    idx=copy_conv_idx_reverse(sppf9.cv1, model.module_list[idx],idx)
    sppf9.m=model.module_list[idx]
    idx = idx + 1
    sppf9.m=model.module_list[idx]
    idx = idx + 1
    sppf9.m=model.module_list[idx]
    idx = idx + 1
    #route
    idx = idx + 1
    idx=copy_conv_idx_reverse(sppf9.cv2, model.module_list[idx],idx)
    conv10 = list(modelyolov5.model.children())[10]
    idx=copy_conv_idx_reverse(conv10, model.module_list[idx],idx)
    upsample11 = list(modelyolov5.model.children())[11]
    model.module_list[idx] = upsample11
    idx = idx + 1
    #route
    idx=idx+1
    cspnet13 = list(modelyolov5.model.children())[13]
    idx = copy_c3_reverse(cspnet13, model.module_list, idx,round(3 * depth_multiple),False)
    conv14 = list(modelyolov5.model.children())[14]
    idx=copy_conv_idx_reverse(conv14, model.module_list[idx],idx)
    upsample15 = list(modelyolov5.model.children())[15]
    model.module_list[idx] = upsample15
    idx = idx + 1
    # route
    idx = idx + 1
    cspnet17 = list(modelyolov5.model.children())[17]
    idx = copy_c3_reverse(cspnet17, model.module_list, idx,round(3 * depth_multiple), False)
    #conv
    conv_detect1_idx=idx
    idx=idx+1
    #yolo
    idx=idx+1
    #route
    idx=idx+1
    conv18 = list(modelyolov5.model.children())[18]
    idx=copy_conv_idx_reverse(conv18, model.module_list[idx],idx)
    # route
    idx = idx + 1
    cspnet20 = list(modelyolov5.model.children())[20]
    idx = copy_c3_reverse(cspnet20, model.module_list,idx, round(3 * depth_multiple), False)
    # conv
    conv_detect2_idx = idx
    idx = idx + 1
    # yolo
    idx = idx + 1
    # route
    idx = idx + 1
    conv21 = list(modelyolov5.model.children())[21]
    idx=copy_conv_idx_reverse(conv21, model.module_list[idx],idx)
    # route
    idx = idx + 1
    cspnet23 = list(modelyolov5.model.children())[23]
    idx = copy_c3_reverse(cspnet23, model.module_list, idx,round(3 * depth_multiple), False)
    # conv
    conv_detect3_idx = idx
    detect24 = list(modelyolov5.model.children())[24]

    name_list = [x[0] for x in model.module_list[conv_detect1_idx].named_sublayers()]
    if "Conv2D" in name_list:
        detect24.m[0] = model.module_list[conv_detect1_idx]["Conv2D"]
        detect24.m[1] = model.module_list[conv_detect2_idx]["Conv2D"]
        detect24.m[2] = model.module_list[conv_detect3_idx]["Conv2D"]
    else:
        detect24.m[0] = model.module_list[conv_detect1_idx][0]
        detect24.m[1] = model.module_list[conv_detect2_idx][0]
        detect24.m[2] = model.module_list[conv_detect3_idx][0]

def copy_weight_v6x_reverse(modelyolov5,model):
    idx=0
    depth_multiple=1
    if 'depth_multiple' in modelyolov5.yaml:
        depth_multiple=modelyolov5.yaml['depth_multiple']
    conv0 = list(modelyolov5.model.children())[0]
    idx=copy_conv_idx_reverse(conv0, model.module_list[idx],idx)
    conv1 = list(modelyolov5.model.children())[1]
    idx=copy_conv_idx_reverse(conv1, model.module_list[idx],idx)
    cspnet2 = list(modelyolov5.model.children())[2]
    idx=copy_c3_reverse(cspnet2,model.module_list,idx, round(3 * depth_multiple))
    conv3 = list(modelyolov5.model.children())[3]
    idx=copy_conv_idx_reverse(conv3, model.module_list[idx],idx)
    cspnet4 = list(modelyolov5.model.children())[4]
    idx = copy_c3_reverse(cspnet4, model.module_list,idx, round(6 * depth_multiple))
    conv5 = list(modelyolov5.model.children())[5]
    idx=copy_conv_idx_reverse(conv5, model.module_list[idx],idx)
    cspnet6 = list(modelyolov5.model.children())[6]
    idx = copy_c3_reverse(cspnet6, model.module_list,idx, round(9 * depth_multiple))
    conv7 = list(modelyolov5.model.children())[7]
    idx=copy_conv_idx_reverse(conv7, model.module_list[idx],idx)
    cspnet8 = list(modelyolov5.model.children())[8]
    idx = copy_c3_reverse(cspnet8, model.module_list,idx, round(3 * depth_multiple))
    conv9 = list(modelyolov5.model.children())[9]
    idx = copy_conv_idx_reverse(conv9, model.module_list[idx], idx)
    cspnet10 = list(modelyolov5.model.children())[10]
    idx = copy_c3_reverse(cspnet10, model.module_list, idx, round(3 * depth_multiple))
    sppf11 = list(modelyolov5.model.children())[11]
    idx=copy_conv_idx_reverse(sppf11.cv1, model.module_list[idx],idx)
    sppf11.m = model.module_list[idx]
    idx = idx + 1
    sppf11.m = model.module_list[idx]
    idx = idx + 1
    sppf11.m = model.module_list[idx]
    idx = idx + 1
    #route
    idx = idx + 1
    idx=copy_conv_idx_reverse(sppf11.cv2, model.module_list[idx],idx)
    conv12 = list(modelyolov5.model.children())[12]
    idx=copy_conv_idx_reverse(conv12, model.module_list[idx],idx)
    upsample13 = list(modelyolov5.model.children())[13]
    model.module_list[idx] = upsample13
    idx = idx + 1
    #route
    idx=idx+1
    cspnet15 = list(modelyolov5.model.children())[15]
    idx = copy_c3_reverse(cspnet15, model.module_list, idx,round(3 * depth_multiple),False)
    conv16 = list(modelyolov5.model.children())[16]
    idx=copy_conv_idx_reverse(conv16, model.module_list[idx],idx)
    upsample17 = list(modelyolov5.model.children())[17]
    model.module_list[idx] = upsample17
    idx = idx + 1
    # route
    idx = idx + 1
    cspnet19 = list(modelyolov5.model.children())[19]
    idx = copy_c3_reverse(cspnet19, model.module_list, idx,round(3 * depth_multiple), False)
    conv20 = list(modelyolov5.model.children())[20]
    idx = copy_conv_idx_reverse(conv20, model.module_list[idx], idx)
    upsample21 = list(modelyolov5.model.children())[21]
    model.module_list[idx] = upsample21
    idx = idx + 1
    # route
    idx = idx + 1
    cspnet23 = list(modelyolov5.model.children())[23]
    idx = copy_c3_reverse(cspnet23, model.module_list, idx, round(3 * depth_multiple), False)
    #conv
    conv_detect1_idx=idx
    idx=idx+1
    #yolo
    idx=idx+1
    #route
    idx=idx+1
    conv24 = list(modelyolov5.model.children())[24]
    idx=copy_conv_idx_reverse(conv24, model.module_list[idx],idx)
    # route
    idx = idx + 1
    cspnet26 = list(modelyolov5.model.children())[26]
    idx = copy_c3_reverse(cspnet26, model.module_list,idx, round(3 * depth_multiple), False)
    # conv
    conv_detect2_idx = idx
    idx = idx + 1
    # yolo
    idx = idx + 1
    # route
    idx = idx + 1
    conv27 = list(modelyolov5.model.children())[27]
    idx=copy_conv_idx_reverse(conv27, model.module_list[idx],idx)
    # route
    idx = idx + 1
    cspnet29 = list(modelyolov5.model.children())[29]
    idx = copy_c3_reverse(cspnet29, model.module_list, idx,round(3 * depth_multiple), False)
    # conv
    conv_detect3_idx = idx
    idx = idx + 1
    # yolo
    idx = idx + 1
    # route
    idx = idx + 1
    conv30 = list(modelyolov5.model.children())[30]
    idx = copy_conv_idx_reverse(conv30, model.module_list[idx], idx)
    # route
    idx = idx + 1
    cspnet32 = list(modelyolov5.model.children())[32]
    idx = copy_c3_reverse(cspnet32, model.module_list, idx, round(3 * depth_multiple), False)
    # conv
    conv_detect4_idx = idx
    detect33 = list(modelyolov5.model.children())[33]
    detect33.m[0] = model.module_list[conv_detect1_idx][0]
    detect33.m[1] = model.module_list[conv_detect2_idx][0]
    detect33.m[2] = model.module_list[conv_detect3_idx][0]
    detect33.m[3] = model.module_list[conv_detect4_idx][0]