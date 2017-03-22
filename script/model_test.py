#!/usr/bin/env python

import _init_paths
import caffe, os, sys, cv2, math
from fast_rcnn.test import im_detect
from fast_rcnn.config import cfg, CLASSES
from fast_rcnn.nms_wrapper import nms
from math import pow,sqrt
from pascal_voc_io import PascalVocWriter, PascalVocReader
import argparse
import numpy as np

DEBUG = False
#DEBUG = True
class_score = {}

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--input', dest='input_dir', help='input dir(with two sub-dir img and annotation)',
                        type=str)
    parser.add_argument('--output', dest='output_dir', help='output dir, output label file(xml)',
                        type=str)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]')
    parser.add_argument('--model', dest='model', help='model to use')

    args = parser.parse_args()

    return args

def predict_label(net, im_file, thresh = 0.8):
    # Load image
    im = cv2.imread(im_file)

    scores, boxes = im_detect(net, im)

    im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    CONF_THRESH = thresh
    NMS_THRESH = 0.3
    label_num = 0

    shapes = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
                           #class  xmin  ymin  xmax  ymax
            shapes.append([cls, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
            label_num += 1
        #vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)
    return shapes


def convert_shapes(shapes):
    labels = []
    for i in shapes:
        labels.append([i[0],i[1][0][0], i[1][0][1], i[1][2][0], i[1][2][1]])
    return labels

def distance(label_a, label_b):
    a_center = ((label_a[3] - label_a[1])/2.0 + label_a[1], (label_a[4] - label_a[2])/2.0 + label_a[2])
    b_center = ((label_b[3] - label_b[1])/2.0 + label_b[1], (label_b[4] - label_b[2])/2.0 + label_b[2])
    #if DEBUG:  print a_center, b_center
    return sqrt(pow(a_center[0] - b_center[0], 2) + pow(a_center[1] - b_center[1], 2))

def label2dict(label_list):
    d = {}
    for i in label_list:
        if i[0] not in d:
            d[i[0]] = []
        d[i[0]].append(i)
    return d
    
def compare_label(true_labels, pre_labels, grid = 25.0):
    """find every true label in predict_label"""
    
    total_score = 4.0 * len(true_labels)
    score = 0.0

    #print true_labels
    #print pre_labels
    # store the same class name together
    t_dict = label2dict(true_labels)
    p_dict = label2dict(pre_labels)
    #print t_dict
    #print p_dict

    for cls in t_dict.iterkeys():
        #print "##### match for class {0}".format(cls)
        if cls not in class_score:
            class_score[cls] = [0.0, 0.0]
        t_list = t_dict[cls]
        if cls not in p_dict:
            s = - 4.0 * len(t_list)
            score += s
            class_score[cls][0] += s
            class_score[cls][1] += 4.0 * len(t_list)
            if DEBUG: print "not paired,score - {0}".format(4.0 * len(t_list)),t_list, score,total_score
            continue

        p_list = p_dict[cls]

        while len(t_list) > 0 and len(p_list) > 0: # pair true label with predict label one by one
            tp_pair = None
            pair_dis = 0.0
            for t in t_list:
                for p in p_list:
                    if tp_pair is None:
                        tp_pair = (t,p)
                        pair_dis = distance(t, p)
                    else:
                        tmp_dis = distance(t, p)
                        if tmp_dis < pair_dis: #
                            #print "change tp pair"
                            tp_pair = (t,p)
                            pair_dis = tmp_dis
                    #print t,p, tp_pair, pair_dis
            if pair_dis <= grid * 0.5:
                s = 4.0
            elif pair_dis <= grid:
                s = 3.0
            elif pair_dis <= grid * 1.5:
                s = 2.0
            elif pair_dis <= grid * 2.0:
                s = 1.0
            else :
                s = -4.0
            score += s
            class_score[cls][0] += s
            class_score[cls][1] += 4.0
            if DEBUG: print "paired",tp_pair, pair_dis, s, score,total_score
            t_list.remove(tp_pair[0])
            p_list.remove(tp_pair[1])
        if len(t_list) > 0 : #not paired true label:
            s = -4.0 * len(t_list)
            score += s
            class_score[cls][0] += s
            class_score[cls][1] += 4.0 * len(t_list)
            if DEBUG: print "not paired,score - {0}".format(4.0 * len(t_list)),t_list, score,total_score
        if len(p_list) > 0 : #mis-predicted label:
            s = -4.0 * len(p_list)
            score += s 
            class_score[cls][0] += s
            class_score[cls][1] += 4.0 * len(p_list)
            if DEBUG: print "predict error, score - {0}".format(4.0 * len(p_list)),p_list, score,total_score
    if DEBUG: print "compare label out:",score,total_score
    return total_score,score
          
                

def model_test(net, test_list):
    total_score = 0.0
    score = 0.0
    for im_name in test_list:
        if DEBUG: print "############## scoring {0} ##############".format(im_name)
        im_file = os.path.join(input_dir, 'img', im_name)

        label_file = os.path.join(input_dir, 'annotation', im_name)
        index = label_file.rindex('.')
        label_file = label_file[:index] + ".xml"

        if not os.path.isfile(label_file):
            sys.stderr.write("can't find label file for {0}".format(im_file))
            continue
           
        label_reader = PascalVocReader(label_file)
        size = label_reader.getSize()
        shapes = label_reader.getShapes()
        true_label = convert_shapes(shapes)
        
        pre_label = predict_label(net, im_file)

        t_s,s = compare_label(true_label, pre_label)
        print "scoring {0}".format(im_name), t_s, s, s/t_s 
        total_score += t_s
        score += s
    print "total_score", score, total_score,score/total_score
    return total_score,score

def compare_label_test():
    print "test case 1"
    t = []
    t.append(["giant",100,100, 150,150])
    p = []
    p.append(["giant",95,101, 140,152])
    print compare_label(t,p)

    print "test case 2"
    t = []
    t.append(["giant",100,100, 150,150])
    p = []
    p.append(["giant",80,101, 120,152])
    p.append(["giant",95,101, 140,152])
    print compare_label(t,p)

    print "test case 3"
    t = []
    t.append(["giant",100,100, 150,150])
    t.append(["giant",80,101, 120,152])
    p = []
    p.append(["giant",95,101, 140,152])
    print compare_label(t,p)

    print "test case 4"
    t = []
    t.append(["giant",100,100, 150,150])
    p = []
    p.append(["giant",90,101, 120,152])
    print compare_label(t,p)

    print "test case 5"
    t = []
    t.append(["giant",100,100, 150,150])
    t.append(["giant",90,101, 120,152])

    t.append(["prince",200,200, 250,250])
    t.append(["printce",190,201, 220,252])
    p = []
    p.append(["giant",90,101, 120,152])
    print compare_label(t,p)

    print "test case 6"
    t = []
    t.append(["giant",100,100, 150,150])
    t.append(["giant",90,101, 120,152])

    t.append(["prince",200,200, 250,250])
    t.append(["printce",190,201, 220,252])
    p = []
    p.append(["giant",90,101, 120,152])
    p.append(["prince",90,101, 120,152])
    print compare_label(t,p)

if __name__ == '__main__':
    #compare_label_test()
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    input_dir =  args.input_dir
    if not (os.path.isdir(input_dir + '/annotation') and os.path.isdir(input_dir + '/img')):
        raise IOError('input dir must have two dirs: img and annotation')


    caffemodel = args.model
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
    prototxt = 'model/test.prototxt'
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    im_names = os.listdir(input_dir+"/img")

    a,b = model_test(net, im_names)
    print a,b,b/a
    for i in class_score.iterkeys():
        print i, class_score[i], class_score[i][0]/class_score[i][1]
