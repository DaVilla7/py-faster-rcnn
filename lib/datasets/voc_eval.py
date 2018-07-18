# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
# <annotation verified="no">
#   <folder>dataImageDeme1-748</folder>
#   <filename>000036.jpg</filename>
#   <path>/home/mqxwd68/Documents/datasetPre/sonarDemo1-748/dataImageDeme1-748/000036.jpg</path>
#   <source>
#     <database>Unknown</database>
#   </source>
#   <size>
#     <width>224</width>
#     <height>224</height>
#     <depth>3</depth>
#   </size>
#   <segmented>0</segmented>
#   <object>
#     <name>corpse</name>
#     <pose>Unspecified</pose>
#     <truncated>0</truncated>
#     <difficult>0</difficult>
#     <bndbox>
#       <xmin>9</xmin>
#       <ymin>10</ymin>
#       <xmax>217</xmax>
#       <ymax>125</ymax>
#     </bndbox>
#   </object>
# </annotation>
import xml.etree.ElementTree as ET
import os
import cPickle
import numpy as np

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    #xml.etree.ElementTree为xml的python接口，可以以树的形式组织xml文件
    #这里只读取了object相关的参数
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
        计算AP值，若use_07_metric=true,则用11个点采样的方法，将rec从0-1分成11个点，这些点prec值求平均近似表示AP
    若use_07_metric=false,则采用更为精确的逐点积分方法
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations#缓存路径
    [ovthresh]: Overlap threshold (default = 0.5)#重叠阈值
    [use_07_metric]: Whether to use VOC07's 11 point AP computation#11个采样点的VOC07的AP计算方法
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    #图像名称list
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            #parse_rec函数读取当前图像标注文件，返回当前图像标注，存于recs字典（key是图像名，values是gt）
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
             #recs字典c保存到只读文件
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            #如果已经有了只读文件，加载到recs
            recs = cPickle.load(f)

    # extract gt objects for this class
    #按类别获取标注文件，recall和precision都是针对不同类别而言的，AP也是对各个类别分别算的
    class_recs = {} #当前类别的标注
    npos = 0 #npos标记的目标数量
    for imagename in imagenames:
        #R代表imagename中指定类别的项（过滤）
        #R的数量代表gt中标记的目标物的数量
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        #bbox代表该类目标物对应的bbox(两个点的坐标，四个值)
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
         #len(R)就是当前类别的gt目标个数，det表示是否检测到，初始化为false
        det = [False] * len(R)
        #自增，非difficult样本数量，如果数据集没有difficult，npos数量就是gt数量
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    # 从此处开始读取目标检测的结果
    detfile = detpath.format(classname)
    #检测检测结果的数据格式：[image_id,confidence,[bbox]]*n,n为检测结果的数量（检测到的区域块的数量）
    with open(detfile, 'r') as f:
        lines = f.readlines()
    #假设检测结果有20000个，则splitlines长度20000
    splitlines = [x.strip().split(' ') for x in lines]
    #检测结果中的图像名，image_ids长度20000，但实际图像只有1000张，因为一张图像上可以有多个目标检测结果
    image_ids = [x[0] for x in splitlines]
     #检测结果置信度
    confidence = np.array([float(x[1]) for x in splitlines])
    #变为浮点型的bbox
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    #对confidence的index根据值大小进行降序排列
    sorted_ind = np.argsort(-confidence)
    #置信度本身按照降序排列
    sorted_scores = np.sort(-confidence)
    #bbox 按照置信度排序
    BB = BB[sorted_ind, :]
    #image_id 按照置信度排序
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    #判断为正确的结果中包括tp和fp
    nd = len(image_ids)
    tp = np.zeros(nd)#true positie
    fp = np.zeros(nd)#false positive
    for d in range(nd):
        #遍历所有检测结果，因为已经排序，所以这里是从置信度最高到最低遍历
        #当前检测结果所在图像的所有同类别gt
        R = class_recs[image_ids[d]]
        #当前检测结果bbox坐标
        bb = BB[d, :].astype(float)
        #设置重合的最大值初始值：负无穷
        ovmax = -np.inf
        #label中的groundtruth的bbox #当前检测结果所在图像的所有同类别gt的bbox坐标
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            # compute overlaps 计算当前检测结果，与该检测结果所在图像的标注重合率，一对多用到python的broadcast机制
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            #重合部分面积
            inters = iw * ih

            # union
            #union是两个图像并集，即两个区域相加减去重合部分
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            #重合率
            overlaps = inters / uni
            #最大重合率
            ovmax = np.max(overlaps)
            #最大重合率对应的gt
            jmax = np.argmax(overlaps)
           
        #如果当前检测结果与真实标注最大重合率满足阈值
        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    #tp数目+1
                    tp[d] = 1.
                    #该gt被置为已检测到，下一次若还有另一个检测结果与之重合率满足阈值，则不能认为多检测到一个目标
                    R['det'][jmax] = 1
                else:
                    #fp数目+1
                    fp[d] = 1.
        else:
            #不满足阈值，肯定是虚警
            fp[d] = 1.

    # compute precision recall
    #积分图，在当前节点前的虚警数量，fp长度
    fp = np.cumsum(fp)
    #积分图，在当前节点前的正检数量
    tp = np.cumsum(tp)
    #召回率 正检数量 / 指定类别中的gt数量
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    #精确率 正检数量 / 总检测到的非虚警数量（即tp + fp）
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    
    #准确率 = PR曲线积分
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
