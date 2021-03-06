#!/usr/bin/env python
# -*- coding: utf8 -*-
#import _init_path
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import pdb


class PascalVocWriter:

    def __init__(self, foldername, filename, imgSize, databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True)

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None or \
                len(self.boxlist) <= 0:
            return None

        top = Element('annotation')
        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        localImgPath = SubElement(top, 'path')
        localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def addBndBox(self, xmin, ymin, xmax, ymax, name, pose = 'Unspecified'):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox['pose'] = pose
        print bndbox
        self.boxlist.append(bndbox)

    def appendObjects(self, top):
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            name.text = str(each_object['name'])
            pose = SubElement(object_item, 'pose')
            pose.text = each_object['pose']
            truncated = SubElement(object_item, 'truncated')
            truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = "0"
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = open(self.filename + '.xml', 'w')
        else:
            out_file = open(targetFile, 'w')

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult)
        out_file.close()


class PascalVocReader:

    def __init__(self, filepath):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color]
        self.shapes = []
        self.img_shape = (0, 0, 0)
        self.filepath = filepath
        self.parseXML()

    def getShapes(self):
        return self.shapes

    def addShape(self, label, rect):
        xmin = rect[0]
        ymin = rect[1]
        xmax = rect[2]
        ymax = rect[3]
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        self.shapes.append((label, points, None, None))

    def getSize(self):
        return self.img_shape

    def parseXML(self):
        assert self.filepath.endswith('.xml'), "Unsupport file format"
        parser = etree.XMLParser(encoding='utf-8')
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        filename = xmltree.find('filename').text

        size = xmltree.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        depth = int(size.find('depth').text)
        self.img_shape = (height, width, depth)

        for object_iter in xmltree.findall('object'):
            rects = []
            bndbox = object_iter.find("bndbox")
            rects.append([int(it.text) for it in bndbox])
            label = object_iter.find('name').text

            for rect in rects:
                self.addShape(label, rect)
        return True



#tempParseReader = PascalVocReader('/tmp/test.xml')
#shapes = tempParseReader.getShapes()
#size = tempParseReader.getSize()
## Test
#tmp = PascalVocWriter('/tmp','write', size)
#for i in shapes:
#  tmp.addBndBox(i[1][0][0], i[1][0][1], i[1][2][0], i[1][2][1], i[0])
##tmp.addBndBox(10,10,20,30,'chair')
##tmp.addBndBox(1,1,600,600,'car')
#tmp.save('/tmp/write.xml')
