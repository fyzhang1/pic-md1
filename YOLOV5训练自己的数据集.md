# YOLOV5训练自己的数据集

数据集以VOC2007为例

## 一  将voc2007放到根目录下

 **我们通过main函数在 ImageSets这个文件的Main文件下生成四个txt文件**

![QQ截图20220922201500](https://raw.githubusercontent.com/zhangfuyao/pic-md1/main/img/202209222144423.png)



## 二 在根目录下写voc_label.py程序，生成labels文件夹

**voc_label.py 代码** **，要注意的是程序和VOC2007-1数据集文件是在同一目录下的**

```python
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets = ['train', 'test', 'val']
classes = ["yanwu", "fire"]


def convert(size, box):
    dw = 1. / ((size[0]) + 0.1)
    dh = 1. / ((size[0]) + 0.1)
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open('VOC2007-1/Annotations/%s.xml' % (image_id))
    out_file = open('VOC2007-1/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
print(wd)
for image_set in sets:
    if not os.path.exists('VOC2007-1/labels/'):
        os.makedirs('VOC2007-1/labels/')
    image_ids = open('VOC2007-1/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    list_file = open('VOC2007-1/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('VOC2007-1/JPEGImages/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()
```

**顺利生成labels文件夹和三个txt文件**

## 三 在data目录下创建voc.yaml:

```python
train: E:\yolo\yolov5_origin\VOC2007-1\test.txt
val: E:\yolo\yolov5_origin\VOC2007-1\val.txt
nc: 2
names: ["smoke","fire"]
```

**这里的文件路径要注意，是二中与labels同时生成的txt文件**



## 四 在运行train.py时遇到几个问题：

1. ###  OSError: [WinError 1455] 页面文件太小,无法完成操作。 Error loading

   这个最有效的解决方法如下博客：

   [(35条消息) OSError: [WinError 1455\] 页面文件太小,无法完成操作。 Error loading “D:\Anaco_xiaoyue_yan的博客-CSDN博客](https://blog.csdn.net/weixin_43817670/article/details/116748349)

2. ### 使用yolov5时出现“assertionerror:no labels found in */*/*/JPEGImages.cache can not train without labels” 问题，

   解决方案：

   [(35条消息) 一步真实解决AssertionError: train: No labels in /xxx/xxx/xxx/datasets/VOC_To_YOLO/train.cache._蓝胖胖▸的博客-CSDN博客_assertionerror怎么解决](https://blog.csdn.net/Thebest_jack/article/details/125647537)

**把images改为JPEGImages即可**

**再次运行，成功！**