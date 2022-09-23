# YOLOv5

## detect.py

### weights

```python
parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
```

**yolov5s.pt 还有 yolo5m.pt 等等可以去githup上面查找，******训练自己的数据集时先训练然后会在run中从产生一个.pt文件然后在放到这里面****

### source

```python
parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
```

其中   default=ROOT / 'data/images',是检测文件图片的路径，现在是指定一个文件夹，也可以指定一张图片的路径，也可以检测视频（也是放视频的路径或放在文件夹下）。

### image-size

```python
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
```

在训练的过程中对图片的尺寸进行缩放，但是输入和输出图片的尺寸是不变的

### conf-thres 置信度

```python
parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
```

设置的预值 default=0.25,如果大于这个预值才显示出来。

### IOU 置信度

```python
parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
```

iou=预测和实际的交集/预测和实际的并集，如果iou小于这个设定值就当作两个对象处理。

### device

```python
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
```

### view-img（已加）

```python
parser.add_argument('--view-img', action='store_true', help='show results')
```

可以实时看到检测的结果，比如检测一个视频就能动态看到检测的过程

**使用方法：**

1.看右上角的点击detect中的 Edit Configuration

2.在parameters中输入 --view-img 启动这个参数

（目前我已经开着）

### save

```python
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
```

有好几个save--都差不多跟保存相关

### class

```python
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
```

### agnostic-nms，augment（未加）

```python
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
```

增强结果，影响不大

### update（不用管）

```python
parser.add_argument('--update', action='store_true', help='update all models')
```

把网络模型当时一些不必要的部分去掉

### project

```python
parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
```

把生成的结果保存在什么地方

### exist（未加）

```python
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
```

这个作用是，因为每次生成的结果都会新创建一个文件exp1，exp2，exp3........，加上这个参数后就不在生成新的保存在最新的那个里面





## train.py

### weigths

```python
parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
```

**指的是训练好的网络模型，用来初始化网络权重**

### cfg

```python
parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
```

**为configuration的缩写，指的是网络结构，一般对应models文件夹下的xxx.yaml文件**

### data

```python 
parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
```

**训练数据路径，一般为data文件夹下的xxx.yaml文件**

### hyp

```python 
parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
```

**训练网络的一些超参数设置，(一般用不到)**

### epochs

```python 
parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
```

**设置训练的轮数**

### batch-size

```python
parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
```

**每次输出给神经网络的图片数**

### img-size

```python 
parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
```



### rect： 是否采用矩形训练

```python 
parser.add_argument('--rect', action='store_true', help='rectangular training')
```

### resume： 指定之前训练的网络模型，并继续训练这个模型

```python 
parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
```

### nosave： 只保留最后一次的网络模型

```python 
parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
```

### noval：只在最后一次进行验证

```python 
parser.add_argument('--noval', action='store_true', help='only validate final epoch')
```

### noautoanchor：是否采用锚点

```python 
parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
```

### evolve：是否寻找最优参数

```python 
parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
```

### bucket：这个参数是 yolov5 作者将一些东西放在谷歌云盘，可以进行下载

```python 
parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
```

### cache-images：是否对图片进行缓存，可以加快训练

```python 
parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
```

### image-weights：测试过程中，图像的那些测试地方不太好，对这些不太好的地方加权重

```python 
parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
```

### device：训练网络的设备cpu还是gpu

### multi-scale：训练过程中对图片进行尺度变换

```python 
parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
```

### single-cls：训练数据集是单类别还是多类别

```python 
parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
```

### sync-bn：生效后进行多 GPU 进行分布式训练

```python
parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
```

### local_rank：DistributedDataParallel 单机多卡训练，一般不改动

### workers: 多线程训练

```python 
parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
```

### project：训练结果保存路径

```python 
parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
```

### entity：wandb 库对应的东西，作用不大，不必考虑

### name： 训练结果保存文件名

### exist-ok： 覆盖掉上一次的结果，不新建训练结果文件

### quad：在dataloader时采用什么样的方式读取我们的数据

```python 
parser.add_argument('--quad', action='store_true', help='quad dataloader')
```

### upload_dataset：不必考虑

### bbox_interval：不必考虑

### save_period：用于记录训练日志信息，int 型，默认为 -1

### label-smoothing： 对标签进行平滑处理，防止过拟合

```python 
parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
```

### freeze：冻结哪些层，不去更新训练这几层的参数

```python 
parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
```

### save-period：训练多少次保存一次网络模型

```python 
parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
```

### artifact_alias：忽略即可

