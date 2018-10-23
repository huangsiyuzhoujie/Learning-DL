Learning-DeepLearning
====================
slim、keras、tensorlayer学习网址<br>
>https://www.jianshu.com/p/13eda2462214

keras手册<br>
>https://keras-cn.readthedocs.io/en/latest/backend/

tensorflow issue地址<br>
>https://github.com/tensorflow/tensorflow/issues

inception-v3迁移学习例子<br>
>https://blog.csdn.net/gaoyueace/article/details/79222225

tensorflow NLP<br>
>https://github.com/zhedongzheng/finch#word-embedding%E8%AF%8D%E5%90%91%E9%87%8F

奇异ai<br>
>http://ai.loliloli.pro/

各类数据集<br>
>https://github.com/Strange-AI/datasets

21个项目 -- 代码<br>
>https://github.com/hzy46/Deep-Learning-21-Examples

吴恩达 -- 深度学习 -- 机器学习<br>
>https://github.com/fengdu78/deeplearning_ai_books<br>
>https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes

>pytorch-new中是编译好的caffe2.<br>
>tensorflow-new中时编译好的freeze_graph，tensorflow-hsy中是编译好的toco。<br>
>.pb文件，保存的是图模型的计算流程图，包括图中的常量，但不保存变量。<br>
>.ckpt文件，保存的是图模型中的变量的值。<br>
>.lite文件，包含图模型的计算流程图和图模型中的变量的值。<br>
>生成litea文件的方式，通过.pb文件和.ckpt文件，进行图中的变量的固化，再生成.lite文件。<br>
>一般tensorflow保存模型后会得到三个文件，.data,.meta,.index.<br>
>meta文件 -- 保存graphu结构。<br>
>data文件 -- 保存模型所有变量的值。<br>
>index文件 -- 保存变量名。<br>
>(1)depthwise -- 深度可分离卷积网络，减少参数量。一般用于android应用中。<br>

学习中遇到的问题<br>
>(1)import torch 与 import cv2会发生冲突，需要将import torch在import cv2之后导入。<br>
>(2)CUDNN_STATUS_INTERNAL_ERROR 错误的解决办法<br>
>sudo rm -rf ~/.nv<br>
>(3)InvalidArgumentError: Expected image (JPEG, PNG, or GIF), got unknown format starting with '\000\000\000\001Bud1\000\000\020\000\000\000\010\000'<br>
>出现此错误时表示文件夹中含有其他隐藏文件，删除即可。<br>
>(4)为了在pytorch中使用精确除法，需要使用 from __future__ import division.对于整型的tensor需要先转化为numpy类型。<br>
>(5)出现image_size must contain 3 elements[4] 错误的解决方法<br>
>有图片不是RGB格式。<br>
>(6)python2中使用raw_input函数.<br>

>--------------------------------------------------------------------------------------------------------------------------------
Pelee 手机上的分类和目标检测模型<br>
>https://github.com/Robert-JunWang/Pelee

tensorflow 迁移学习实现inception-v3<br>
>https://blog.csdn.net/White_Idiot/article/details/78816850<br>

tensorflow slim fine-tune 分类 移植到android的全过程<br>
>https://blog.csdn.net/u011092156/article/details/80607601<br>
>https://blog.csdn.net/u011092156/article/details/80642133<br>

tensorflow lite ssd 目标检测 博客<br>
>https://segmentfault.com/a/1190000015584764<br>

tensorflow 目标检测移植到android的全过程<br>
>https://github.com/Zod20/TF-Object-Detection-Demo-Daisy<br>
>https://stackoverflow.com/questions/50330184/build-and-run-tensorflow-lite-demo-with-gradle<br>
>https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md<br>
>https://github.com/naisy/train_ssd_mobilenet#6<br>
>https://github.com/tz28/Chinese-number-gestures-recognition

caffe 训练的所有过程<br>
>http://www.cnblogs.com/denny402/tag/caffe/<br>

caffe2 学习<br>
>入门教程<br>
>https://www.cnblogs.com/allcloud/p/7681966.html<br>
>官网教程<br>
>https://caffe2.ai/docs/tutorial-MNIST.html<br>
>博客教程<br>
>https://blog.csdn.net/zziahgf/article/category/6925068/2<br>

pytorch 训练并移植到android<br>
>https://zhuanlan.zhihu.com/p/32342366<br>

>--------------------------------------------------------------------------------------------------------------------------------

博客<br>
>https://www.cnblogs.com/missidiot/archive/2018/07.html<br>
>squeeze与unsqueeze的用法<br>
https://blog.csdn.net/weixin_32365557/article/details/80488965

cuda和cudnn安装<br>
>https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html

训练样本-将xml文件转化为txt文件<br>
>https://blog.csdn.net/shangpapa3/article/details/77483324

可视化训练过程的中间参数<br>
>https://blog.csdn.net/shangpapa3/article/details/76687191<br>
>https://blog.csdn.net/vvyuervv/article/details/72868749

训练样本增强的方法<br>
>https://zhuanlan.zhihu.com/p/23249000<br>
>https://github.com/aleju/imgaug

caffe相关<br>
>安装<br>
>https://www.linuxidc.com/Linux/2017-01/139313p2.htm<br>
>https://www.cnblogs.com/AbcFly/p/6306201.html<br>
caffe实现yolov3<br>
>https://blog.csdn.net/maweifei/article/details/81066578<br>
caffe移植到arm<br>
>https://blog.csdn.net/q6324266/article/details/74563618<br>
>caffe训练测试自己图片教程<br>
>https://www.cnblogs.com/luludeboke/p/7813060.html

pytorch相关<br>
>入门教程<br>
>https://www.jianshu.com/u/5e2b32ff790c<br>
>https://blog.csdn.net/zzlyw/article/category/6527133<br>
>中文文档<br>
>https://ptorch.com/docs/8<br>
>中文教程---很多实例(使用pytorch实现yolov3等)<br>
>http://www.pytorchtutorial.com/<br>
>http://pytorch.apachecn.org/cn/tutorials/beginner/pytorch_with_examples.html 包括60分钟快速入门<br>
>caffe2官方教程<br>
>https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=compile#install-with-gpu-support<br>
>caffe2部署到android的全过程<br>
>https://zhuanlan.zhihu.com/p/32342366<br>
>caffe2的预训练模型地址<br>
https://github.com/caffe2/models<br>
>pytorch相关问题解答社区<br>
>https://github.com/pytorch/pytorch/issues?utf8=%E2%9C%93&q=onnx_caffe2<br>
>pytorch到caffe转换工具<br>
>http://www.pytorchtutorial.com/pytorch-to-caffe/<br>
>pytorch使用已有框架训练模型的总体过程介绍<br>
>https://blog.csdn.net/u014380165/article/details/79222243<br>
>pytorch中个各种数据类型之间的转换<br>
>https://blog.csdn.net/hustchenze/article/details/79154139<br>

码云开源<br>
>https://gitee.com/herensheng/Pytorch2caffe2

yolov3批处理样本--测试<br>
>http://baijiahao.baidu.com/s?id=1601053158190529853&wfr=spider&for=pc

目标检测相关的所有算法<br>
>https://github.com/amusi/awesome-object-detection<br>
>http://baijiahao.baidu.com/s?id=1601053158190529853&wfr=spider&for=pc

/root空间内存不足的解决办法<br>
>https://blog.csdn.net/t765833631/article/details/79031063

cmake安装方法<br>
>https://www.cnblogs.com/freshmen/p/6506053.html

np中axis解释<br>
>https://blog.csdn.net/xiongchengluo1129/article/details/79062991

mace相关--小米深度学习框架<br>
>中文翻译<br>
>https://www.showdoc.cc/web/#/mace?page_id=575412381009217<br>
源码解读<br>
>https://www.jianshu.com/nb/27109636<br>
使用实例<br>
>https://blog.csdn.net/qq_27063119/article/details/81015227<br>
解决mace编译库时bazel的问题<br>
>bazel clean<br>
>删除/home/huangsiyu/.cache/bazel/文件下下的所有东西

tensorflow<br>
>命令行安装tensorflow<br>
>pip install tensorflow-gpu==1.4.0 --user<br>
>tensorflow lite的预训练模型<br>
>https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/models.md<br>
>tensorflow slim预训练模型<br>
>https://github.com/tensorflow/models/tree/master/research/slim<br>
>使用tensorflow在android上实现手势数字识别的例子<br>
>https://github.com/tz28/Chinese-number-gestures-recognition<br>
>tensorflow android 手写数字识别<br>
>https://blog.csdn.net/guyuealian/article/details/79672257<br>
>tensorflow slim的使用介绍<br>
>https://blog.csdn.net/u014061630/article/details/80632736<br>
>tensorflow源码编译<br>
>https://blog.csdn.net/yhily2008/article/details/79967118<br>
>从.pb文件生成quant tflite文件的命令行的顺序<br>
>https://blog.csdn.net/leifengpeng/article/details/78559208<br>
>转化成tflite文件的方法<br>
>https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/toco/g3doc/python_api.md#example-2-export-with-variables<br>
>完整的保存和载入tensorflow模型的过程<br>
>https://blog.csdn.net/tan_handsome/article/details/79303269<br>
>tensorflow slim官方教程<br>
>https://blog.csdn.net/u014061630/article/details/80632736#44-%E5%AE%9E%E4%BE%8B%E5%88%9B%E5%BB%BAvgg%E7%BD%91%E7%BB%9Cworking-example-specifying-the-vgg16-layers<br>
>tensorflow 实现inception-v3 finetune<br>
>https://blog.csdn.net/White_Idiot/article/details/78816850

ubuntu安装qq<br>
>https://www.jianshu.com/p/f38187cdda0f
