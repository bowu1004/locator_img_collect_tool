other resources: 
1. https://unitedretail.github.io/Unitail-OCR/

2. SAM识别有无，+MobileNet-v2分类器（需要大量照片）
---


定位器+分类器框架是基于当前开发的双模态RGBD-YOLO训练的识别常见物体的定位器，并结合仅需采集一张图片作为模板可以做到准确分类且无需数据训练的分类器，进一步提升性能和灵活性。设计快速、准确且易于使用，使其成为各种物体检测与跟踪任务的绝佳选择。

我们希望这里的资源能帮助您充分利用定位器+分类器框架。请浏览下述文档了解详细信息，在 <a href="https://github.com/bowu1004/RealMan_AI_Lab">GitHub</a> 上提交问题以获得支持，并加入我们的 <a href="https://github.com/bowu1004/RealMan_AI_Lab/issues">Discord</a> 社区进行问题和讨论！
<img width="100%" height="80%" src="img.png"></a>

## <div align="center">文档</div>

请参阅下面的快速安装和使用示例。

<details open>
<summary>安装</summary>

使用Pip在一个[**Python>=3.8**](https://www.python.org/)环境中安装`ultralytics`包，此环境还需包含[**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)。这也会安装所有必要的[依赖项](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt)。

```bash
pip install ultralytics
pip install -r requirements.txt
```

</details>

<details open>
<summary>Usage</summary>

#### CLI

定位器+定位器 可以在命令行界面（CLI）中直接使用，只需输入如下命令：

```bash
python predict.py
```
#### Python

定位器+定位器 也可以在 Python 环境中直接使用，并接受与上述 CLI 示例中相同的参数。

</details>

## <div align="center">模型</div>

RGBD-Yolo定位器是在[COCO](https://docs.ultralytics.com/datasets/detect/coco)数据集上预训练的模型并迁移学习我们自主采集的生活中常见物品，相关模型权重可以通过[我们提供的google drive](https://drive.google.com/file/d/1d1AqtlIJZSzG1Zgph3EEhL1RGZ4G0gZ2/view?usp=drive_link)进行下载并解压至./CD_pth目录下。分类器支持所有的检测，分割模型的集成。
<details open><summary>详细参数</summary>

| 操作系统      | 定位器推理速度   | 分类器推理速度 | 
|-----------|-----------|---------|
| [Windows] | 0.068s       | 0.339s     |
| [NX]      | 0.58s      | 2.17s    |
</details>

