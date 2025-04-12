# object_detection
目标检测
```text
mubiaojiance/
├── config/                 # 配置文件目录
│   ├── labels_map.txt     # 标签映射文件
│   └── coco_labels.txt    # COCO数据集标签
├── data/                  # 数据目录
│   ├── annotations_trainval2017/  # 数据集标注
│   ├── data_process_eff/  # 数据处理相关
│   ├── sample_video.mp4   # 测试视频
│   └── xiongmao.png      # 测试图片
├── models/                # 模型目录
│   ├── efficientdet-d7x/
│   ├── efficientnet_pytorch/
│   └── ssd_mobilenet_v2_320x320_coco17_tpu-8/
├── src/                  # 源代码目录
│   ├── object_detection.py  # 主要实现文件
│   ├── setup.py          # 项目安装配置
│   ├── hubconf.py        # PyTorch Hub配置
│   ├── examples/         # 示例代码
│   └── tests/           # 测试代码
├── tf_to_pytorch/        # 模型转换工具
├── .venv/               # Python虚拟环境
└── requirements.txt     # 项目依赖
