快速开始
========

本指南将帮助你快速运行SemRoCL进行低光照图像增强。

基本工作流程
------------

SemRoCL采用两阶段训练策略：

1. **Stage 1**: MoCo v3对比学习预训练
2. **Stage 2**: 语义引导的图像增强

Stage 1: 对比学习预训练
-----------------------

在低光照图像上进行无监督特征学习：

.. code-block:: bash

   cd src
   python train_stage1_moco.py --config ../configs/train_stage1.yaml

**主要特性:**

* 无监督对比学习
* 动态负样本队列
* 光照和噪声增强
* 输出: ``outputs/moco_pretrain_stage1/checkpoints/moco_pretrain_final.pth``

轻量级训练
~~~~~~~~~~

如需快速迭代，可使用轻量级配置：

.. code-block:: bash

   python train_stage1_moco.py --config ../configs/train_stage1_light.yaml

* ResNet-18 backbone (~45% 更快)
* 更小的队列大小和图像分辨率

Stage 2: 语义增强训练
---------------------

使用语义引导和自适应课程学习进行增强：

.. code-block:: bash

   python train_stage2_enhanced.py --config ../configs/train_stage2_enhanced.yaml

**主要特性:**

* 多尺度增强
* SegFormer-B0语义引导
* 自适应课程学习
* 多目标优化

优化配置
~~~~~~~~

使用优化配置可获得更快的训练速度：

.. code-block:: bash

   python train_stage2_enhanced.py --config ../configs/train_stage2_enhanced_optimized.yaml

   # 或使用批处理脚本
   ..\scripts\resume_training_optimized.bat

* 30-40% 更快的数据加载
* 增强的色彩损失权重
* 全面的指标记录

评估模型
--------

在测试集上评估训练好的模型：

.. code-block:: bash

   python evaluate.py \
       --config ../configs/train_stage2_enhanced.yaml \
       --checkpoint ../outputs/semantic_enhancement_stage2_ENHANCED/checkpoints/best_model.pth

**计算的指标:**

* PSNR (峰值信噪比)
* SSIM (结构相似性)
* LPIPS (感知相似性)
* DeltaE (色彩差异)
* NIQE (自然图像质量)
* BRISQUE (无参考质量评估)

使用预训练模型
--------------

加载和使用预训练模型进行推理：

.. code-block:: python

   import torch
   from model.models_enhanced import EnhancementGenerator
   from model.semantic_head import SemanticGuidanceModule

   # 加载模型
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   generator = EnhancementGenerator(config).to(device)
   semantic_module = SemanticGuidanceModule(num_classes=19).to(device)

   # 加载权重
   checkpoint = torch.load('path/to/checkpoint.pth')
   generator.load_state_dict(checkpoint['generator'])
   semantic_module.load_state_dict(checkpoint['semantic_head'])

   # 推理
   generator.eval()
   semantic_module.eval()

   with torch.no_grad():
       semantic_features, confidence = semantic_module(low_img)
       enhanced = generator(low_img, semantic_features, confidence)

图像质量优化
------------

项目包含自动图像后处理以减少伪影：

.. code-block:: python

   from utils import save_image

   # 应用所有优化
   save_image(tensor, 'output.png', quality=95, apply_post_processing=True)

   # 禁用后处理（更快但质量较低）
   save_image(tensor, 'output.png', apply_post_processing=False)

监控训练
--------

使用TensorBoard监控训练进度：

.. code-block:: bash

   tensorboard --logdir outputs/semantic_enhancement_stage2_ENHANCED/logs

或使用提供的监控脚本：

.. code-block:: bash

   ..\scripts\monitor_training.bat

下一步
------

* 查看 :doc:`api/index` 了解详细的API文档
* 阅读 :doc:`tutorials/index` 学习高级用法
* 参考 :doc:`contributing` 了解如何贡献代码
