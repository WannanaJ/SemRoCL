SemRoCL Documentation
====================

**SemRoCL** (Semantic-guided Robust Contrastive Learning) 是一个用于低光照图像增强的两阶段无监督学习框架。

.. toctree::
   :maxdepth: 2
   :caption: 📚 内容目录:

   installation
   quickstart
   api/index
   tutorials/index
   contributing

.. toctree::
   :maxdepth: 1
   :caption: 🔗 链接:

   GitHub Repository <https://github.com/your-username/SemRoCL>
   changelog

项目特点
--------

* 🎯 **无监督学习**: 消除对配对训练数据的依赖
* 🧠 **语义引导**: 利用场景理解进行针对性增强
* 💪 **鲁棒性**: 处理极端光照和噪声条件
* 📊 **可解释性**: 提供可解释的语义热图
* 🚀 **轻量级**: 支持边缘设备部署

快速开始
--------

安装
~~~~

.. code-block:: bash

   pip install -r requirements.txt

训练
~~~~

**Stage 1: 对比学习预训练**

.. code-block:: bash

   cd src
   python train_stage1_moco.py --config ../configs/train_stage1.yaml

**Stage 2: 语义增强训练**

.. code-block:: bash

   python train_stage2_enhanced.py --config ../configs/train_stage2_enhanced.yaml

索引和表格
----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
