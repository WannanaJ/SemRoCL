安装指南
========

系统要求
--------

* Python >= 3.8
* CUDA >= 11.8 (推荐用于GPU加速)
* 8GB+ RAM
* 4GB+ GPU显存 (如果使用GPU)

使用Conda安装
-------------

推荐使用Conda创建独立的Python环境：

.. code-block:: bash

   cd SemRoCL
   conda env create -f environment.yml
   conda activate semrocl

使用pip安装
-----------

也可以使用pip安装依赖：

.. code-block:: bash

   pip install -r requirements.txt

验证安装
--------

运行以下命令验证安装：

.. code-block:: python

   import torch
   import torchvision
   import timm
   import transformers

   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"CUDA version: {torch.version.cuda}")
       print(f"GPU: {torch.cuda.get_device_name(0)}")

开发环境安装
------------

如果你想参与开发或运行测试，需要安装额外的开发依赖：

.. code-block:: bash

   pip install -r requirements-dev.txt

这将安装以下工具：

* **测试**: pytest, pytest-cov
* **代码质量**: pylint, flake8, black, mypy
* **文档**: sphinx, sphinx-rtd-theme

数据集准备
----------

下载并组织数据集到 ``data/`` 目录：

.. code-block:: text

   data/
   ├── LOL-v1/
   │   ├── eval15/
   │   │   ├── high/
   │   │   └── low/
   │   └── our485/
   │       ├── high/
   │       └── low/
   └── LOL-v2/
       ├── Real_captured/
       └── Synthetic/

下载链接:

* `LOL-v1 <https://daooshee.github.io/BMVC2018website/>`_
* `LOL-v2 <https://github.com/flyywh/CVPR-2020-Semi-Low-Light>`_

常见问题
--------

CUDA版本不匹配
~~~~~~~~~~~~~~

如果遇到CUDA版本问题，请安装对应的PyTorch版本：

.. code-block:: bash

   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

内存不足
~~~~~~~~

如果遇到GPU内存不足，可以：

1. 减小batch size
2. 使用轻量级配置
3. 启用梯度累积
