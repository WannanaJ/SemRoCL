API 参考
========

本节包含SemRoCL所有模块的详细API文档。

.. toctree::
   :maxdepth: 2
   :caption: 模块:

   models
   losses
   data
   utils
   metrics

核心模块
--------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   model.encoder_moco
   model.semantic_head
   model.models
   model.models_enhanced
   loss_functions
   data_loader
   utils
   metrics_utils

训练模块
--------

.. autosummary::
   :toctree: _autosummary

   train_stage1_moco
   train_stage2_enhanced
   train_stage2_curriculum
   adaptive_curriculum

评估模块
--------

.. autosummary::
   :toctree: _autosummary

   evaluate
