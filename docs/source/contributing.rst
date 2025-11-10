贡献指南
========

感谢您对SemRoCL项目的兴趣！我们欢迎各种形式的贡献。

开发环境设置
------------

1. Fork并克隆仓库：

.. code-block:: bash

   git clone https://github.com/your-username/SemRoCL.git
   cd SemRoCL

2. 创建虚拟环境并安装依赖：

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt

3. 安装pre-commit hooks：

.. code-block:: bash

   pip install pre-commit
   pre-commit install

贡献流程
--------

1. 创建新分支
~~~~~~~~~~~~~

.. code-block:: bash

   git checkout -b feature/your-feature-name

2. 进行修改
~~~~~~~~~~~

* 遵循代码风格指南
* 添加测试覆盖新功能
* 更新相关文档

3. 运行测试
~~~~~~~~~~~

.. code-block:: bash

   make test
   make lint

4. 提交代码
~~~~~~~~~~~

.. code-block:: bash

   git add .
   git commit -m "Add: your feature description"
   git push origin feature/your-feature-name

5. 创建Pull Request
~~~~~~~~~~~~~~~~~~~

* 在GitHub上创建PR
* 描述你的更改
* 关联相关Issue

代码风格
--------

我们使用以下工具确保代码质量：

* **Black**: 代码格式化 (120字符行宽)
* **isort**: Import排序
* **Flake8**: 代码检查
* **Pylint**: 代码质量
* **mypy**: 类型检查

格式化代码：

.. code-block:: bash

   make format

检查代码质量：

.. code-block:: bash

   make lint

测试指南
--------

编写测试
~~~~~~~~

* 单元测试放在 ``tests/unit/``
* 集成测试放在 ``tests/integration/``
* 使用pytest fixtures for共享设置
* 测试函数名应清楚描述测试内容

运行测试：

.. code-block:: bash

   # 运行所有测试
   make test

   # 运行特定测试
   pytest tests/unit/test_utils.py::TestTensorToImage::test_basic_conversion

   # 生成覆盖率报告
   make test-cov

文档贡献
--------

更新文档
~~~~~~~~

1. 编辑 ``docs/source/`` 中的 .rst 文件
2. 构建文档查看效果：

.. code-block:: bash

   make docs
   make docs-serve  # 本地预览

3. 确保所有公共API都有docstring

Docstring格式
~~~~~~~~~~~~~

使用Google风格的docstring：

.. code-block:: python

   def function_name(param1: str, param2: int) -> bool:
       \"\"\"
       Brief description of function.

       Longer description with details about what the function does.

       Args:
           param1: Description of param1
           param2: Description of param2

       Returns:
           Description of return value

       Raises:
           ValueError: When param1 is empty

       Example:
           >>> result = function_name("test", 42)
           >>> print(result)
           True
       \"\"\"
       pass

提交信息规范
------------

使用语义化的提交信息：

* ``Add:`` 新功能
* ``Fix:`` Bug修复
* ``Update:`` 更新现有功能
* ``Refactor:`` 重构代码
* ``Docs:`` 文档更新
* ``Test:`` 测试相关
* ``Chore:`` 构建/工具更改

示例：

.. code-block:: text

   Add: semantic-guided enhancement module

   - Implement SegFormer-B0 integration
   - Add confidence masking
   - Include unit tests

   Closes #123

报告问题
--------

使用Issue模板报告：

* Bug报告
* 功能请求
* 性能问题
* 文档改进

提供以下信息：

1. 问题描述
2. 复现步骤
3. 期望行为
4. 实际行为
5. 环境信息（Python版本，GPU等）
6. 相关日志

代码审查
--------

所有PR都需要代码审查。审查者会检查：

* 代码质量和可读性
* 测试覆盖
* 文档完整性
* 性能影响
* 向后兼容性

行为准则
--------

* 尊重所有贡献者
* 建设性反馈
* 欢迎新手
* 专注技术讨论

获得帮助
--------

* 查看 `文档 <https://semrocl.readthedocs.io>`_
* 搜索现有Issues
* 在Discussion区提问
* 联系维护者

许可证
------

通过贡献代码，您同意您的贡献将按照项目的MIT许可证进行许可。
