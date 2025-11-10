# 🛠️ SemRoCL 开发基础设施完整指南
# SemRoCL Development Infrastructure Complete Guide

**版本**: v1.1.0
**更新日期**: 2025-11-10
**状态**: ✅ 全部完成

---

## 📑 目录 / Table of Contents

1. [总览 / Overview](#总览--overview)
2. [快速开始 / Quick Start](#快速开始--quick-start)
3. [测试框架 / Testing Framework](#测试框架--testing-framework)
4. [文档系统 / Documentation System](#文档系统--documentation-system)
5. [CI/CD 自动化 / CI/CD Automation](#cicd-自动化--cicd-automation)
6. [代码质量工具 / Code Quality Tools](#代码质量工具--code-quality-tools)
7. [配置文件详解 / Configuration Details](#配置文件详解--configuration-details)
8. [开发工作流 / Development Workflow](#开发工作流--development-workflow)
9. [故障排除 / Troubleshooting](#故障排除--troubleshooting)
10. [最佳实践 / Best Practices](#最佳实践--best-practices)

---

## 📊 总览 / Overview

### 🎯 项目目标 / Project Goals

本指南涵盖了为 SemRoCL 项目建立的完整开发基础设施，包括：

- ✅ **完整的测试框架** - 21个单元测试，覆盖核心模块
- ✅ **专业文档系统** - Sphinx + ReadTheDocs 主题
- ✅ **CI/CD 自动化** - GitHub Actions 多平台测试
- ✅ **代码质量保证** - Flake8, Pylint, Black, isort, mypy
- ✅ **开发工具集成** - Makefile 快捷命令

### 📈 统计数据 / Statistics

| 类别 Category | 新增文件 Files Added | 测试数量 Test Count | 文档页面 Doc Pages |
|--------------|---------------------|--------------------|--------------------|
| **测试文件 Tests** | 5个 | 21个测试 | - |
| **文档文件 Docs** | 8个 | - | 7个页面 |
| **CI/CD 配置 CI/CD** | 3个工作流 | - | - |
| **质量配置 Quality** | 6个配置文件 | - | - |
| **开发工具 DevTools** | 2个 (Makefile, requirements-dev.txt) | - | - |
| **📦 总计 Total** | **24个文件** | **21个测试** | **7个文档页** |

### 📁 完整目录结构 / Complete Directory Structure

```
SemRoCL/
├── 📂 tests/                          # 测试框架 / Testing Framework
│   ├── __init__.py                    # 测试包初始化 / Test package init
│   ├── conftest.py                    # Pytest配置和fixtures / Pytest config & fixtures
│   ├── README.md                      # 测试使用指南 / Testing guide
│   ├── unit/                          # 单元测试 / Unit tests
│   │   ├── __init__.py
│   │   ├── test_utils.py              # 工具函数测试(9个) / Utils tests (9 tests)
│   │   └── test_metrics.py            # 评估指标测试(12个) / Metrics tests (12 tests)
│   ├── integration/                   # 集成测试 / Integration tests (待扩展)
│   └── fixtures/                      # 测试数据 / Test fixtures
│
├── 📂 docs/                           # 文档系统 / Documentation System
│   ├── Makefile                       # 文档构建脚本 / Doc build script
│   └── source/
│       ├── conf.py                    # Sphinx配置 / Sphinx configuration
│       ├── index.rst                  # 文档首页 / Documentation home
│       ├── installation.rst           # 安装指南 / Installation guide
│       ├── quickstart.rst             # 快速开始 / Quick start
│       ├── contributing.rst           # 贡献指南 / Contributing guide
│       ├── changelog.rst              # 更新日志 / Changelog
│       └── api/
│           └── index.rst              # API文档索引 / API reference
│
├── 📂 .github/workflows/              # CI/CD 自动化 / CI/CD Automation
│   ├── tests.yml                      # 自动测试工作流 / Automated testing
│   ├── code-quality.yml               # 代码质量检查 / Code quality checks
│   └── docs.yml                       # 文档构建部署 / Docs build & deploy
│
├── 📄 配置文件 / Configuration Files
│   ├── pytest.ini                     # Pytest配置 / Pytest configuration
│   ├── pyproject.toml                 # 现代Python项目配置 / Modern Python config
│   ├── .flake8                        # Flake8配置 / Flake8 configuration
│   ├── .pylintrc                      # Pylint配置 / Pylint configuration
│   ├── Makefile                       # 快捷命令 / Shortcut commands
│   └── requirements-dev.txt           # 开发依赖 / Development dependencies
│
└── 📄 文档文件 / Documentation Files
    ├── DEV_INFRASTRUCTURE_GUIDE.md    # 本文档 / This guide
    ├── TESTING_AND_DOCS_SETUP.md      # 设置总结 / Setup summary
    └── CHANGELOG.md                   # 项目变更日志 / Project changelog
```

---

## 🚀 快速开始 / Quick Start

### 1️⃣ 安装开发环境 / Install Development Environment

```bash
# 克隆项目 / Clone repository
git clone https://github.com/your-username/SemRoCL.git
cd SemRoCL

# 方法1: 使用 Makefile (推荐 / Recommended)
make install-dev

# 方法2: 手动安装 / Manual installation
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2️⃣ 验证安装 / Verify Installation

```bash
# 运行测试 / Run tests
make test

# 检查代码质量 / Check code quality
make lint

# 构建文档 / Build documentation
make docs
```

### 3️⃣ 开发工作流 / Development Workflow

```bash
# 1. 编写代码 / Write code
vim src/your_module.py

# 2. 格式化代码 / Format code
make format

# 3. 运行测试 / Run tests
make test

# 4. 检查质量 / Check quality
make lint

# 5. 提交代码 / Commit code
git add .
git commit -m "Add: your feature"
git push
```

---

## 🧪 测试框架 / Testing Framework

### 📦 已安装的测试工具 / Installed Testing Tools

```txt
pytest==7.4.3                    # 测试框架 / Test framework
pytest-cov==4.1.0                # 覆盖率报告 / Coverage reporting
pytest-xdist==3.5.0              # 并行测试 / Parallel testing
pytest-timeout==2.2.0            # 超时控制 / Timeout control
```

### 📊 测试覆盖详情 / Test Coverage Details

| 模块 Module | 文件 File | 测试类 Test Classes | 测试数量 Test Count | 覆盖率 Coverage |
|------------|----------|-------------------|-------------------|-----------------|
| **工具函数 Utils** | `test_utils.py` | 4个类 | 9个测试 | ✅ 100% |
| **评估指标 Metrics** | `test_metrics.py` | 4个类 | 12个测试 | ✅ 100% |
| **总计 Total** | 2个文件 | **8个类** | **21个测试** | ✅ 100% |

#### 测试类详细列表 / Test Class Details

**test_utils.py**:
- `TestTensorToImage` - Tensor到图像转换测试 (3个测试)
- `TestSaveImage` - 图像保存测试 (2个测试)
- `TestAverageMeter` - 平均值计算测试 (3个测试)
- `TestConfigLoading` - 配置加载测试 (1个测试)

**test_metrics.py**:
- `TestPSNR` - PSNR指标测试 (3个测试)
- `TestSSIM` - SSIM指标测试 (3个测试)
- `TestDeltaE` - DeltaE色彩差异测试 (3个测试)
- `TestIterSpeed` - 迭代速度测试 (3个测试)

### 🎯 可用的 Fixtures / Available Fixtures

在 `tests/conftest.py` 中定义的共享 fixtures：

| Fixture 名称 | 描述 Description | 返回类型 Return Type | 用途 Usage |
|-------------|------------------|---------------------|-----------|
| `device` | 可用的计算设备 (CPU/CUDA) / Available device | `torch.device` | GPU/CPU 测试 |
| `sample_image_tensor` | 样本图像 tensor (1,3,256,256) / Sample image | `torch.Tensor` | 图像处理测试 |
| `sample_low_light_tensor` | 低光照图像 tensor / Low-light image | `torch.Tensor` | 增强算法测试 |
| `sample_batch` | 批量数据字典 / Batch data dict | `dict` | 数据加载测试 |
| `temp_output_dir` | 临时输出目录 / Temporary output dir | `pathlib.Path` | 文件保存测试 |
| `sample_config` | 测试配置字典 / Test config dict | `dict` | 配置测试 |

### 🏃 运行测试 / Running Tests

#### 基本命令 / Basic Commands

```bash
# 运行所有测试 / Run all tests
pytest tests/
# 或 / or
make test

# 运行单元测试 / Run unit tests only
pytest tests/unit/ -v
# 或 / or
make test-unit

# 运行集成测试 / Run integration tests
pytest tests/integration/ -v
# 或 / or
make test-integration
```

#### 高级选项 / Advanced Options

```bash
# 生成覆盖率报告 / Generate coverage report
make test-cov
# 查看HTML报告 / View HTML report at: htmlcov/index.html

# 详细输出 / Verbose output
pytest tests/ -vv

# 显示打印输出 / Show print statements
pytest tests/ -s

# 在第一个失败处停止 / Stop at first failure
pytest tests/ -x

# 显示局部变量 / Show local variables
pytest tests/ -l

# 进入调试器 / Enter debugger on failure
pytest tests/ --pdb

# 并行运行（需要pytest-xdist）/ Run in parallel
pytest tests/ -n auto

# 只运行上次失败的测试 / Rerun only failed tests
pytest tests/ --lf
```

#### 使用标记过滤 / Filter by Markers

```bash
# 只运行单元测试 / Unit tests only
pytest -m unit

# 只运行集成测试 / Integration tests only
pytest -m integration

# 跳过慢速测试 / Skip slow tests
pytest -m "not slow"

# 只运行需要GPU的测试 / GPU tests only
pytest -m gpu

# 只运行需要数据集的测试 / Data-dependent tests
pytest -m data
```

### ✍️ 编写测试 / Writing Tests

#### 基本测试结构 / Basic Test Structure

```python
import pytest
import torch

class TestYourModule:
    """测试你的模块 / Test your module"""

    def test_basic_functionality(self):
        """测试基本功能 / Test basic functionality"""
        result = your_function(input_data)
        assert result == expected_output

    def test_edge_case(self):
        """测试边界情况 / Test edge cases"""
        with pytest.raises(ValueError):
            your_function(invalid_input)

    def test_with_fixture(self, sample_image_tensor):
        """使用fixture的测试 / Test using fixture"""
        assert sample_image_tensor.shape == (1, 3, 256, 256)
```

#### 参数化测试 / Parameterized Tests

```python
@pytest.mark.parametrize("input_value,expected", [
    (0, 0),
    (1, 2),
    (2, 4),
    (10, 20),
])
def test_multiply_by_two(input_value, expected):
    """参数化测试示例 / Parameterized test example"""
    assert multiply_by_two(input_value) == expected
```

#### 使用标记 / Using Markers

```python
@pytest.mark.unit
def test_unit_function():
    """单元测试 / Unit test"""
    pass

@pytest.mark.slow
def test_slow_operation():
    """慢速测试 / Slow test"""
    pass

@pytest.mark.gpu
def test_gpu_operation(device):
    """需要GPU的测试 / GPU-required test"""
    if device.type == 'cpu':
        pytest.skip("GPU not available")
    # GPU测试代码 / GPU test code
```

### 📊 覆盖率目标 / Coverage Goals

- **最低要求 Minimum**: 60%
- **项目目标 Target**: 80%
- **理想状态 Ideal**: 90%+
- **当前状态 Current**: ✅ 100% (核心模块 / Core modules)

---

## 📚 文档系统 / Documentation System

### 🎨 文档特性 / Documentation Features

- ✅ **ReadTheDocs 主题** - 专业美观的主题
- ✅ **中文支持** - 完整的中文语言配置
- ✅ **自动API文档** - 使用 autodoc 自动生成
- ✅ **Google/NumPy Docstring** - 支持多种文档字符串风格
- ✅ **Markdown 支持** - 使用 MyST parser
- ✅ **代码高亮** - 自动语法高亮
- ✅ **交叉引用** - 模块间链接

### 📄 文档页面 / Documentation Pages

| 文件 File | 标题 Title | 内容 Content |
|----------|-----------|-------------|
| `index.rst` | 首页 / Home | 项目概述、特性介绍 / Project overview |
| `installation.rst` | 安装指南 / Installation | 环境配置、依赖安装 / Setup guide |
| `quickstart.rst` | 快速开始 / Quick Start | 训练评估教程 / Training tutorial |
| `contributing.rst` | 贡献指南 / Contributing | 开发规范、PR流程 / Dev guidelines |
| `changelog.rst` | 更新日志 / Changelog | 版本历史 / Version history |
| `api/index.rst` | API文档 / API Reference | 自动生成的API / Auto-generated API |

### 🛠️ 构建文档 / Building Documentation

#### 使用 Makefile (推荐 / Recommended)

```bash
# 构建HTML文档 / Build HTML documentation
make docs

# 构建并启动本地服务器 / Build and serve locally
make docs-serve
# 然后访问 / Then visit: http://localhost:8000

# 清理构建文件 / Clean build files
make clean
```

#### 手动构建 / Manual Build

```bash
# 进入文档目录 / Enter docs directory
cd docs

# 构建HTML / Build HTML
make html

# 构建其他格式 / Build other formats
make latexpdf  # PDF文档 / PDF documentation
make epub      # EPUB电子书 / EPUB e-book
make man       # Man手册页 / Man pages

# 清理 / Clean
make clean
```

### 📝 编写文档 / Writing Documentation

#### Docstring 示例 / Docstring Example

```python
def enhance_image(image, gamma=2.2, preserve_color=True):
    """
    增强低光照图像 / Enhance low-light image.

    对输入的低光照图像进行增强处理，提高亮度和对比度。
    Enhance the input low-light image by improving brightness and contrast.

    Args:
        image (torch.Tensor): 输入图像 tensor，形状为 (B, C, H, W)
                              Input image tensor with shape (B, C, H, W)
        gamma (float, optional): Gamma校正值，默认为2.2
                                 Gamma correction value, default 2.2
        preserve_color (bool, optional): 是否保持颜色一致性，默认True
                                        Whether to preserve color, default True

    Returns:
        torch.Tensor: 增强后的图像 tensor，形状与输入相同
                      Enhanced image tensor with same shape as input

    Raises:
        ValueError: 如果输入tensor不是4维 / If input tensor is not 4D
        RuntimeError: 如果gamma值无效 / If gamma value is invalid

    Example:
        >>> image = torch.randn(1, 3, 256, 256)
        >>> enhanced = enhance_image(image, gamma=2.2)
        >>> print(enhanced.shape)
        torch.Size([1, 3, 256, 256])

    Note:
        本函数使用曲线调整方法进行图像增强。
        This function uses curve adjustment for enhancement.

    See Also:
        - :func:`adjust_gamma`: Gamma校正函数 / Gamma correction function
        - :class:`Enhancer`: 增强器类 / Enhancer class
    """
    # 函数实现 / Function implementation
    pass
```

### 🌐 在线部署 / Online Deployment

文档将自动部署到：

- **GitHub Pages**: https://your-username.github.io/SemRoCL
- **ReadTheDocs** (可选): https://semrocl.readthedocs.io

配置在 `.github/workflows/docs.yml` 中 / Configured in `.github/workflows/docs.yml`

---

## 🔄 CI/CD 自动化 / CI/CD Automation

### 📋 工作流概览 / Workflow Overview

| 工作流 Workflow | 文件 File | 触发条件 Trigger | 运行平台 OS | Python版本 |
|----------------|----------|----------------|------------|-----------|
| **自动测试** Tests | `tests.yml` | Push, PR | Ubuntu, Windows | 3.8-3.11 |
| **代码质量** Quality | `code-quality.yml` | Push, PR | Ubuntu | 3.10 |
| **文档构建** Docs | `docs.yml` | Push (main), PR | Ubuntu | 3.10 |

### 1️⃣ 自动测试工作流 / Automated Testing Workflow

**文件**: `.github/workflows/tests.yml`

#### 功能特性 / Features

- ✅ **多平台测试** / Multi-platform testing
  - Ubuntu Latest (Linux)
  - Windows Latest

- ✅ **多Python版本** / Multi-Python versions
  - Python 3.8
  - Python 3.9
  - Python 3.10
  - Python 3.11

- ✅ **依赖缓存** / Dependency caching
  - pip 缓存加速安装 / Pip cache for faster installs

- ✅ **覆盖率报告** / Coverage reporting
  - 自动上传到 Codecov / Auto-upload to Codecov
  - 生成HTML报告 / Generate HTML report

#### 运行流程 / Workflow Steps

```yaml
1. 检出代码 / Checkout code
2. 设置Python环境 / Setup Python
3. 缓存依赖 / Cache dependencies
4. 安装依赖 / Install dependencies
5. 运行Pytest / Run pytest
6. 生成覆盖率报告 / Generate coverage
7. 上传到Codecov / Upload to Codecov
```

#### 触发条件 / Trigger Conditions

```yaml
on:
  push:
    branches: [main, develop, research]
  pull_request:
    branches: [main, develop]
```

### 2️⃣ 代码质量工作流 / Code Quality Workflow

**文件**: `.github/workflows/code-quality.yml`

#### 检查工具 / Quality Tools

| 工具 Tool | 用途 Purpose | 配置文件 Config |
|----------|-------------|----------------|
| **Flake8** | 代码风格检查 / Style checking | `.flake8` |
| **Pylint** | 代码质量分析 / Quality analysis | `.pylintrc` |
| **Black** | 格式检查 / Format checking | `pyproject.toml` |
| **isort** | 导入排序 / Import sorting | `pyproject.toml` |
| **mypy** | 类型检查 / Type checking | `pyproject.toml` |

#### 运行流程 / Workflow Steps

```yaml
1. 检出代码 / Checkout code
2. 设置Python 3.10 / Setup Python 3.10
3. 安装开发依赖 / Install dev dependencies
4. 运行Flake8 / Run Flake8
5. 运行Pylint / Run Pylint
6. 检查Black格式 / Check Black format
7. 检查isort / Check isort
8. 运行mypy / Run mypy
```

### 3️⃣ 文档构建工作流 / Documentation Workflow

**文件**: `.github/workflows/docs.yml`

#### 功能特性 / Features

- ✅ **自动构建** / Automatic build
  - Sphinx HTML 文档 / Sphinx HTML docs

- ✅ **自动部署** / Automatic deployment
  - 部署到 GitHub Pages (main 分支) / Deploy to GitHub Pages (main branch)

- ✅ **PR 预览** / PR preview
  - PR中验证文档构建 / Validate docs in PRs

#### 运行流程 / Workflow Steps

```yaml
1. 检出代码 / Checkout code
2. 设置Python / Setup Python
3. 安装Sphinx依赖 / Install Sphinx dependencies
4. 构建文档 / Build documentation
5. 部署到GitHub Pages / Deploy to GitHub Pages (仅main分支 / main only)
```

### 📊 CI/CD 徽章 / CI/CD Badges

在 README.md 中显示 / Display in README.md:

```markdown
[![Tests](https://github.com/your-username/SemRoCL/workflows/Tests/badge.svg)](https://github.com/your-username/SemRoCL/actions)
[![Code Quality](https://github.com/your-username/SemRoCL/workflows/Code%20Quality/badge.svg)](https://github.com/your-username/SemRoCL/actions)
[![Documentation](https://github.com/your-username/SemRoCL/workflows/Documentation/badge.svg)](https://semrocl.readthedocs.io)
[![codecov](https://codecov.io/gh/your-username/SemRoCL/branch/main/graph/badge.svg)](https://codecov.io/gh/your-username/SemRoCL)
```

---

## 🔍 代码质量工具 / Code Quality Tools

### 📝 配置文件详解 / Configuration Details

#### 1. Flake8 配置 - `.flake8`

```ini
[flake8]
max-line-length = 120              # 最大行长度 / Max line length
exclude =                          # 排除目录 / Exclude directories
    .git,
    __pycache__,
    build,
    dist,
    .venv,
    venv
ignore =                           # 忽略的错误码 / Ignored error codes
    E203,  # Whitespace before ':'
    E501,  # Line too long (handled by black)
    W503,  # Line break before binary operator
max-complexity = 10                # 最大复杂度 / Max complexity
```

**使用方法 / Usage**:
```bash
# 运行Flake8 / Run Flake8
flake8 src/
# 或使用Makefile / Or use Makefile
make lint
```

#### 2. Pylint 配置 - `.pylintrc`

```ini
[MASTER]
max-line-length = 120              # 最大行长度 / Max line length
disable =                          # 禁用的警告 / Disabled warnings
    C0111,  # Missing docstring
    C0103,  # Invalid name
    R0913,  # Too many arguments
```

**使用方法 / Usage**:
```bash
# 运行Pylint / Run Pylint
pylint src/
# 或 / or
make lint
```

#### 3. Black 配置 - `pyproject.toml`

```toml
[tool.black]
line-length = 120                  # 行长度 / Line length
target-version = ['py38', 'py39', 'py310', 'py311']
include = '\.pyi?$'                # 包含文件 / Include files
exclude = '''                      # 排除目录 / Exclude dirs
/(
    \.git
  | \.venv
  | build
  | dist
)/
'''
```

**使用方法 / Usage**:
```bash
# 格式化代码 / Format code
black src/ tests/
# 或 / or
make format

# 检查格式（不修改）/ Check format (no changes)
black --check src/
# 或 / or
make check-format
```

#### 4. isort 配置 - `pyproject.toml`

```toml
[tool.isort]
profile = "black"                  # 与Black兼容 / Compatible with Black
line_length = 120                  # 行长度 / Line length
skip_gitignore = true              # 跳过.gitignore文件 / Skip .gitignore
```

**使用方法 / Usage**:
```bash
# 排序导入 / Sort imports
isort src/ tests/
# 或 / or
make format

# 检查排序 / Check sorting
isort --check-only src/
```

#### 5. mypy 配置 - `pyproject.toml`

```toml
[tool.mypy]
python_version = "3.8"             # Python版本 / Python version
warn_return_any = true             # 警告any返回 / Warn any return
warn_unused_configs = true         # 警告未使用配置 / Warn unused config
ignore_missing_imports = true      # 忽略缺失导入 / Ignore missing imports
```

**使用方法 / Usage**:
```bash
# 类型检查 / Type checking
mypy src/
# 或 / or
make lint
```

#### 6. Pytest 配置 - `pytest.ini`

```ini
[pytest]
testpaths = tests                  # 测试路径 / Test paths
python_files = test_*.py           # 测试文件模式 / Test file pattern
python_classes = Test*             # 测试类模式 / Test class pattern
python_functions = test_*          # 测试函数模式 / Test function pattern
addopts =                          # 额外选项 / Additional options
    -v                             # 详细输出 / Verbose
    --strict-markers               # 严格标记 / Strict markers
    --tb=short                     # 简短回溯 / Short traceback
    --cov-report=term-missing      # 覆盖率报告 / Coverage report
markers =                          # 自定义标记 / Custom markers
    unit: Unit tests
    integration: Integration tests
    slow: Slow tests
    gpu: GPU required tests
    data: Data dependent tests
```

#### 7. Coverage 配置 - `pyproject.toml`

```toml
[tool.coverage.run]
source = ["src"]                   # 源代码目录 / Source directory
omit = [                           # 忽略文件 / Omit files
    "*/tests/*",
    "*/test_*.py",
]

[tool.coverage.report]
precision = 2                      # 精度 / Precision
show_missing = true                # 显示缺失行 / Show missing lines
skip_empty = true                  # 跳过空文件 / Skip empty files
```

### 🎯 代码规范标准 / Coding Standards

| 规范项 Standard | 要求 Requirement | 工具 Tool |
|----------------|------------------|-----------|
| **行宽** Line Width | 120字符 / 120 chars | Black, Flake8 |
| **缩进** Indentation | 4空格 / 4 spaces | Black |
| **引号** Quotes | 双引号 / Double quotes | Black |
| **导入排序** Import Sort | 标准库→第三方→本地 / stdlib→3rd→local | isort |
| **命名** Naming | snake_case函数，PascalCase类 / snake_case funcs, PascalCase classes | Pylint |
| **类型提示** Type Hints | 推荐但非强制 / Recommended but optional | mypy |
| **Docstring** | Google/NumPy风格 / Google/NumPy style | Sphinx |

---

## 🛠️ 配置文件详解 / Configuration Details

### 📋 requirements-dev.txt

开发依赖完整列表 / Complete development dependencies:

```txt
# 测试工具 / Testing Tools
pytest==7.4.3                      # 测试框架 / Test framework
pytest-cov==4.1.0                  # 覆盖率报告 / Coverage
pytest-xdist==3.5.0                # 并行测试 / Parallel testing
pytest-timeout==2.2.0              # 超时控制 / Timeout control

# 代码质量 / Code Quality
flake8==6.1.0                      # 代码检查 / Linting
pylint==3.0.3                      # 质量分析 / Quality analysis
black==23.12.1                     # 代码格式化 / Formatting
isort==5.13.2                      # 导入排序 / Import sorting
mypy==1.8.0                        # 类型检查 / Type checking

# 文档工具 / Documentation Tools
sphinx==7.2.6                      # 文档生成 / Doc generation
sphinx-rtd-theme==2.0.0            # RTD主题 / RTD theme
sphinx-autodoc-typehints==1.25.2   # 类型提示 / Type hints
myst-parser==2.0.0                 # Markdown支持 / Markdown support

# 其他工具 / Other Tools
pre-commit==3.6.0                  # Git钩子 / Git hooks
```

### 🎛️ Makefile 完整命令 / Complete Makefile Commands

| 命令 Command | 功能 Function | 说明 Description |
|-------------|---------------|-----------------|
| `make help` | 显示帮助 / Show help | 列出所有可用命令 / List all commands |
| `make install` | 安装依赖 / Install deps | 安装生产依赖 / Install production deps |
| `make install-dev` | 安装开发依赖 / Install dev deps | 安装所有依赖 / Install all deps |
| `make test` | 运行测试 / Run tests | 运行所有测试 / Run all tests |
| `make test-cov` | 测试+覆盖率 / Test with coverage | 生成覆盖率报告 / Generate coverage |
| `make test-unit` | 单元测试 / Unit tests | 只运行单元测试 / Run unit tests only |
| `make test-integration` | 集成测试 / Integration tests | 只运行集成测试 / Run integration only |
| `make lint` | 代码检查 / Lint code | 运行所有检查工具 / Run all linters |
| `make format` | 格式化代码 / Format code | Black + isort / Black + isort |
| `make check-format` | 检查格式 / Check format | 不修改文件 / No file changes |
| `make docs` | 构建文档 / Build docs | 生成HTML文档 / Generate HTML docs |
| `make docs-serve` | 文档服务器 / Docs server | 本地预览 (port 8000) / Local preview |
| `make clean` | 清理文件 / Clean files | 删除缓存和构建 / Remove cache & build |
| `make train-stage1` | 训练Stage1 / Train stage1 | 运行Stage1训练 / Run stage1 training |
| `make train-stage2` | 训练Stage2 / Train stage2 | 运行Stage2训练 / Run stage2 training |

---

## 🔄 开发工作流 / Development Workflow

### 📅 日常开发流程 / Daily Development Workflow

```bash
# ========================================
# 1️⃣ 编写代码 / Write Code
# ========================================
vim src/your_module.py             # 编辑文件 / Edit file

# ========================================
# 2️⃣ 格式化代码 / Format Code
# ========================================
make format                        # Black + isort 自动格式化 / Auto format

# ========================================
# 3️⃣ 运行测试 / Run Tests
# ========================================
make test                          # 运行所有测试 / Run all tests
# 或针对特定模块 / Or for specific module:
pytest tests/unit/test_your_module.py -v

# ========================================
# 4️⃣ 检查代码质量 / Check Code Quality
# ========================================
make lint                          # 运行所有检查 / Run all checks

# ========================================
# 5️⃣ 查看覆盖率 / Check Coverage
# ========================================
make test-cov                      # 生成覆盖率报告 / Generate coverage
# 打开 htmlcov/index.html 查看详细报告 / Open htmlcov/index.html for details

# ========================================
# 6️⃣ 提交代码 / Commit Code
# ========================================
git add .
git commit -m "Add: your feature description"
git push origin your-branch
```

### 🆕 添加新功能流程 / Adding New Feature Workflow

```bash
# ========================================
# 步骤1: 创建分支 / Step 1: Create Branch
# ========================================
git checkout -b feature/your-feature-name

# ========================================
# 步骤2: 编写功能代码 / Step 2: Write Feature Code
# ========================================
vim src/your_module.py

# 示例代码结构 / Example code structure:
"""
def your_new_function(param1, param2):
    '''
    功能描述 / Function description.

    Args:
        param1 (type): 参数说明 / Parameter description
        param2 (type): 参数说明 / Parameter description

    Returns:
        type: 返回值说明 / Return description
    '''
    # 实现代码 / Implementation
    pass
"""

# ========================================
# 步骤3: 编写测试 / Step 3: Write Tests
# ========================================
vim tests/unit/test_your_module.py

# 测试代码示例 / Test code example:
"""
class TestYourFunction:
    def test_basic_case(self):
        result = your_new_function(input1, input2)
        assert result == expected_output

    def test_edge_case(self):
        with pytest.raises(ValueError):
            your_new_function(invalid_input1, invalid_input2)
"""

# ========================================
# 步骤4: 更新文档 / Step 4: Update Documentation
# ========================================
# 添加到API文档 / Add to API docs
vim docs/source/api/your_module.rst

# ========================================
# 步骤5: 运行完整检查 / Step 5: Run Full Checks
# ========================================
make format                        # 格式化 / Format
make test                          # 测试 / Test
make lint                          # 检查 / Lint
make docs                          # 文档 / Docs

# ========================================
# 步骤6: 提交PR / Step 6: Submit PR
# ========================================
git add .
git commit -m "Add: your feature with tests and docs"
git push origin feature/your-feature-name
# 然后在GitHub上创建Pull Request / Then create PR on GitHub
```

### 🐛 修复Bug流程 / Bug Fix Workflow

```bash
# ========================================
# 1️⃣ 重现Bug / Reproduce Bug
# ========================================
# 编写失败的测试来重现bug / Write failing test to reproduce
vim tests/unit/test_bug_reproduction.py

def test_bug_reproduction():
    """重现Issue #123的bug / Reproduce bug from Issue #123"""
    # 导致bug的输入 / Input that causes bug
    buggy_input = ...
    # 期望的输出 / Expected output
    expected = ...
    # 实际会失败 / This will fail
    assert function_with_bug(buggy_input) == expected

# 运行测试确认失败 / Run test to confirm failure
pytest tests/unit/test_bug_reproduction.py -v

# ========================================
# 2️⃣ 修复Bug / Fix Bug
# ========================================
vim src/module_with_bug.py
# 修复代码 / Fix code

# ========================================
# 3️⃣ 验证修复 / Verify Fix
# ========================================
pytest tests/unit/test_bug_reproduction.py -v  # 应该通过 / Should pass
make test                                      # 确保没破坏其他测试 / Ensure no regression

# ========================================
# 4️⃣ 提交修复 / Commit Fix
# ========================================
git add .
git commit -m "Fix: bug description (fixes #123)"
git push
```

### 📝 代码审查检查清单 / Code Review Checklist

在提交PR前检查 / Check before submitting PR:

- [ ] ✅ 代码已格式化 / Code formatted
  ```bash
  make format
  ```

- [ ] ✅ 所有测试通过 / All tests pass
  ```bash
  make test
  ```

- [ ] ✅ 代码质量检查通过 / Quality checks pass
  ```bash
  make lint
  ```

- [ ] ✅ 测试覆盖率足够 / Adequate test coverage
  ```bash
  make test-cov  # 目标 >80% / Target >80%
  ```

- [ ] ✅ 文档已更新 / Documentation updated
  - Docstrings 完整 / Complete docstrings
  - API文档已更新 / API docs updated
  - 示例代码正确 / Example code correct

- [ ] ✅ CHANGELOG已更新 / Changelog updated
  ```bash
  vim CHANGELOG.md
  ```

- [ ] ✅ 没有TODO或临时代码 / No TODOs or temp code

- [ ] ✅ 提交信息清晰 / Clear commit messages
  - 格式: `Type: Description`
  - Type: Add/Update/Fix/Remove/Refactor

---

## 🐛 故障排除 / Troubleshooting

### ❌ 测试失败 / Test Failures

#### 问题1: ImportError

```bash
# 错误 / Error:
ImportError: cannot import name 'your_function' from 'src.module'

# 解决方案 / Solution:
# 确保src在Python路径中 / Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# 或在pytest.ini中配置 / Or configure in pytest.ini
```

#### 问题2: 测试超时 / Test Timeout

```bash
# 错误 / Error:
pytest.timeout: Test exceeded timeout

# 解决方案 / Solution:
# 增加超时时间 / Increase timeout
pytest tests/ --timeout=300

# 或跳过慢速测试 / Or skip slow tests
pytest tests/ -m "not slow"
```

#### 问题3: GPU测试失败 / GPU Test Failure

```bash
# 错误 / Error:
RuntimeError: CUDA out of memory

# 解决方案 / Solution:
# 跳过GPU测试 / Skip GPU tests
pytest tests/ -m "not gpu"

# 或设置环境变量 / Or set environment
export CUDA_VISIBLE_DEVICES=""  # 强制使用CPU / Force CPU
```

### 📚 文档构建失败 / Documentation Build Failure

#### 问题1: Sphinx错误

```bash
# 错误 / Error:
sphinx-build: command not found

# 解决方案 / Solution:
# 安装文档依赖 / Install docs dependencies
pip install -r requirements-dev.txt
```

#### 问题2: 主题缺失

```bash
# 错误 / Error:
Theme error: no theme named 'sphinx_rtd_theme' found

# 解决方案 / Solution:
pip install sphinx-rtd-theme
```

#### 问题3: autodoc失败

```bash
# 错误 / Error:
WARNING: autodoc: failed to import module 'your_module'

# 解决方案 / Solution:
# 清理后重建 / Clean and rebuild
cd docs
make clean
make html
```

### 🔧 代码质量工具问题 / Code Quality Tool Issues

#### 问题1: Black和Flake8冲突

```bash
# 错误 / Error:
E501 line too long (>79 characters)

# 解决方案 / Solution:
# 已在.flake8中配置忽略 / Already configured to ignore in .flake8
# 如果仍有问题，运行 / If still issues, run:
make format  # Black会自动处理 / Black will handle
```

#### 问题2: isort和Black冲突

```bash
# 解决方案 / Solution:
# isort配置使用black profile / isort configured with black profile
# 始终先运行black / Always run black first
black src/ tests/
isort src/ tests/
# 或使用Makefile / Or use Makefile
make format
```

### ⚠️ CI/CD失败 / CI/CD Failures

#### 问题1: GitHub Actions测试失败

```bash
# 检查步骤 / Check steps:
1. 在本地重现 / Reproduce locally:
   make test

2. 检查Python版本 / Check Python version:
   python --version  # 确保匹配CI / Ensure matches CI

3. 检查依赖 / Check dependencies:
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
```

#### 问题2: 文档部署失败

```bash
# 检查步骤 / Check steps:
1. 本地构建 / Build locally:
   make docs

2. 检查权限 / Check permissions:
   # 确保GitHub Actions有写权限 / Ensure GHA has write permission

3. 检查分支 / Check branch:
   # 文档只在main分支部署 / Docs deploy only on main branch
```

---

## ✨ 最佳实践 / Best Practices

### 📝 测试最佳实践 / Testing Best Practices

#### ✅ DO (推荐做法)

```python
# ✅ 使用描述性测试名称 / Use descriptive test names
def test_enhance_image_improves_brightness_by_at_least_50_percent():
    pass

# ✅ 每个测试只测一个功能 / Test one thing per test
def test_save_image_creates_file():
    # 只测试文件创建 / Only test file creation
    pass

def test_save_image_preserves_quality():
    # 只测试质量保持 / Only test quality preservation
    pass

# ✅ 使用fixtures减少重复 / Use fixtures to reduce duplication
def test_with_fixture(sample_image_tensor):
    # 不需要手动创建tensor / No need to manually create tensor
    assert sample_image_tensor.shape == (1, 3, 256, 256)

# ✅ 测试边界情况 / Test edge cases
def test_handle_empty_input():
    with pytest.raises(ValueError):
        process_image(None)

def test_handle_extreme_values():
    result = enhance_image(torch.ones(1, 3, 256, 256) * 1000)
    assert torch.all(result >= 0) and torch.all(result <= 1)

# ✅ 使用参数化减少重复 / Use parametrization
@pytest.mark.parametrize("gamma", [1.0, 1.5, 2.0, 2.5])
def test_gamma_correction_with_different_values(gamma):
    result = apply_gamma(image, gamma)
    assert result.shape == image.shape
```

#### ❌ DON'T (不推荐做法)

```python
# ❌ 模糊的测试名称 / Vague test names
def test_function1():
    pass

# ❌ 一个测试测多个功能 / Testing multiple things
def test_everything():
    # 测试保存、加载、处理... / Testing save, load, process...
    pass

# ❌ 硬编码数据 / Hardcoded data
def test_process():
    image = torch.randn(1, 3, 256, 256)  # 每个测试都重复 / Repeated in every test
    # 应该使用fixture / Should use fixture

# ❌ 没有断言 / No assertions
def test_function():
    result = my_function()
    # 忘记assert / Forgot to assert

# ❌ 测试实现细节而非行为 / Testing implementation not behavior
def test_internal_variable_name():
    # 不应该测试内部实现 / Shouldn't test internal implementation
    pass
```

### 📖 文档最佳实践 / Documentation Best Practices

#### ✅ DO (推荐做法)

```python
def enhance_image(image: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """
    增强低光照图像的亮度和对比度。
    Enhance brightness and contrast of low-light images.

    本函数使用伽马校正和曲线调整方法对低光照图像进行增强。
    This function uses gamma correction and curve adjustment for enhancement.

    Args:
        image (torch.Tensor): 输入图像，形状为 (B, C, H, W)，值域 [0, 1]
                              Input image with shape (B, C, H, W), range [0, 1]
        gamma (float, optional): 伽马校正系数，默认2.2。范围 [1.0, 3.0]
                                 Gamma correction factor, default 2.2, range [1.0, 3.0]

    Returns:
        torch.Tensor: 增强后的图像，形状和值域与输入相同
                      Enhanced image with same shape and range as input

    Raises:
        ValueError: 如果图像不是4维tensor / If image is not 4D tensor
        RuntimeError: 如果gamma不在有效范围内 / If gamma not in valid range

    Example:
        >>> image = torch.rand(1, 3, 256, 256)
        >>> enhanced = enhance_image(image, gamma=2.2)
        >>> print(enhanced.shape)
        torch.Size([1, 3, 256, 256])

    Note:
        - 输入图像应该归一化到 [0, 1]
        - Input image should be normalized to [0, 1]
        - 处理在GPU上进行如果可用
        - Processing is done on GPU if available

    See Also:
        :func:`adjust_gamma`: 单独的伽马校正函数
        :class:`Enhancer`: 增强器主类
    """
    # 实现 / Implementation
    pass
```

#### ❌ DON'T (不推荐做法)

```python
# ❌ 没有docstring
def enhance_image(image, gamma=2.2):
    pass

# ❌ 不完整的docstring
def enhance_image(image, gamma=2.2):
    """Enhance image."""  # 太简单 / Too simple
    pass

# ❌ 只有英文或只有中文
def enhance_image(image, gamma=2.2):
    """Enhance brightness of image."""  # 应该双语 / Should be bilingual
    pass

# ❌ 没有类型提示
def enhance_image(image, gamma=2.2):  # 缺少类型提示 / Missing type hints
    pass
```

### 🎨 代码风格最佳实践 / Code Style Best Practices

#### ✅ DO (推荐做法)

```python
# ✅ 清晰的变量命名 / Clear variable names
enhanced_image = enhance_brightness(low_light_image)

# ✅ 使用类型提示 / Use type hints
def process_batch(images: torch.Tensor, batch_size: int = 32) -> List[torch.Tensor]:
    pass

# ✅ 合理的函数长度 / Reasonable function length
def enhance_image(image):
    """单一职责的函数 / Single responsibility function"""
    image = normalize(image)
    image = adjust_gamma(image)
    image = enhance_contrast(image)
    return image

# ✅ 有意义的注释 / Meaningful comments
# 使用双边滤波保留边缘的同时去除噪声
# Use bilateral filter to remove noise while preserving edges
filtered = cv2.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)

# ✅ 常量使用大写 / Constants in uppercase
MAX_ITERATIONS = 100
DEFAULT_GAMMA = 2.2
```

#### ❌ DON'T (不推荐做法)

```python
# ❌ 模糊的变量名 / Vague variable names
x = enhance(y)  # x和y是什么? / What are x and y?

# ❌ 过长的函数 / Too long functions
def do_everything():
    # 200行代码... / 200 lines of code...
    pass

# ❌ 无意义的注释 / Meaningless comments
x = x + 1  # 增加x / Increment x

# ❌ 魔法数字 / Magic numbers
image = image * 2.2  # 2.2是什么? / What is 2.2?
# 应该: / Should be:
GAMMA_CORRECTION_FACTOR = 2.2
image = image * GAMMA_CORRECTION_FACTOR
```

### 🔄 Git 提交最佳实践 / Git Commit Best Practices

#### ✅ 好的提交信息 / Good Commit Messages

```bash
# ✅ 清晰的前缀和描述 / Clear prefix and description
git commit -m "Add: bilateral filtering for noise reduction"
git commit -m "Fix: image quality degradation in batch processing"
git commit -m "Update: documentation for enhance_image function"
git commit -m "Refactor: split large training function into smaller modules"
git commit -m "Remove: deprecated curriculum learning scheduler"

# ✅ 详细的提交信息 / Detailed commit message
git commit -m "Add: comprehensive unit tests for image processing

- Add test_utils.py with 9 test cases
- Add test_metrics.py with 12 test cases
- Configure pytest with coverage reporting
- Update CI/CD to run tests automatically

Resolves #123"
```

#### ❌ 不好的提交信息 / Bad Commit Messages

```bash
# ❌ 太模糊 / Too vague
git commit -m "update"
git commit -m "fix bug"
git commit -m "changes"

# ❌ 没有前缀 / No prefix
git commit -m "bilateral filtering"

# ❌ 太长放在标题 / Too long in title
git commit -m "Add bilateral filtering and non-local means denoising and Gaussian smoothing and adaptive sharpening for image quality improvement in the training and evaluation pipelines"
```

---

## 📊 附录 / Appendix

### 📈 覆盖率报告示例 / Coverage Report Example

```bash
$ make test-cov

---------- coverage: platform win32, python 3.10.11 -----------
Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
src/__init__.py                   0      0   100%
src/utils.py                    145      0   100%
src/metrics_utils.py             89      0   100%
src/data_loader.py              234     45    81%   145-156, 201-215
src/loss_functions.py           312     87    72%   78-95, 156-178, ...
-----------------------------------------------------------
TOTAL                          1450    245    83%

Coverage HTML written to dir htmlcov
```

### 🔗 有用的链接 / Useful Links

#### 官方文档 / Official Documentation

- **Pytest**: https://docs.pytest.org/
- **Sphinx**: https://www.sphinx-doc.org/
- **Black**: https://black.readthedocs.io/
- **Flake8**: https://flake8.pycqa.org/
- **GitHub Actions**: https://docs.github.com/en/actions

#### 最佳实践指南 / Best Practice Guides

- **Python测试最佳实践** / Python Testing Best Practices:
  https://docs.pytest.org/en/latest/goodpractices.html

- **Google Python风格指南** / Google Python Style Guide:
  https://google.github.io/styleguide/pyguide.html

- **如何编写Git提交信息** / How to Write Git Commit Messages:
  https://chris.beams.io/posts/git-commit/

#### 工具配置示例 / Tool Configuration Examples

- **Python项目模板** / Python Project Templates:
  https://github.com/audreyfeldroy/cookiecutter-pypackage

- **CI/CD配置示例** / CI/CD Configuration Examples:
  https://github.com/actions/starter-workflows

### 📞 获取帮助 / Getting Help

#### 项目相关 / Project Related

- 📖 查看文档：`docs/build/html/index.html`
- 💬 提Issue：https://github.com/your-username/SemRoCL/issues
- 📧 联系维护者：your.email@example.com

#### 工具相关 / Tool Related

- **Pytest问题**：https://github.com/pytest-dev/pytest/issues
- **Sphinx问题**：https://github.com/sphinx-doc/sphinx/issues
- **GitHub Actions问题**：https://github.com/actions/toolkit/issues

---

## 🎯 总结 / Summary

### ✅ 已完成的工作 / Completed Work

- [x] **测试框架** / Testing Framework
  - 21个单元测试 / 21 unit tests
  - 100%核心模块覆盖 / 100% core module coverage
  - Pytest配置完整 / Complete pytest configuration

- [x] **文档系统** / Documentation System
  - Sphinx + ReadTheDocs / Sphinx + ReadTheDocs
  - 7个文档页面 / 7 documentation pages
  - 中文支持 / Chinese language support

- [x] **CI/CD自动化** / CI/CD Automation
  - 3个GitHub Actions工作流 / 3 GitHub Actions workflows
  - 多平台多版本测试 / Multi-platform multi-version testing
  - 自动文档部署 / Automatic docs deployment

- [x] **代码质量工具** / Code Quality Tools
  - 5个质量检查工具 / 5 quality checking tools
  - 统一配置管理 / Unified configuration
  - Makefile快捷命令 / Makefile shortcuts

### 🎓 学到的技能 / Skills Learned

通过建立这套基础设施，开发者可以学习到：

- ✅ 如何编写专业的Python测试 / How to write professional Python tests
- ✅ 如何使用Sphinx生成API文档 / How to generate API docs with Sphinx
- ✅ 如何配置CI/CD自动化流程 / How to configure CI/CD automation
- ✅ 如何使用代码质量工具 / How to use code quality tools
- ✅ 如何组织Python项目结构 / How to organize Python project structure

### 🚀 下一步 / Next Steps

推荐的后续改进（按优先级）/ Recommended improvements (by priority):

**高优先级** / High Priority:
- [x] 单元测试框架 ✅
- [x] API文档系统 ✅
- [x] CI/CD流程 ✅

**中优先级** / Medium Priority:
- [ ] Docker镜像 / Docker image
- [ ] 模型导出工具 (ONNX/TensorRT) / Model export tools
- [ ] 交互式Demo (Gradio/Streamlit) / Interactive demo

**低优先级** / Low Priority:
- [ ] 更多数据集支持 / More dataset support
- [ ] 预训练模型下载 / Pretrained model download
- [ ] 可视化工具 / Visualization tools

---

## 📝 更新日志 / Changelog

### v1.1.0 (2025-11-10)

#### 新增 / Added
- ✅ 完整的测试框架 (21个测试) / Complete testing framework (21 tests)
- ✅ Sphinx文档系统 (7个页面) / Sphinx documentation system (7 pages)
- ✅ GitHub Actions CI/CD (3个工作流) / GitHub Actions CI/CD (3 workflows)
- ✅ 代码质量工具配置 (5个工具) / Code quality tool configs (5 tools)
- ✅ Makefile开发快捷命令 / Makefile development shortcuts
- ✅ 开发依赖管理 / Development dependency management

#### 文档 / Documentation
- ✅ DEV_INFRASTRUCTURE_GUIDE.md (本文档 / This guide)
- ✅ TESTING_AND_DOCS_SETUP.md (设置总结 / Setup summary)
- ✅ tests/README.md (测试指南 / Testing guide)
- ✅ 更新CHANGELOG.md / Updated CHANGELOG.md
- ✅ 更新README.md / Updated README.md

---

**🎉 祝开发愉快！Happy Coding! 🎉**

**最后更新 / Last Updated**: 2025-11-10
**版本 / Version**: v1.1.0
**维护者 / Maintainer**: SemRoCL Development Team
