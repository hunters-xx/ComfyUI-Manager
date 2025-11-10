# 系统依赖配置说明

## 配置位置

系统依赖配置项需要写在**统一资源 JSON 配置文件**的根级别，与 `version`、`custom_nodes`、`models` 同级。

## 配置文件格式

系统依赖配置应该在 JSON 配置文件的根级别，格式如下：

```json
{
  "version": "1.0.2",
  "system_dependencies": {
    "ubuntu": [
      "libjpeg-dev",
      "libpng-dev",
      "libtiff-dev",
      "libfreetype-dev"
    ],
    "centos": [
      "libjpeg-devel",
      "libpng-devel",
      "libtiff-devel",
      "freetype-devel"
    ],
    "macos": [
      "jpeg",
      "libpng",
      "libtiff",
      "freetype"
    ]
  },
  "custom_nodes": [...],
  "models": [...]
}
```

## 支持的平台类型

- `ubuntu` - Ubuntu/Debian 系统（使用 apt 包管理器）
- `centos` - CentOS/RHEL/Fedora 系统（使用 yum/dnf 包管理器）
- `macos` - macOS 系统（使用 brew 包管理器）

## 配置说明

1. **platform 名称**：使用操作系统类型作为键名
   - `ubuntu` - 适用于 Ubuntu、Debian 等基于 apt 的系统
   - `centos` - 适用于 CentOS、RHEL、Fedora 等基于 yum/dnf 的系统
   - `macos` - 适用于 macOS（使用 Homebrew）

2. **包名差异**：不同平台的包名可能不同
   - Ubuntu: `libjpeg-dev`, `libpng-dev`, `ffmpeg`
   - CentOS: `libjpeg-devel`, `libpng-devel`, `ffmpeg`
   - macOS: `jpeg`, `libpng`, `ffmpeg` (Homebrew 包名)

3. **常见系统依赖包名**：
   - **ffmpeg**（视频处理）：所有平台都使用 `ffmpeg`
   - **图像库**：Ubuntu 使用 `-dev` 后缀，CentOS 使用 `-devel` 后缀，macOS 使用简化名称

4. **自动检测**：程序会自动检测操作系统类型，并选择对应的依赖列表

## 工作流程

1. 程序启动时，从配置的 URL 获取 JSON 配置文件
2. 检查 `system_dependencies` 字段是否存在
3. 自动检测当前操作系统类型（ubuntu/centos/macos）
4. 根据检测到的系统类型，从 `system_dependencies` 中获取对应的依赖列表
5. 检查每个依赖包是否已安装
6. 如果缺少依赖，尝试自动安装（需要 sudo 权限）
7. 安装完成后继续处理 custom_nodes 和 models

## 示例配置

### 完整示例（包含 ffmpeg）

```json
{
  "version": "1.0.0",
  "system_dependencies": {
    "ubuntu": [
      "libjpeg-dev",
      "libpng-dev",
      "libtiff-dev",
      "libfreetype-dev",
      "libopenexr-dev",
      "libwebp-dev",
      "python3-dev",
      "python3-pip",
      "build-essential",
      "ffmpeg"
    ],
    "centos": [
      "libjpeg-devel",
      "libpng-devel",
      "libtiff-devel",
      "freetype-devel",
      "libopenexr-devel",
      "libwebp-devel",
      "python3-devel",
      "python3-pip",
      "gcc",
      "gcc-c++",
      "ffmpeg"
    ],
    "macos": [
      "jpeg",
      "libpng",
      "libtiff",
      "freetype",
      "openexr",
      "webp",
      "python3",
      "cmake",
      "ffmpeg"
    ]
  },
  "custom_nodes": [
    {
      "id": "example-node",
      "install_type": "git-clone",
      "files": ["https://github.com/example/node.git"],
      "pip": ["diffusers", "imageio"]
    }
  ],
  "models": []
}
```

### Python 依赖配置

除了系统依赖，自定义节点还可以通过 `pip` 字段配置 Python 包依赖。`pip` 字段应该放在每个 `custom_nodes` 项的配置中：

```json
{
  "custom_nodes": [
    {
      "id": "wan-video-wrapper",
      "install_type": "git-clone",
      "files": ["https://github.com/example/ComfyUI-WanVideoWrapper"],
      "pip": ["diffusers", "imageio"]
    }
  ]
}
```

**pip 字段说明**：
- `pip` 是一个字符串数组，列出需要安装的 Python 包
- 支持版本号约束，如 `"diffusers>=0.21.0"` 或 `"numpy<2"`
- 支持从 git 安装，如 `"git+https://github.com/example/package"`
- 安装顺序：先安装系统依赖，再安装 pip 依赖，最后安装自定义节点

## 注意事项

1. **权限要求**：自动安装系统依赖需要 sudo 权限（macOS 的 brew 除外）
2. **网络连接**：需要网络连接来下载和安装包
3. **安装失败处理**：如果自动安装失败，程序会继续执行，但会记录错误信息
4. **包名准确性**：确保包名在不同平台上是正确的
5. **可选配置**：如果没有 `system_dependencies` 字段，程序会使用默认的最小依赖列表

## 默认依赖

如果配置文件中没有 `system_dependencies` 字段，程序会使用以下默认依赖（仅适用于 Ubuntu）：

```python
['libjpeg-dev', 'libpng-dev', 'libtiff-dev', 'libfreetype-dev']
```

## 使用 direct_installer.py

使用 `direct_installer.py` 时，配置文件通过 `--resource-url` 参数指定：

```bash
python direct_installer.py --resource-url https://example.com/config.json --interval 60
```

程序会定期检查该 URL 的配置文件，如果版本号变化，会重新处理所有资源（包括检查系统依赖）。

## 常见问题

### 1. 缺少 ffmpeg 导致 imageio_ffmpeg 无法导入

**错误信息**：
```
[VideoHelperSuite] - WARNING - Failed to import imageio_ffmpeg
[VideoHelperSuite] - ERROR - No valid ffmpeg found.
```

**解决方案**：在 `system_dependencies` 中添加 `ffmpeg`：

```json
{
  "system_dependencies": {
    "ubuntu": ["ffmpeg"],
    "centos": ["ffmpeg"],
    "macos": ["ffmpeg"]
  }
}
```

安装后，`imageio_ffmpeg` 包会自动检测并使用系统安装的 ffmpeg。

### 2. 缺少 Python 包（如 diffusers）

**错误信息**：
```
ModuleNotFoundError: No module named 'diffusers'
```

**解决方案**：在对应自定义节点的配置中添加 `pip` 字段：

```json
{
  "custom_nodes": [
    {
      "id": "wan-video-wrapper",
      "install_type": "git-clone",
      "files": ["https://github.com/example/ComfyUI-WanVideoWrapper"],
      "pip": ["diffusers"]
    }
  ]
}
```

### 3. 同时需要系统依赖和 Python 依赖

如果自定义节点既需要系统依赖（如 ffmpeg），又需要 Python 包（如 diffusers），需要同时配置：

```json
{
  "version": "1.0.0",
  "system_dependencies": {
    "ubuntu": ["ffmpeg"],
    "centos": ["ffmpeg"],
    "macos": ["ffmpeg"]
  },
  "custom_nodes": [
    {
      "id": "video-node",
      "install_type": "git-clone",
      "files": ["https://github.com/example/video-node"],
      "pip": ["diffusers", "imageio"]
    }
  ]
}
```

### 4. 如何验证系统依赖是否已安装

**查看日志输出**：

如果配置了 `system_dependencies`，安装过程中应该看到以下日志：

```
Checking system dependencies...
Checking system dependencies for ubuntu...
Missing system dependencies: ffmpeg
Attempting to install missing dependencies automatically...
Installing system dependencies using apt...
✅ System dependencies installed successfully
```

或者如果所有依赖都已满足：

```
Checking system dependencies...
Checking system dependencies for ubuntu...
✅ All system dependencies are satisfied
```

**如果只看到 "Checking system dependencies..." 没有后续日志**：

可能的原因和解决方法：

1. **OS 类型检测失败**：
   - 检查系统是否正确识别为 ubuntu/centos/macos
   - 查看是否有 "Detected OS type: ..." 日志
   - 如果检测失败，会使用默认的 ubuntu 依赖列表

2. **配置格式问题**：
   - 确认 `system_dependencies` 字段在根级别（与 `version`、`custom_nodes` 同级）
   - 确认键名正确：`ubuntu`、`centos`、`macos`（小写）
   - 确认值是数组格式：`["ffmpeg"]` 而不是字符串 `"ffmpeg"`

3. **日志被截断**：
   - 查看完整的日志输出
   - 检查是否有错误信息被过滤

4. **依赖列表为空**：
   - 如果 OS 类型检测失败，可能获取不到对应平台的依赖列表
   - 查看是否有 "Required dependencies for {os_type}: ..." 日志

5. **安装失败但未显示错误**：
   - 检查是否有权限问题（需要 sudo）
   - 检查网络连接是否正常
   - 查看是否有 "Failed to install system dependencies" 错误信息

**手动验证 ffmpeg 是否已安装**：

```bash
# Ubuntu/Debian
dpkg -l | grep ffmpeg
# 或者直接测试
ffmpeg -version

# CentOS/RHEL/Fedora
rpm -q ffmpeg

# macOS
brew list | grep ffmpeg
```

**如果 ffmpeg 未安装，有两种解决方案**：

### 方案 1：在配置文件中添加 system_dependencies（推荐）

在 JSON 配置文件的根级别添加 `system_dependencies` 字段：

```json
{
  "version": "1.0.2",
  "system_dependencies": {
    "ubuntu": ["ffmpeg"],
    "centos": ["ffmpeg"],
    "macos": ["ffmpeg"]
  },
  "custom_nodes": [...],
  "models": [...]
}
```

下次 direct_installer.py 运行时，会自动检测并安装 ffmpeg。

### 方案 2：手动安装（临时方案）

如果无法立即修改配置文件，可以手动安装：

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y ffmpeg

# CentOS/RHEL/Fedora
sudo yum install -y ffmpeg
# 或
sudo dnf install -y ffmpeg

# macOS
brew install ffmpeg
```

安装后，`imageio_ffmpeg` 会自动检测并使用系统安装的 ffmpeg。





