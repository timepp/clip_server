# CLIP Image Search Server

一个基于 OpenAI CLIP 模型的语义图像搜索服务器，支持文本查询和相似图片搜索。

## 🚀 快速开始

### 使用 Docker（推荐）

1. **构建 Docker 镜像**
   ```bash
   ./build-docker.sh
   ```
   
2. **运行容器**
   ```bash
   ./run-docker.sh
   ```

3. **访问 Web 界面**
   打开浏览器访问：http://localhost:5000/index.html

### 直接运行

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **启动服务器**
   ```bash
   python clip_server.py
   ```

## 🎯 功能特性

- **📁 目录扫描**：自动扫描 `./db` 目录中的图片并生成嵌入向量
- **💬 文本搜索**：使用自然语言描述搜索相关图片
- **🖼️ 相似搜索**：上传图片查找相似的图片
- **📊 状态监控**：实时查看服务器状态和图片数量
- **🌐 Web 界面**：现代化的用户界面，支持桌面和移动设备

## 🐳 Docker 优势

### 预下载模型
- CLIP 模型在构建时预下载到镜像中
- 容器启动无需等待模型下载（节省 1-2 分钟启动时间）
- 离线环境可直接使用

### 环境隔离
- 无需在本地安装 Python 依赖
- 统一的运行环境，避免版本冲突
- 一键部署，跨平台兼容

## 📁 目录结构

```
clip_server/
├── clip_server.py      # 主服务器代码
├── index.html          # Web 客户端界面
├── requirements.txt    # Python 依赖
├── dockerfile          # Docker 构建文件
├── download_model.py   # 模型预下载脚本
├── build-docker.sh     # Docker 构建脚本
├── run-docker.sh       # Docker 运行脚本
├── .dockerignore       # Docker 忽略文件
└── db/                 # 图片存储目录（自动创建）
```

## 🔧 API 接口

- `GET /status` - 获取服务器状态
- `GET /scandir` - 扫描目录生成嵌入向量
- `GET /query?text=<描述>&top_k=<数量>` - 文本搜索
- `POST /similar` - 相似图片搜索（上传图片）
- `GET /db/<path>` - 访问存储的图片文件
- `GET /index.html` - Web 客户端界面

## 🎨 使用说明

1. **添加图片**：将图片文件放入 `./db` 目录
2. **扫描图片**：点击"Start Scan"按钮或调用 `/scandir` API
3. **搜索图片**：
   - 文本搜索：输入描述性文字
   - 相似搜索：上传参考图片
4. **查看结果**：点击搜索结果中的图片查看详情

## 🔧 配置选项

### 环境变量
- `TRANSFORMERS_CACHE`: 模型缓存目录
- `HF_HOME`: HuggingFace 缓存目录

### 支持的图片格式
- JPEG (.jpg, .jpeg)
- PNG (.png)

## 📊 性能说明

- **模型**：OpenAI CLIP ViT-B/32
- **嵌入维度**：512
- **设备**：CPU（支持 macOS x86_64）
- **索引**：FAISS 内积搜索

## 🛠️ 开发调试

### 查看容器日志
```bash
docker logs -f clip-server-instance
```

### 进入容器调试
```bash
docker exec -it clip-server-instance bash
```

### 本地开发
```bash
python clip_server.py
```

## 📝 注意事项

1. 首次运行时，模型下载需要网络连接
2. Docker 镜像大小约 2-3GB（包含模型）
3. 建议为图片目录挂载持久化存储
4. 大量图片处理需要足够的内存和存储空间

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License
