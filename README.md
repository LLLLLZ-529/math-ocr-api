# ☁️ math-ocr-api

手写数学公式识别的 **Flask API 部署版**：上传手写公式图片，返回 LaTeX 识别结果。基于 ABM 架构（DenseNet 编码器 + GRU 解码器 + Attention），可部署到 Render 等 PaaS 平台。

> 本项目是 `Handwritten-mathematical-formula-recognition`（SURF 2022）的 API 精简衍生版，代码更规范，适合独立部署。

## ✨ 功能特性

- 🖼️ 支持**文件上传**与 **Base64** 两种图片输入方式
- 📝 返回 LaTeX 识别结果（JSON）
- 🧠 复用原项目模型（`encoder_decoder.py` / `package/utils.py`）
- 🔤 词表自动加载，兼容 `data/` 子目录等常见路径
- 💾 支持加载部分权重（`load_checkpoint_part_weight`）
- 🩺 提供 `/health` 健康检查接口

## 🚀 快速开始

### 环境依赖

- Python 3.8+
- PyTorch、Flask、Pillow、torchvision（以实际运行环境为准【待确认】）

```bash
pip install flask torch torchvision pillow
```

### 本地启动

```bash
python api_server.py
# 服务默认监听 5000 端口
```

### 调用方式

**方式一：上传文件**

```bash
curl -X POST http://localhost:5000/recognize -F "image=@formula.png"
```

**方式二：Base64**

```bash
curl -X POST http://localhost:5000/recognize \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "图片的base64字符串"}'
```

**健康检查**

```bash
curl http://localhost:5000/health
# => {"status": "healthy", "model_loaded": true}
```

## 📁 项目结构

```
math-ocr-api/
└── Handwritten-Mathematical-Formula-Recognition-main/
    ├── api_server.py       # Flask 服务入口
    ├── encoder.py          # DenseNet 编码器
    ├── decoder.py          # GRU 解码器
    ├── encoder_decoder.py  # Encoder-Decoder 模型
    ├── dictionary.txt      # 词表（每行一个 token）
    └── ...
```

## 📌 部署前必读

1. **模型权重**：代码会加载权重文件（`model_weights.pkl` 或类似），仓库未包含 → 需自行放入并从原项目训练/导出。
2. **依赖清单**：目前无 `requirements.txt`，部署到 Render 前必须补充（Render 靠它装依赖）。
3. **启动命令**：Render 需配置启动命令（如 `gunicorn api_server:app` 或 `python api_server.py`）。


## ⚠️ 相关仓库

- 主项目：[Handwritten-mathematical-formula-recognition](https://github.com/LLLLLZ-529/Handwritten-mathematical-formula-recognition)

## 📄 许可

未指定开源许可（默认保留所有权利）。
