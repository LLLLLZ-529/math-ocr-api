from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import sys
import os

# 确保能导入项目中的模型文件
sys.path.append('.')

# 使用仓内实际类 / 函数
from encoder_decoder import Encoder_Decoder
from package.utils import load_dict, gen_sample_bidirection, load_checkpoint_part_weight

app = Flask(__name__)


def build_model_params(vocab_size):
    """
    根据 encoder.py / decoder.py 中的结构，构造一致的 params 字典
    这些值基于仓内网络结构推断，必要时可微调。
    """
    params = {
        'growthRate': 12,       # DenseNet growth rate
        'reduction': 0.5,       # transition reduction
        'bottleneck': True,
        'use_dropout': True,
        'L2R': 1,               # 启用从左到右解码
        'R2L': 0,               # 可按需启用双向
        'D': 342,               # encoder 最终通道数（与 growthRate 相关，按 encoder 推断）
        'n': 512,               # GRU 隐藏维度
        'm': 256,               # embedding /中间维度
        'K': vocab_size,        # 词表大小
        'dim_attention': 512,   # attention 内部维度
        'input_channels': 1     # 灰度输入
    }
    return params


def load_vocab(path='dictionary.txt'):
    """
    从词表文件读取 token（每行一个 token），返回 (idx2word, word2idx, pad_idx, eos_idx)
    - 会尝试常见路径（当前路径和 data/ 子目录）
    - 自动查找常见的 PAD/EOS 标记，找不到时用默认值（PAD=0, EOS=最后一个索引）
    """
    import os

    # 尝试常见路径
    if not os.path.isabs(path):
        cand = [path, os.path.join('data', path), os.path.join(os.getcwd(), path)]
        found = None
        for c in cand:
            if os.path.exists(c):
                found = c
                break
        if found is None:
            raise FileNotFoundError(f"词表文件未找到（尝试路径: {cand}）")
        path = found
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"词表文件未找到: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        tokens = [line.strip() for line in f if line.strip()]

    if not tokens:
        raise ValueError(f"词表文件为空: {path}")

    idx2word = {i: tok for i, tok in enumerate(tokens)}
    word2idx = {tok: i for i, tok in idx2word.items()}

    # 识别常见的 PAD/EOS 标记
    pad_candidates = ['<PAD>', '<pad>', 'PAD', '_PAD', '<blank>']
    eos_candidates = ['<EOS>', '<eos>', '</s>', '<s>', '[EOS]']

    pad_idx = None
    eos_idx = None
    for p in pad_candidates:
        if p in word2idx:
            pad_idx = word2idx[p]
            break
    for e in eos_candidates:
        if e in word2idx:
            eos_idx = word2idx[e]
            break

    # 默认回退
    if pad_idx is None:
        pad_idx = 0
    if eos_idx is None:
        eos_idx = len(tokens) - 1

    return idx2word, word2idx, pad_idx, eos_idx


def load_model_and_vocab():
    """
    加载词表与模型权重，返回 (model, idx2word, params, special_tokens)
    """
    import os
    try:
        # 1) 读取词表
        idx2word, word2idx, pad_idx, eos_idx = load_vocab('dictionary.txt')
        if not idx2word:
            print("词表为空或加载失败")
            return None, None, None, None

        VOCAB_SIZE = len(idx2word)

        # 2) 获取模型配置（若 package.utils.get_model_config 不存在，使用合理默认）
        try:
            from package.utils import get_model_config
            config = get_model_config()
        except Exception:
            config = {'embed_dim': 256, 'hidden_size': 512, 'D': 342, 'n': 512, 'm': 256, 'dim_attention': 512}

        # 3) 构造 params 并初始化模型（必须在加载权重前）
        params = build_model_params(VOCAB_SIZE)
        try:
            model = Encoder_Decoder(params)
        except Exception as e:
            # 兼容不同构造器签名的回退
            try:
                model = Encoder_Decoder(vocab_size=VOCAB_SIZE,
                                        embedding_dim=config.get('embed_dim', 256),
                                        hidden_size=config.get('hidden_size', 512))
            except Exception as e2:
                print("无法使用 Encoder_Decoder 初始化模型:", e, e2)
                return None, None, None, None

        # 4) 加载权重（优先使用 model_weights.pkl）
        ckpt_vocab = 'model_weights.pkl'
        if os.path.exists(ckpt_vocab):
            try:
                checkpoint = torch.load(ckpt_vocab, map_location='cpu')
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
                    model.load_state_dict(checkpoint)
                else:
                    # 如果结构不识别，可尝试部分加载工具
                    try:
                        load_checkpoint_part_weight(model, ckpt_vocab)
                    except Exception as e3:
                        raise RuntimeError("权重文件结构不识别，且部分加载失败") from e3
                print("从", ckpt_vocab, "加载权重成功")
            except Exception as e:
                print("从", ckpt_vocab, "加载失败:", e)
        else:
            print(f"未找到指定权重文件 {ckpt_vocab}，将使用随机初始化模型。")

        model.eval()

        special_tokens = {
            'PAD_TOKEN': idx2word.get(pad_idx, '<PAD>'),
            'EOS_TOKEN': idx2word.get(eos_idx, '<EOS>')
        }

        print("模型与词表加载完成")
        return model, idx2word, params, special_tokens
    except Exception as e:
        print("load_model_and_vocab 失败:", e)
        return None, None, None, None


GLOBAL_MODEL, GLOBAL_IDX2WORD, GLOBAL_PARAMS, GLOBAL_SPECIAL_TOKENS = load_model_and_vocab()


def preprocess_image(image_bytes, target_height=64):
    """
    将输入图片转为模型需要的 (1, C, H, W) tensor
    注意：encoder 期望灰度图 (C=1)
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((target_height, target_height)),  # 可改为按比例 resize + pad
        transforms.ToTensor(),
        # 不强制标准化，原训练可能没做相同归一化；如训练时做了请替换 mean/std
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = transform(image)  # (C,H,W)
    return image.unsqueeze(0)  # (1,C,H,W)


def decode_with_beam(model, input_tensor, params, idx2word, k=1, maxlen=120):
    """
    使用仓内的 gen_sample_bidirection 进行解码（贪婪 k=1 为默认）。
    返回解码出的 token string。
    """
    # gen_sample_bidirection 在 [`package.utils.gen_sample_bidirection`](package/utils.py)
    samples, scores, _, _ = gen_sample_bidirection(model, input_tensor, params, gpu_flag=False, k=k, maxlen=maxlen,
                                                   idx_decoder=1)
    if not samples:
        return ""
    best = samples[0]  # 最佳序列（索引列表）
    tokens = []
    for idx in best:
        token = idx2word.get(idx, None)
        if token is None:
            continue
        # 假设词表中有结束符 <eos> 或类似标记；这里直接停止在常见结束符上
        if token in ('<eos>', '<EOS>', '<EOS>'):
            break
        tokens.append(token)
    return ' '.join(tokens)


# ----------------------------------------------------
# 步骤三：修改 decode_output 函数
# ----------------------------------------------------
def decode_output(output, idx2word, special_tokens):
    """
    将模型输出 (Tensor) 转换为 LaTeX 字符串
    - output: logits 或概率张量，形状 (batch, seq_len, vocab) 或 (seq_len, vocab)
    - idx2word: {idx: token}
    - special_tokens: {'EOS_TOKEN': str, 'PAD_TOKEN': str}
    """
    if not idx2word or not special_tokens:
        raise ValueError("模型或词表信息未加载，无法进行解码")

    # handling tensor dimensions -> 得到 token 索引序列（按最大概率）
    if hasattr(output, 'dim') and output.dim() >= 2:
        token_indices = output.argmax(dim=-1).squeeze().tolist()
    else:
        # 若不是 tensor，尝试直接当作索引列表
        token_indices = list(output)

    # 确保为列表
    if isinstance(token_indices, int):
        token_indices = [token_indices]

    formula_tokens = []

    EOS_TOKEN = special_tokens.get('EOS_TOKEN', '<EOS>')
    PAD_TOKEN = special_tokens.get('PAD_TOKEN', '<PAD>')

    for idx in token_indices:
        # 获取 token，如果索引超出范围或为 None，默认为 PAD_TOKEN
        token = idx2word.get(int(idx), PAD_TOKEN)

        if token == EOS_TOKEN:
            break
        if token != PAD_TOKEN:
            formula_tokens.append(token)

    return ' '.join(formula_tokens)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": GLOBAL_MODEL is not None})


    return jsonify({"status": "healthy", "model_loaded": GLOBAL_MODEL is not None})


@app.route('/recognize', methods=['POST'])
def recognize_formula():
    if GLOBAL_MODEL is None or GLOBAL_IDX2WORD is None or GLOBAL_PARAMS is None:
        return jsonify({"success": False, "error": "模型尚未加载或加载失败"}), 503
    try:
        image_bytes = None
        if request.json and 'image_base64' in request.json:
            image_base64 = request.json['image_base64']
            image_bytes = base64.b64decode(image_base64)
        elif 'image' in request.files:
            image_bytes = request.files['image'].read()
        else:
            return jsonify({"error": "请提供 'image_base64' 或 'image' 文件"}), 400

        input_tensor = preprocess_image(image_bytes)  # (1,1,H,W)
        # 确保 CPU 推理（如果要用 GPU，可把 model 和 tensor 放到 cuda）
        with torch.no_grad():
            latex = decode_with_beam(GLOBAL_MODEL, input_tensor, GLOBAL_PARAMS, GLOBAL_IDX2WORD, k=1, maxlen=120)

        return jsonify({"success": True, "latex_formula": latex, "message": "识别成功"})
    except Exception as e:
        print("识别过程中发生错误:", e)
        return jsonify({"success": False, "error": f"识别失败: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)