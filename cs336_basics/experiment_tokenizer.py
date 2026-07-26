# experiment_tokenizer.py
import json
import os
import pickle
import random
import time

from cs336_basics.bpe_tokenizer import train_bpe
from cs336_basics.tokenizer import Tokenizer

SPECIAL_TOKENS = ["<|endoftext|>"]
TINYSTORIES_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "tinystories_sample_5M.txt")


def save_tokenizer(vocab, merges, path):
    """保存 tokenizer 到文件"""
    with open(path, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)


def load_tokenizer(path, special_tokens=None):
    """从文件加载 tokenizer"""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return Tokenizer(data["vocab"], data["merges"], special_tokens)


def compute_compression_ratio(tokenizer, text):
    """计算压缩比 = 字节数 / token数"""
    byte_len = len(text.encode("utf-8"))
    token_len = len(tokenizer.encode(text))
    if token_len == 0:
        return 0
    return byte_len / token_len


def sample_documents(file_path, n=10, seed=42):
    """从文件中随机采样 n 个文档（按段落分割）"""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # 按双换行分割成文档
    docs = [d.strip() for d in text.split("\n\n") if d.strip()]
    random.seed(seed)
    sampled = random.sample(docs, min(n, len(docs)))
    return sampled


def measure_throughput(tokenizer, text):
    """测量吞吐量 (bytes/second)"""
    byte_len = len(text.encode("utf-8"))
    start = time.time()
    tokenizer.encode(text)
    elapsed = time.time() - start
    if elapsed == 0:
        return float("inf")
    return byte_len / elapsed


def main():
    # ========================================
    # 步骤 1: 训练 TinyStories tokenizer (10K vocab)
    # ========================================
    tokenizer_path = "tinystories_tokenizer.pkl"
    if os.path.exists(tokenizer_path):
        print("加载已保存的 TinyStories tokenizer...")
        tokenizer = load_tokenizer(tokenizer_path, SPECIAL_TOKENS)
    else:
        print("训练 TinyStories tokenizer (vocab_size=10000)...")
        vocab, merges = train_bpe(TINYSTORIES_PATH, 10000, SPECIAL_TOKENS)
        save_tokenizer(vocab, merges, tokenizer_path)
        tokenizer = Tokenizer(vocab, merges, SPECIAL_TOKENS)
        print("训练完成并已保存")

    # ========================================
    # (a) 压缩比实验
    # ========================================
    print("\n=== (a) 压缩比实验 ===")
    docs = sample_documents(TINYSTORIES_PATH, n=10)
    ratios = []
    for i, doc in enumerate(docs):
        ratio = compute_compression_ratio(tokenizer, doc)
        ratios.append(ratio)
        print(f"  文档 {i+1}: {len(doc.encode('utf-8'))} bytes, "
              f"{len(tokenizer.encode(doc))} tokens, ratio={ratio:.4f}")
    avg_ratio = sum(ratios) / len(ratios)
    print(f"  平均压缩比: {avg_ratio:.4f} bytes/token")

    # ========================================
    # (b) 交叉分词实验（需要 OpenWebText，此处跳过）
    # ========================================
    print("\n=== (b) 交叉分词实验 ===")
    print("  （需要 OpenWebText 数据集和 tokenizer，暂未实现）")

    # ========================================
    # (c) 吞吐量测量
    # ========================================
    print("\n=== (c) 吞吐量测量 ===")
    with open(TINYSTORIES_PATH, "r", encoding="utf-8") as f:
        full_text = f.read()
    sample_text = full_text[:10000]
    throughput = measure_throughput(tokenizer, sample_text)
    print(f"  吞吐量: {throughput:.2f} bytes/second")
    pile_size = 825 * 1024 ** 3  # 825GB
    time_needed = pile_size / throughput
    print(f"  分词 Pile (825GB) 预计需要: {time_needed:.2f} 秒 = {time_needed/3600:.2f} 小时")

    # ========================================
    # (d) uint16 说明
    # ========================================
    print("\n=== (d) uint16 说明 ===")
    max_vocab_id = max(tokenizer.vocab.keys())
    print(f"  词表最大 ID: {max_vocab_id}")
    print(f"  uint16 范围: 0 ~ 65535")
    print(f"  vocab_size <= 65535 时 uint16 足够，且比 uint32 节省一半存储空间")


if __name__ == "__main__":
    main()
