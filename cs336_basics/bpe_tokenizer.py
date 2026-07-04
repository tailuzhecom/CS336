import regex
import os
from collections import defaultdict
import re

# def train_bpe(
#     input_path: str | os.PathLike,
#     vocab_size: int,
#     special_tokens: list[str],
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#     pass

def impl1(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # GPT-2 预分词正则
    GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    text = "low low low low low lower lower newest newest newest newest newest newest"
    # pre_tokens = regex.findall(GPT2_PATTERN, text)
    # print(pre_tokens)
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    for i, token in enumerate(special_tokens):
        vocab[i] = token.encode("utf-8")

    for token in range(256):
        vocab[len(vocab)] = bytes([token])

    cur_vocab = vocab
    # merge_freq = dict[tuple[bytes, bytes], int]
    pre_tokens_freqs : dict[bytes, int] = {}
    for match in regex.finditer(GPT2_PATTERN, text):
        key = match.group().encode("utf-8")
        pre_tokens_freqs[key] = pre_tokens_freqs.get(key, 0) + 1
    
    freqs : dict[bytes, int] = {}
    pair_freqs : dict[tuple[bytes, bytes], int] = {}
    for key, value in pre_tokens_freqs.items():
        for i, token in enumerate(key):
            freqs[token] = freqs.get(token, 0) + value
            if i < len(key) - 1:
                pair = (key[i], key[i+1])
                pair_freqs[pair]= pair_freqs.get(pair, 0) + value
    print(pair_freqs)
    max_pair = max(pair_freqs, key=lambda x: (pair_freqs[x], x))
    print("max_pair: ", max_pair, ", max_pair_freq: ", pair_freqs[max_pair])
    merges.append(max_pair)
    for pre_token, freq in pre_tokens_freqs.items():
        if max_pair[0] in pre_token:
            pass


# vocab[tuple(bytes)] - > freq，tuple(bytes)为一个单词，freq为该单词出现的次数
def get_vocab(input_path: str | os.PathLike, special_tokens=None):
    # text = "low low low low low lower lower newest newest newest newest newest newest"
    GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    vocab = defaultdict(int)

    escaped = []
    if special_tokens is not None:
        escaped = [re.escape(t) for t in special_tokens]
    pattern = "|".join(escaped)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    segments = []
            # line = line.strip()  # 去掉末尾换行符和空白
    if len(pattern) != 0:
        segments = re.split(pattern, text)
    else:
        segments = [text]
    for segment in segments:
        for word in regex.finditer(GPT2_PATTERN, segment):
            # print(word.group())
            seq = tuple(bytes([b]) for b in word.group().encode("utf-8"))
            vocab[seq] += 1        
    return vocab

def get_pair_freq(vocab):
    pair_freq = defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair_freq[word[i], word[i+1]] += freq
    return pair_freq

# def merge_token(max_pair, vocab_in):
#     vocab_out = defaultdict(int)
#     # p_raw = re.escape(' '.join(max_pair))
#     # print(p_raw)
#     # pattern = re.compile(r'(?<!\S)' + p_raw + r'(?!\S)')
#     for word, freq in vocab_in.items():
#         # w_out = pattern.sub(''.join(max_pair), word)
#         # print(w_out)
#         new_seq = []
#         new_token = max_pair[0] + max_pair[1]
#         i = 0
#         while i < len(word):
#             if i < len(word) - 1 and word[i] == max_pair[0] and word[i+1] == max_pair[1]:
#                 new_seq.append(new_token)
#                 i += 2
#             else:
#                 new_seq.append(word[i])
#                 i += 1
#         vocab_out[tuple(new_seq)] = freq
#     return vocab_out

def merge_token(max_pair, vocab, pair_freqs):
    vocab_out = defaultdict(int)
    a, b = max_pair
    new_token = a + b
    for word, freq in vocab.items():
        new_seq = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i+1] == b:
                pair_freqs[(a, b)] = pair_freqs.get((a, b), 0) - freq
                if len(new_seq) > 0:
                    # 原来(new_seq[-1], a)能组合，现在只能和new_token组合，所以减去这个单词出现的次数
                    pair_freqs[(new_seq[-1], a)] = pair_freqs.get((new_seq[-1], a), 0) - freq
                    pair_freqs[(new_seq[-1], new_token)] = pair_freqs.get((new_seq[-1], new_token), 0) + freq
                if i < len(word) - 2:
                    # 原来(b, word[i+2])能组合，现在只能和new_token组合，所以减去这个单词出现的次数
                    pair_freqs[(b, word[i+2])] = pair_freqs.get((b, word[i+2]), 0) - freq
                    pair_freqs[(new_token, word[i+2])] = pair_freqs.get((new_token, word[i+2]), 0) + freq
                new_seq.append(new_token)
                i += 2
            else:
                new_seq.append(word[i])
                i += 1
        vocab_out[tuple(new_seq)] = freq
    return vocab_out


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    final_vocab = defaultdict(int)
    merges: list[tuple[bytes, bytes]] = []
    for i, token in enumerate(special_tokens):
        final_vocab[i] = token.encode("utf-8")

    for token in range(256):
        final_vocab[len(final_vocab)] = bytes([token])

    vocab = get_vocab(input_path, special_tokens)
    pair_freqs = get_pair_freq(vocab)
    # print(vocab)
    round = 0
    while len(final_vocab) < vocab_size:
        # print("round: ", round)
        round += 1
        
        max_pair = max(pair_freqs, key=lambda x: (pair_freqs[x], x))
        # print("max_pair: ", max_pair, ", max_pair_freq: ", pair_freqs[max_pair])
        vocab = merge_token(max_pair, vocab, pair_freqs)
        # print(vocab)
        
        merges.append(max_pair)
        final_vocab[len(final_vocab)] = max_pair[0] + max_pair[1]
    
    return final_vocab, merges

    

# if __name__ == "__main__":
#     # vocab = get_vocab()
#     # loop_times = 2
#     # for l in range(loop_times):
#     #     print("loop: ", l)
#     #     pair_freq = get_pair_freq(vocab)
#     #     max_pair = max(pair_freq, key=lambda x: (pair_freq[x], x))
#     #     print("max_pair: ", max_pair, ", max_pair_freq: ", pair_freq[max_pair])
#     #     vocab = merge_token(max_pair, vocab)
#     #     print(vocab)
#     vocab, merges = train_bpe("./data/bpe_test_data", 260, []) 
#     print("vocab: ", vocab)
#     print("merges: ", merges)