import re
import regex

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        初始化 Tokenizer。
        
        你需要在这里建立必要的数据结构，比如：
        - bytes 到 token ID 的反向映射
        - merges 的优先级查找结构
        - 特殊 token 相关的处理
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.token_to_id = {}
        for id, token in vocab.items():
            self.token_to_id[token] = id
        self.merges_ranks = {}
        for  i, (a, b) in enumerate(merges):
            self.merges_ranks[(a, b)] = i

    def encode(self, text: str) -> list[int]:
        """
        将输入文本编码为 token ID 列表。
        
        步骤提示：
        1. 处理特殊 token（从文本中分离出来）
        2. 对非特殊部分进行 GPT-2 正则预分词
        3. 对每个预分词片段应用 BPE 合并
        4. 将合并后的 bytes 映射为 token ID
        """
        GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        escaped = []
        if self.special_tokens is not None:
            escaped = [re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)]
        pattern = ""
        if len(escaped) != 0:
            pattern = "(" + "|".join(escaped) + ")"

        seq_list = []
        segments = []
        if len(pattern) != 0:
            segments = re.split(pattern, text)
        else:
            segments = [text]
        for segment in segments:
            if self.special_tokens is not None and segment in self.special_tokens:
                seq_list.append([segment.encode("utf-8")])
                continue
            for word in regex.finditer(GPT2_PATTERN, segment):
                # print(word.group())
                seq = [bytes([b]) for b in word.group().encode("utf-8")]
                seq_list.append(seq)
        # print(seq_list)
        # escaped = [re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)]
        # print(self.special_tokens)

        # re_pattern = "(" + "|".join(escaped) + ")"
        # print(re_pattern)
        # tokens = re.split(re_pattern, text)
        # print(tokens)
        # seq_list = []
        # for token in tokens:
        #     seq_list.append([bytes([b]) for b in token.encode("utf-8")])
        # print(seq_list)
        while True:
            best_rank = float("inf")
            best_pair = None
            for seq in seq_list:
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i+1])
                    if pair in self.merges_ranks:
                        rank = self.merges_ranks[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair
            
            if best_pair is None:
                break

            new_seq_list = []
            for seq in seq_list:
                new_seq = []
                i = 0
                while i < len(seq):
                    if i != len(seq) - 1 and seq[i] == best_pair[0] and seq[i+1] == best_pair[1]:
                        new_seq.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_seq_list.append(new_seq)
            seq_list = new_seq_list

        # for merge in self.merges:
        #     new_seq_list = []
        #     for token in seq_list:
        #         new_seq = []
        #         i = 0
        #         while i < len(token):
        #             if i != len(token) - 1 and token[i] == merge[0] and token[i+1] == merge[1]:
        #                 new_seq.append(merge[0] + merge[1])
        #                 i += 2
        #             else:
        #                 new_seq.append(token[i])
        #                 i += 1
        #         new_seq_list.append(new_seq)
        #     seq_list = new_seq_list
        res = []
        for token in seq_list:
            for b in token:
                res.append(self.token_to_id[b])
        return res
        

    def decode(self, ids: list[int]) -> str:
        """
        将 token ID 列表解码为字符串。
        
        步骤提示：
        1. 将每个 ID 映射回对应的 bytes
        2. 拼接所有 bytes
        3. 解码为 UTF-8 字符串
        """
        res = b""
        if ids is None:
            return res
        for id in ids:
            res += self.vocab[id]
        res = res.decode("utf-8", errors="replace")
        return res

    def encode_iterable(self, iterable) -> ...:
        """
        内存高效地编码一个可迭代对象（如文件对象）。
        
        步骤提示：
        - 逐行读取，逐行编码
        - 使用 yield 逐个产出 token ID
        - 不要一次性把整个内容读入内存
        """
        for line in iterable:
            yield from self.encode(line)


# if __name__ == "__main__":
#     vocab = {1: b's', 2: b'a', 3: b'bc', 4: b'cde'}
#     merges = []
#     special_tokens = ["<|endoftext|>", 'a', 'bc', 'cde']
#     tokenizer = Tokenizer(vocab, merges, special_tokens)
#     print(tokenizer.encode("s"))