import os
import PyPDF2
import tiktoken
from typing import List

# 用于数据切分时，判断字块的 token 长度，速度比较快
enc = tiktoken.get_encoding("cl100k_base")

class TextChunker:
    def __init__(self, text: str):
        """
        初始化分块器
        :param text: 要分块的长文本
        """
        self.text = text

    def chunk_text(self, chunk_size: int, overlap: int) -> List[str]:
        """
        将长文本分块
        :param chunk_size: 每个块的 token 长度
        :param overlap: 块之间的重叠 token 数
        :return: 分块后的文本列表
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if overlap < 0:
            raise ValueError("overlap 必须为非负数")
        if chunk_size <= overlap:
            raise ValueError("chunk_size 必须大于 overlap")

        chunks = []
        start = 0

        # 使用 tiktoken 编码文本为 tokens
        tokens = enc.encode(self.text)
        total_tokens = len(tokens)

        while start < total_tokens:
            # 获取从 start 开始的 chunk_size 长度的 token
            chunk = tokens[start: start + chunk_size]
            decoded_chunk = enc.decode(chunk)  # 将 tokens 解码为文本
            chunks.append(decoded_chunk)

            # 更新起始位置，步长为 chunk_size - overlap
            start += chunk_size - overlap

            # 如果最后的 chunk 太短，可以选择丢弃或保留，这里选择保留
            if start >= total_tokens and len(chunk) < chunk_size:
                break

        return chunks