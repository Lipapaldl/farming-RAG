import numpy as np
from typing import List
from transformers import BertTokenizer, BertModel
import torch
import logging

#本地模型路径
ChineseBERTEmbeddingModelPath = r"E:\hugging-face-model\google-bert\bert-base-chinese"

class ChineseBERTEmbedding:
    def __init__(self, model_path: str =ChineseBERTEmbeddingModelPath ,device: str = None):
        """
        初始化ChineseBERT嵌入类
        :param model_path: ChineseBERT模型的路径
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.embedding_model = BertModel.from_pretrained(model_path).to(self.device)
        logging.info(f"Initialized ChineseBertEmbedding with model: {model_path} on {self.device}")

    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入
        :param text: 输入文本
        :return: 嵌入向量
        """
        try:
            # 对文本进行 token 化
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

            # 获取模型输出
            outputs = self.embedding_model(**inputs)

            # 使用最后一层隐藏状态（hidden_states）作为嵌入
            # 获取 [CLS] token 的嵌入向量，通常第一个 token 代表整个句子的嵌入
            cls_embedding = outputs.last_hidden_state[:, 0, :]

            # 转为 numpy 数组并返回
            return cls_embedding.squeeze().detach().cpu().numpy().tolist()

        except Exception as e:
            raise Exception(f"Failed to get embedding for text: {text}. Error: {str(e)}")

    @staticmethod
    def compare_vectors(vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个嵌入向量的余弦相似度
        :param vector1: 向量1
        :param vector2: 向量2
        :return: 余弦相似度
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0.0
        return dot_product / magnitude

    def compare_texts(self, text1: str, text2: str) -> float:
        """
        比较两段文本的相似性
        :param text1: 文本1
        :param text2: 文本2
        :return: 余弦相似度
        """
        try:
            embed1 = self.get_embedding(text1)
            embed2 = self.get_embedding(text2)
            return self.compare_vectors(embed1, embed2)
        except Exception as e:
            raise Exception(f"Failed to compare texts: {text1}, {text2}. Error: {str(e)}")
