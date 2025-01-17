import numpy as np
from typing import List
from transformers import BertTokenizer, BertModel
import torch
import logging
from transformers import AutoTokenizer,AutoModelForSequenceClassification

#本地模型存放路径
file_path = "E:/hugging-face-model/"

class ChineseBERTEmbedding:
    def __init__(self, model_name: str ,device: str = None):
        """
        初始化ChineseBERT嵌入类
        :param model_name: ChineseBERT模型的名称
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(file_path + model_name)
        self.embedding_model = BertModel.from_pretrained(file_path + model_name).to(self.device)
        logging.info(f"Initialized ChineseBertEmbedding with model: {file_path + model_name} on {self.device}")

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
    def compare_vectors(vector1: List[float], vector2: List[float], method: str = "cosine") -> float:
        """
        计算两个嵌入向量的余弦相似度
        :param vector1: 向量1
        :param vector2: 向量2
        :param method: 相似度计算方法 ('cosine', 'euclidean', 'weighted_cosine')
        :return: 相似度分数
        """
        #基于余弦相似度
        if method == "cosine":
            dot_product = np.dot(vector1, vector2)
            magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
            return dot_product / magnitude if magnitude else 0.0
        #基于负的欧几里得距离，数值越大越相似
        elif method == "euclidean":
            return -np.linalg.norm(np.array(vector1) - np.array(vector2))   
        #基于加权余弦相似度
        elif method == "weighted_cosine":
            weights = np.array([1.0] * len(vector1))
            dot_product = np.dot(weights * vector1, weights * vector2)
            magnitude = np.linalg.norm(weights * vector1) * np.linalg.norm(weights * vector2)
            return dot_product / magnitude if magnitude else 0.0
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'cosine', 'euclidean', or 'weighted_cosine'.")
    
    @staticmethod
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

class BGEReranker:
    def __init__(self, model_name: str , device: str = None):
        """
        初始化 BGE-Reranker 模型
        :param model_path: 模型路径或名称
        :param device: 使用设备（CPU 或 CUDA）
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(file_path + model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(file_path + model_name).to(self.device)

    def get_embedding(self, text: str) -> np.ndarray:
        """
        使用 BGE-Reranker 模型获取文本嵌入
        :param text: 输入文本
        :return: 文本嵌入向量
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用 [CLS] token 的嵌入作为句子向量
                cls_embedding = outputs.last_hidden_state[:, 0, :]
            return cls_embedding.squeeze().cpu().numpy()
        except Exception as e:
            raise Exception(f"Failed to get embedding for text: {text}. Error: {str(e)}")
        
    def rerank(self, query: str, candidates: List[str]) -> tuple[List[float], int]:
        """
        使用 BGE-Reranker 对候选文档进行重排序
        :param query: 查询文本
        :param candidates: 候选文档列表
        :return: (候选文档的得分列表, 得分最高的页面索引)
        """
        try:
            # 将查询和候选文档配对
            inputs = self.tokenizer(
                [query] * len(candidates),
                candidates,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)

            # 计算得分
            with torch.no_grad():
                scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
            
            # 将得分转移到 CPU，并获取最大得分索引
            scores_cpu = scores.cpu().numpy()
            max_score_idx = scores_cpu.argmax()

            return scores_cpu.tolist(), max_score_idx
        except Exception as e:
            raise Exception(f"Failed to rerank candidates. Error: {str(e)}")