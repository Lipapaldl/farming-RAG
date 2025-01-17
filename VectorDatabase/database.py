from tqdm import tqdm
import numpy as np
import os
import json
from typing import List


class Vectordatabase:

    # 初始化方法，传入一个字块列表
    def __init__(self, docs: List = []) -> None:
        self.docs = docs

    # 对字块列表进行，批量的embedded编码，传入embedding模型，返回一个向量列表
    def get_vector(self, EmbeddingModel) -> List[List[float]]:
        self.vectors = []
        for doc in tqdm(self.docs):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    # 把向量列表存储到json文件中，把子块列表也存储到json文件,默认路径为'knowledge'
    def persist(self, path: str = 'knowledge') -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/doecment.json", 'w', encoding='utf-8') as f:
            json.dump(self.docs, f, ensure_ascii=False)
        with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
            json.dump(self.vectors, f)

    # 加载json文件中的向量和字块，得到向量列表、字块列表,默认路径为'knowledge'
    def load_vector(self, path: str = 'knowledge') -> None:
        with open(f"{path}/vectors.json", 'r', encoding = 'utf-8') as f:
            self.vectors = json.load(f)
        with open(f"{path}/doecment.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)

    # 求向量的余弦相似度，传入两个向量和一个embedding模型，返回一个相似度
    def get_similarity(self, vector1: List[float], vector2: List[float], Embedding_model, similarity_method:str = "cosine") -> float:
        return Embedding_model.compare_vectors(vector1, vector2, similarity_method)

    def query(self, query: str, EmbeddingModel, Reranker_model, k: int = 10, similarity_method:str = "cosine") ->  List[tuple[str, float]]:
        """
        查询文本与文档的相似性，返回相似度前 k 个的文档，支持重排序
        :param query: 查询字符串
        :param EmbeddingModel: 嵌入模型，用于获取查询字符串的嵌入
        :param k: 返回前 k 个相似文档
        :param similarity_method: 相似度计算方法 ('cosine', 'euclidean', 'weighted_cosine')
        :param reranker_model: 重排序模型（如 bge-reranker-base），用于对初步候选文档重排序
        :return: [(文档, 得分), ...]
        """
        # 获取查询向量
        query_vector = EmbeddingModel.get_embedding(query)
        # 初步检索：计算与所有文档的相似度并排序
        similarities = np.array([
            self.get_similarity(query_vector, vector, EmbeddingModel, similarity_method)
            for vector in self.vectors
        ])
        top_k_indices = similarities.argsort()[-k:][::-1]  # 初步相似度排序的前 k 个文档索引
        top_k_documents = [self.document[i] for i in top_k_indices]  # 初步候选文档
        top_k_scores = similarities[top_k_indices]  # 初步候选文档得分

        # 如果没有提供重排序模型，则直接返回初步排序结果
        if not Reranker_model:
            return list(zip(top_k_documents, top_k_scores))

        # 使用 BGE-Reranker 进行重排序
        try:
            # 使用 rerank() 方法对候选文档进行重排序
            reranked_scores, _ = Reranker_model.rerank(query, top_k_documents)

            # 返回重排序后的文档与得分
            return list(zip(top_k_documents, reranked_scores))

        except Exception as e:
            raise Exception(f"Failed to rerank candidates using BGE-Reranker. Error: {str(e)}")
