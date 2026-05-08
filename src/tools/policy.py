"""
政策查询工具 - RAG 检索实现
=============================

这个文件实现了基于向量检索的政策文档查询功能（RAG）。

工作流程：
1. 从远程URL下载政策文档（瑞士航空FAQ）
2. 将文档分割成块
3. 使用Embedding模型将文档块转换为向量
4. 构建向量检索器，支持语义相似度搜索
5. 提供lookup_policy工具供Agent调用
"""

import os
import re

import numpy as np
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from openai import OpenAI

from src.config import db

load_dotenv()

class VectorStoreRetriever:
    """
    向量存储检索器

    实现简单的向量检索功能，用于语义搜索政策文档。

    核心原理：
    1. 将文档通过Embedding模型转换为高维向量
    2. 查询时，将查询文本也转换为向量
    3. 计算查询向量与所有文档向量的相似度
    4. 返回最相似的K个文档
    """

    def __init__(self, docs: list, vectors: list, client):
        """
        初始化检索器

        Args:
            docs: 原始文档列表，每个文档是一个包含page_content的字典
            vectors: 文档对应的向量列表（与docs顺序对应）
            client: OpenAI客户端，用于生成embedding
        """
        self._arr = np.array(vectors)  # 将向量列表转换为numpy数组，便于计算
        self._docs = docs               # 保存原始文档
        self._client = client           # 保存OpenAI客户端

    @classmethod
    def from_docs(cls, docs, client):
        """
        工厂方法：从文档列表创建检索器

        自动将所有文档转换为向量。

        Args:
            cls: 类本身，在这里是VectorStoreRetriever
            docs: 文档列表
            client: OpenAI客户端

        Returns:
            VectorStoreRetriever实例
        """
        # 调用OpenAI的Embedding接口，将所有文档内容转换为向量
        embeddings = client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL_ID", "text-embedding-v3"),
            input=[doc["page_content"] for doc in docs]
        )
            # ========== 第2步：提取向量列表 ==========
            # embeddings.data 是一个列表，包含所有文档的embedding结果
            # 每个元素有 .embedding 属性，是实际的向量（Python列表）    
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        """
        检索与查询最相关的文档

        使用余弦相似度计算，返回最相似的K个文档。

        Args:
            query: 查询文本（如："改签政策"）
            k: 返回的文档数量，默认为5

        Returns:
            文档列表，每个文档包含page_content和similarity分数
        """
        # 1. 将查询文本转换为向量
        embed = self._client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL_ID", "text-embedding-v3"),
            input=[query]
        )
        query_vector = np.array(embed.data[0].embedding)

        # 2. 计算查询向量与所有文档向量的点积（相似度）
        # 由于embedding向量已经归一化，点积等价于余弦相似度
        # @ 表示矩阵乘法，这里用于计算查询向量与所有文档向量的点积
        # self._arr.T 是文档向量的转置，确保点积计算是行与行之间
        scores = query_vector @ self._arr.T

        # 3. 找到得分最高的K个文档
        # np.argpartition 是一个高效的部分排序算法
        # [-k:] 获取最大的K个索引
        top_k_idx = np.argpartition(scores, -k)[-k:]

        # 4. 按得分从高到低排序
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]

        # 5. 返回文档及其相似度分数
        return [
            {**self._docs[idx], "similarity": scores[idx]}
            for idx in top_k_idx_sorted
        ]

# ==================== 第1步：下载政策文档 ====================
# 从Google Cloud Storage下载瑞士航空的政策FAQ文档
response = requests.get(
    "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
)
# 检查请求是否成功，若失败则抛出异常，若成功就继续往下执行
# 这可以确保在下载文档时不会因为网络问题而导致程序崩溃
response.raise_for_status()
# 从响应中提取文本内容，转换为字符串，数据在内存中，没有下载到本地电脑
faq_text = response.text


# ==================== 第2步：文档分块 ====================
# 将markdown文档按 "## " 标题分割成独立的文档块
# 每个块代表一个独立的政策主题（如：退改签政策、行李规定等）
# re.split(r"(?=\n##)", faq_text) 使用前瞻断言，在每个 "## " 前面分割
docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]

# ==================== 第3步：初始化检索器 ====================
# 创建OpenAI客户端（用于embedding）
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# 从文档构建检索器（这一步会调用OpenAI API生成所有文档的embedding）
retriever = VectorStoreRetriever.from_docs(docs, client)


# ==================== 第4步：定义LangChain工具 ====================
@tool
def lookup_policy(query: str) -> str:
    """
    查询公司政策文档

    这是一个LangChain工具，Agent可以通过它查询政策信息。
    在执行敏感操作（如改签、取消）前，Agent会调用此工具确认政策允许的操作。

    Args:
        query: 用户查询（如："免费改签条件"、"行李托运费"）

    Returns:
        最相关的政策文档内容

    Example:
        >>> lookup_policy("我可以免费改签吗？")
        '## 改签政策\\n\\n根据航空公司的规定...'
    """
    # 从检索器获取最相关的2个文档
    docs = retriever.query(query, k=2)
    # 将多个文档内容用换行连接
    return "\n\n".join([doc["page_content"] for doc in docs])
