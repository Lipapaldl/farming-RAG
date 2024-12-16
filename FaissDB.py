from langchain.vectorstores import FAISS #向量数据库存储
from langchain.text_splitter import RecursiveCharacterTextSplitter #文本分段
import os

#向量知识库-保存
def save_knowledge_base(knowledge_bases_path,name,documents,embeddings):
    vector_db_path = os.path.join(knowledge_bases_path,name)
    if not os.path.exists(vector_db_path):
        os.makedirs(vector_db_path,exist_ok=True)
    #加载分词模型
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    vector_db = FAISS.from_documents(texts,embeddings)
    vector_db.save_local(vector_db_path,"index")

#上传知识库
def load_knowledge_base(knowledge_bases_path,name,embeddings):
    vector_db_path = os.path.join(knowledge_bases_path,name)
    return FAISS.load_local(vector_db_path,embeddings,allow_dangerous_deserialization=True)

#刷新知识库列表
def flash_knowledge_base(knowledge_bases_path):
    os.makedirs(knowledge_bases_path, exist_ok=True)
    existing_kbs = [f for f in os.listdir(knowledge_bases_path) if os.path.isdir(os.path.join(knowledge_bases_path, f))]
    return existing_kbs