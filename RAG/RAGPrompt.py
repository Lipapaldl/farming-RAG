import torch
from langchain.prompts import ChatPromptTemplate
from transformers import AutoTokenizer,AutoModelForSequenceClassification #重排序模型



class RAG:
    def __init__(self,embedding,model_name_rerank,vectstore):
        #embedding模型
        self.embedding = embedding

        #向量数据库
        self.vectstore = vectstore

        #简单提示词工程
        self.prompts = ChatPromptTemplate.from_template("""
           根据上下文回答以下问题,不要自己发挥，要根据以下参考内容总结答案，如果以下内容无法得到答案，就返回无法根据参考内容获取答案\n
           以下是知识库中检索到的信息：\n{retrieved_knowledge}\n
           问题：{user_input}""")

        #tokenizer分词模型
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_name_rerank)

        #重排序模型
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_rerank).cuda()

    #文档格式化
    def format_docs(self,docs):
        return "\n".join([doc.page_content for doc in docs])

    #文档检索
    def retrieve_documents(self,user_input,k=5):
        retrieved_docs = self.vectstore.similarity_search(user_input, k=k)
        return retrieved_docs

    #文档重排序
    def rerank_documents(self,user_input,retrieved_docs):
        # 如果没有传入重排序模型，直接返回原始文档
        if not self.rerank_model:
            return retrieved_docs
        # 为了给每个文档评分，需要把输入文本和文档内容拼接在一起

        inputs = []
        for doc in retrieved_docs:
            input_text = user_input + "\n" + doc.page_content
            inputs.append(input_text)
        #分词处理
        inputs_encodings = self.tokenizer(inputs, truncation=True, padding=True, return_tensors="pt").to('cuda')

        #计算相关性
        with torch.no_grad():
            outputs = self.rerank_model(**inputs_encodings)
            scores = outputs.logits.squeeze().cpu().numpy()

        #重排序
        sorted_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True)]
        return sorted_docs

    #1.传统方式召回，单问题召回，然后llm总结答案回答
    def simple_chunk(self,user_input):
        #传统召回：从数据库中召回文档
        retrieve_docs = self.retrieve_documents(user_input)
        #重排序
        reranked_docs = self.rerank_documents(user_input,retrieve_docs)
        #格式化文档内容
        formatted_docs = self.format_docs(reranked_docs)

        #提示词
        prompt = self.prompts.format(retrieved_knowledge=formatted_docs,user_input=user_input)

        return prompt

