import asyncio
import logging
import time
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from VectorDatabase.database import Vectordatabase
from Embedding.model import ChineseBERTEmbedding
import jwt

#你的API
chatGLM_API = "8b00afab1e7440a29b2cc9dba6e17f94.pjG00RsmnmyiHqsG"
model_name = 'glm-3-turbo'

# 配置日志
logging.basicConfig(
    filename='log.txt',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ChatGLMModel:
    def __init__(self, temperature: float = 0.9) -> None:
        self.api_key = chatGLM_API
        self.model_name = model_name
        self.temperature = temperature

        # 初始化
        self.db = Vectordatabase()
        self.db.load_vector()  # 加载向量数据库

        # 嵌入模型
        self.embedding_model = ChineseBERTEmbedding()

    async def chat_with_rag(self, question: str) -> str:
        retries = 3  # 最大重试次数
        timeout = 120  # 请求超时时间

        # 查询上下文
        info = self.db.query(question, self.embedding_model, 1)

        if not info:
            info = "无相关上下文。"  # 如果没有找到上下文，返回默认的占位提示

        template = f"""使用以下上下文回答问题，你需要结合你自己的理解回答的完善一点。若没获取到相关结果，请根据你自己的知识库回答问题：
        问题: {question}
        可参考的上下文：
        ···
        {info}
        ···
        你的回答："""

        # 配置requests的重试机制
        session = requests.Session()
        retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount('https://', adapter)

        for attempt in range(retries):
            try:
                start_time = time.time()
                logging.info(f"Sending request with question: {question}")

                # 发起请求，调用ChatGLM API
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.ask_glm(
                        key=self.api_key,
                        model=self.model_name,
                        max_tokens=2000,
                        temperature=self.temperature,
                        content=[{'role': 'user', 'content': template}]
                    )
                )

                end_time = time.time()
                logging.info(f"Request completed in {end_time - start_time:.2f} seconds.")

                result = response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

                if not result:
                    logging.warning("Model returned empty result. Using default response.")
                    result = "没有获取到有效的回答。"

                return result

            except requests.exceptions.Timeout as e:
                logging.error(f"Request timed out: {str(e)}")
                if attempt < retries - 1:
                    logging.info(f"Retrying... ({attempt + 1}/{retries})")
                    await asyncio.sleep(30)  # 重试前等待30秒
                else:
                    return "Error: Request timed out after multiple attempts."
            except Exception as e:
                logging.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
                return f"Error: Unable to process your request. Details: {str(e)}"

    # LLM智谱大模型API
    def generate_token(self,apikey: str, exp_seconds: int):
        try:
            id, secret = apikey.split(".")
        except Exception as e:
            raise Exception("invalid apikey", e)
        payload = {
            "api_key": id,
            "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
            "timestamp": int(round(time.time() * 1000)),
        }
        return jwt.encode(
            payload,
            secret,
            algorithm="HS256",
            headers={"alg": "HS256", "sign_type": "SIGN"},
        )

    def ask_glm(self,key, model, max_tokens, temperature, content):
        # 智谱大模型
        url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': self.generate_token(key, 1000)
        }

        data = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": content
        }

        response = requests.post(url, headers=headers, json=data)

        return response.json()