import time
import jwt
import requests
import streamlit as st #网页UI
from langchain_community.embeddings import HuggingFaceEmbeddings#文本嵌入

#参数配置
model_name_embedding = r'E:\hugging-face-model\BAAI\bge-small-zh-v1.5' #嵌入模型
model_name_rerank = r'E:\hugging-face-model\BAAI\bge-reranker-base' #重排序模型
# "7d0a347a309aa30360681614e8d51b69.rLCJISxoyHflCRmJ"


#LLM智谱大模型API
def generate_token(apikey: str, exp_seconds: int):
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

def ask_glm(key,model,max_tokens,temperature,content):
    #智谱大模型
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
      'Content-Type': 'application/json',
      'Authorization': generate_token(key, 1000)
    }

    data = {
        "model": model,
        "max_tokens":max_tokens,
        "temperature":temperature,
        "messages": content
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()

#清除历史对话
def clear_chat_history():
    st.session_state.messages = [{"role":"assistant","content":"有什么可以帮助你的？"}]

if __name__ == '__main__':
    st.set_page_config(page_title="智慧养殖大模型")
    with st.sidebar:
        #配置API
        if 'API_TOKEN' in st.session_state and len(st.session_state['API_TOKEN']) > 1:
            st.success('API Token已经配置', icon='✔')
            key = st.session_state['API_TOKEN']
        else:
            key = ""

        key = st.text_input("输入Token", type='password', value=key)
        st.session_state['API_TOKEN'] = key

        #选择模型，相关配置
        model = st.selectbox("选择模型", ["glm-3-turbo", "glm-4"])
        max_tokens = st.slider("max_tokens", 0, 2000, value=512)
        temperature = st.slider("temperature", 0.0, 2.0, value=0.8)

    st.sidebar.button('清空聊天记录', on_click=clear_chat_history())

    # 初始化对话
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "有什么可以帮助你的？"}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
    #加载嵌入模型        
    embeddings = HuggingFaceEmbeddings(model_name = model_name_embedding)

    #检查key，处理用户输入
    if len(key) > 1:
        if user_input := st.chat_input():
            try:
                #写入用户输入
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                #写入大模型回答
                with st.chat_message("assistant"):
                    with st.spinner("请求中..."):
                        full_response = \
                        ask_glm(key, model, max_tokens, temperature, st.session_state.messages)['choices'][0]['message']['content']
                        st.markdown(full_response)
                        message = {"role": "assistant", "content": full_response}
                        st.session_state.messages.append(message)
            except ValueError as e:
                st.error(f"Error loading FAISS index: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")