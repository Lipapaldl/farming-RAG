import asyncio #异步
import streamlit as st #UI
from LLM.chatglm import ChatGLMModel #大模型接口
import logging #日志库

# 配置日志记录，用于捕获和记录程序运行中的错误和信息
logging.basicConfig(
    filename='log.txt',  # 日志文件名
    level=logging.DEBUG,  # 日志记录级别
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
)


# 定义异步函数，用于通过模型处理用户问题
async def chat_with_rag(message: str) -> str:
    try:
        # 调用模型的 chat_with_rag 方法获取回答
        result = await model.chat_with_rag(message)
        return result
    except Exception as e:
        # 如果发生错误，记录错误日志并返回错误信息
        logging.error(f"Error occurred while fetching result: {str(e)}")
        return f"Error: Unable to process your request. Details: {str(e)}"

# 定义带重试机制的异步函数
async def chat_with_retries(message: str, retries: int = 3, delay: int = 10):
    """
    带重试功能的模型交互函数。
    :param message: 用户输入的消息
    :param retries: 最大重试次数
    :param delay: 每次重试之间的延迟时间（秒）
    :return: 模型生成的回答或错误信息
    """
    result = ""
    for attempt in range(retries):
        try:
            logging.info(f"Attempt {attempt + 1}/{retries}: Calling chat_with_rag")

            # 调用异步函数获取结果
            result = await chat_with_rag(message)
            return result
        except Exception as e:
            # 如果发生错误，记录错误日志并在重试前延迟
            logging.error(f"Error on attempt {attempt + 1}: {str(e)}", exc_info=True)
            if attempt < retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)

    # 如果超过最大重试次数仍失败，返回错误信息
    return "Error: Unable to process your request after multiple attempts."
# 定义示例问题的按钮功能
@st.cache_data()
def set_example(example):
    st.session_state["user_input"] = example  # 更新输入框的内容
@st.cache_data()
def main():
    # 页面标题与描述
    st.title("《齐民要术》问答系统")
    st.markdown(
        """
        这是一个基于 RAG (Retrieval-Augmented Generation) 技术的问答系统，
        专注于《齐民要术》——中国古代农业经典著作。
        请输入您的问题，AI 将为您提供相关的答案。
        """
    )

if __name__ == '__main__':
    # 配置 Streamlit 页面布局和标题
    st.set_page_config(page_title="《齐民要术》问答系统", layout="wide", initial_sidebar_state="expanded")
    main()
    # 侧边栏：配置选项
    with st.sidebar:
        # 在侧边栏添加示例问题
        st.title("示例问题")
        st.button("如何选用优质的稻种？", on_click=set_example, args=("如何选用优质的稻种？",))
        st.button("‘田间管理’章节中提到的灌溉方法有哪些？", on_click=set_example,args=("‘田间管理’章节中提到的灌溉方法有哪些？",))
        st.button("《齐民要术》中对桑树种植的建议是什么？", on_click=set_example,args=("《齐民要术》中对桑树种植的建议是什么？",))
        st.image("Img/齐民要术封面.jpg", caption="《齐民要术》", use_container_width=True)
        

    # 初始化 Session State
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""
    with st.sidebar:
        st.title("模型与算法配置")

        # Embedding 模型选择
        st.subheader("Embedding 模型")
        embedding_model = st.selectbox(
            "选择用于嵌入向量生成的模型：",
            options=["google-bert/bert-base-chinese", "maidalun1020/bce-embedding-base_v1"],
            index=0,
            key="embedding_model"
        )

        # Rerank 模型选择
        st.subheader("Rerank 模型")
        rerank_model = st.selectbox(
            "请选择用于重排序的模型：",
            options=["BAAI/bge-reranker-base", "BM25"],
            index=0,
            key="rerank_model"
        )

        # 相似度计算算法选择
        st.subheader("匹配算法")
        similarity_algorithm = st.selectbox(
            "请选择相似度计算方式：",
            options=["cosine", "euclidean", "weighted_cosine"],
            index=0,
            key="similarity_algorithm"
        )

        # LLM 模型选择
        st.subheader("LLM 模型")
        llm_model = st.selectbox(
            "请选择大语言模型：",
            options=["glm-3-turbo"],
            index=0,
            key="llm_model"
        )

        # 分割线
        st.markdown("---")

        # 显示用户当前选择的配置
        st.subheader("当前选择的配置")
        st.write(f"**Embedding 模型**: {st.session_state['embedding_model']}")
        st.write(f"**Rerank 模型**: {st.session_state['rerank_model']}")
        st.write(f"**相似度计算算法**: {st.session_state['similarity_algorithm']}")
        st.write(f"**LLM 模型**: {st.session_state['llm_model']}")

    # 初始化模型实例，设置生成内容的温度参数（用于控制生成内容的随机性）
    model = ChatGLMModel(temperature=0.9,
                         Embedding=st.session_state['embedding_model'],
                         Rerank=st.session_state['rerank_model'],
                         similarity_algorithm=st.session_state['similarity_algorithm'])

    # 主要内容区域：问答输入框
    with st.form("question_form"):
        user_input = st.text_input(label="请输入您的问题：",
                                   placeholder="例如：‘耕田应注意哪些事项？’",
                                   value=st.session_state["user_input"],  # 设置为 Session State 中的内容
                                   max_chars=200)
         
        submit = st.form_submit_button("提交问题")
        if submit:
            if user_input:
                st.write("正在处理您的问题，请稍候...")
                result = asyncio.run(chat_with_retries(user_input))
                if 'info' in result:
                    info = result['info']
                    st.subheader("检索到的相关文档及得分")
                    # 按照得分降序排列
                    table_data = sorted(
                        [{"文档": doc, "得分": round(score, 4)} for doc, score in info],
                        key=lambda x: x["得分"],
                        reverse=True  # 降序排列
                    )
                    st.table(table_data)
                st.text_area("AI的回答", value=result['result'], height=200)
            else:
                st.warning("请输入您的问题后再提交！")    