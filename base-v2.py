import asyncio #异步
import streamlit as st #UI
from LLM.chatglm import ChatGLMModel #大模型接口
import logging #日志库
import pyperclip  # 引入 pyperclip 库，用于操作剪贴板

# 配置日志记录，用于捕获和记录程序运行中的错误和信息
logging.basicConfig(
    filename='log.txt',  # 日志文件名
    level=logging.DEBUG,  # 日志记录级别
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
)

# 初始化模型实例，设置生成内容的温度参数（用于控制生成内容的随机性）
model = ChatGLMModel(temperature=0.9)

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

# 定义示例问题的点击回调函数
# 点击示例按钮后，将示例问题复制到剪贴板
def set_example(example):
    pyperclip.copy(example)  # 将示例问题复制到剪贴板
    st.toast("示例问题已复制到剪贴板！")
if __name__ == '__main__':
    # 配置 Streamlit 页面布局和标题
    st.set_page_config(page_title="《齐民要术》问答系统", layout="wide", initial_sidebar_state="expanded")

    # 页面标题与描述
    st.title("《齐民要术》问答系统")
    st.markdown(
        """
        这是一个基于 RAG (Retrieval-Augmented Generation) 技术的问答系统，
        专注于《齐民要术》——中国古代农业经典著作。
        请输入您的问题，AI 将为您提供相关的答案。
        """
    )
    # 侧边栏：配置选项
    with st.sidebar:
        # 在侧边栏添加示例问题
        st.title("示例问题")
        st.button("如何选用优质的稻种？", on_click=set_example, args=("如何选用优质的稻种？",))
        st.button("‘田间管理’章节中提到的灌溉方法有哪些？", on_click=set_example,args=("‘田间管理’章节中提到的灌溉方法有哪些？",))
        st.button("《齐民要术》中对桑树种植的建议是什么？", on_click=set_example,args=("《齐民要术》中对桑树种植的建议是什么？",))
        st.image("Img/齐民要术封面.jpg", caption="《齐民要术》", use_container_width=True)

    # 输入框，用于用户输入问题
    user_input = st.text_input(
        "输入你的问题",  # 输入框的标签
        placeholder="例如：‘耕田应注意哪些事项？’"  # 输入框的占位符
    )

    # 主界面按钮，用于提交用户输入的问题并获取答案
    if st.button("提交问题"):
        if user_input:
            # 如果输入框有内容，显示处理中的提示
            st.write("正在处理您的问题，请稍候...")
            # 调用带重试机制的函数获取回答
            result = asyncio.run(chat_with_retries(user_input))
            # 显示回答内容
            st.text_area("AI的回答", value=result, height=200)
        else:
            # 如果输入框为空，显示警告信息
            st.warning("请输入您的问题后再提交！")
