# 齐民要术问答系统

## 项目简介
本项目是基于《齐民要术》构建的智能问答系统，利用现代自然语言处理技术（如 RAG 和 ChatGLM），从古代农业经典著作中提取信息并回答用户问题。项目旨在让用户方便地探索和学习《齐民要术》的知识。

---
![演示](Img/演示.png)
## 项目结构
以下是项目文件夹及其用途的详细说明：

### 1. `Book` 目录
存放《齐民要术》的原始资料和格式化后的文件。
- **`齐民要术-石声汉译注.pdf`**：带译注的《齐民要术》完整 PDF 文档。
- **`齐民要术-石声汉译注.txt`**：PDF 文档的文本版，方便进行分词和语义分析。
- **`齐民要术.txt`**：纯文本版《齐民要术》，用于知识提取和问答训练。

---

### 2. `Chunk` 目录
存储与文本分块和处理相关的代码模块。
- **`datachunk.py`**：实现将长文档拆分为更小的片段（chunks），用于知识检索和模型输入。

---

### 3. `Embedding` 目录
与向量化嵌入生成相关的代码。
- **`model.py`**：加载嵌入模型并生成文本的向量表示，用于知识检索。

---

### 4. `Img` 目录
存储与项目相关的图片资源。
- **`齐民要术封面.jpg`**：项目封面图片。

---

### 5. `knowledge` 目录
保存知识库和检索相关的数据。
- **`doecment.json`**：结构化的知识库文档，包含从《齐民要术》中提取的内容。
- **`vectors.json`**：知识库的向量表示，用于高效检索。

---

### 6. `LLM` 目录
与大语言模型（Large Language Model）交互相关的代码模块。
- **`chatglm.py`**：用于与 ChatGLM 模型进行交互的核心逻辑。

---

---

### 7. `VectorDatabase` 目录
与大语言模型（Large Language Model）交互相关的代码模块。
- **`database.py`**：实现向量数据库，用于高效的文档检索。

---

### 8. 项目根目录
- **`base-v2.py`**：项目主程序，整合知识检索和问答系统功能。
- **`get_pdf_book.ipynb`**：Jupyter Notebook，用于探索和处理《齐民要术》的文本数据。
- **`error_log.txt`**：错误日志文件，记录程序运行时的错误信息。
- **`Readme.md`**：项目说明文档。

---

## 快速开始
1. **安装依赖**  
   确保已安装所需的 Python 库：
   ```bash
   pip install -r requirements.txt
    ```
2. **修改API和模型路径**
    在LLM/chatglm.py中修改chatGLM_API 和 model_name（模型名称）

    在Embedding/model.py中修改 file_path 本地模型保存路径

3. **运行主程序**
    对PDF进行向量化存储:
    运行 jupyter get_pdf_book.ipynb

    执行以下命令启动问答系统：

    ```bash
    streamlit run base-v2.py
    ```

4. **使用说明**

    提交问题，例如“耕田应注意哪些事项？”。

    系统将从《齐民要术》中检索并生成答案。

5. **技术亮点**

    RAG（检索增强生成）：结合知识检索和生成模型，提高问答准确性。

    ChatGLM：基于大语言模型的中文问答技术。

    向量检索：实现高效的知识库查询。

6. **贡献与支持**

    欢迎对本项目进行贡献！如果有任何问题或建议，请提交 Issue 或联系作者。
    
