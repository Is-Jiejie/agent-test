from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings import API_KEY, BASE_URL


def build_knowledge_base():
    print("📚 1. 正在解析本地 Markdown 知识库...")
    # 使用 Markdown 加载器
    loader = TextLoader("data/knowledge.md", encoding="utf-8")
    docs = loader.load()

    print("✂️ 2. 正在进行文本切块 (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n", " ", ""]  # 保证按 Markdown 标题切分，不破坏语意
    )
    splits = text_splitter.split_documents(docs)

    print(f"🧠 3. 正在生成 Embedding 并构建 FAISS 向量库 (共 {len(splits)} 个文本块)...")
    embeddings = OpenAIEmbeddings(
        api_key=API_KEY,
        base_url=BASE_URL,
        model="GLM-Embedding-3"  # 如果你的接口商不支持这个模型，换成他们支持的 embedding 模型即可
    )

    vectorstore = FAISS.from_documents(splits, embeddings)

    print("💾 4. 保存向量库到本地 'faiss_index' 目录...")
    vectorstore.save_local("faiss_index")
    print("✅ 知识构建完成！销售武功秘籍已注入系统。")


if __name__ == "__main__":
    build_knowledge_base()