import httpx
import datetime
import os


from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import embeddings

from config.prompts import get_sales_agent_prompt
from config.settings import API_KEY, BASE_URL, MODEL_NAME

from tools.mcp_server import SalesMCPServer
from sandbox.executor import execute_python_code

mcp_server = SalesMCPServer()


print("加载 RAG 知识库中...")
embeddings = OpenAIEmbeddings(api_key=API_KEY, base_url=BASE_URL,model="GLM-Embedding-3")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 动态拼接出绝对路径：D:\...\AI_Sales_Agent\faiss_index
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

# 使用绝对路径加载向量库，无论从哪里运行都不会报错！
knowledge_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = knowledge_db.as_retriever(search_kwargs={"k": 2})

@tool
def query_sales_metrics_tool(instance_type: str, region: str = None) -> str:
    """从底层数据库安全查询特定产品或区域的销售汇总指标。参数为 instance_type (如 vGPU-RTX4090) 和可选的 region (如 华北-北京)。"""
    return mcp_server.call_tool("query_sales_metrics", {"instance_type": instance_type, "region": region})

@tool
def execute_python_code_tool(code: str) -> str:
    """在安全的 AST 沙盒中执行 Python 代码进行数据计算，并返回 print() 的最终结果。传入参数为完整的 Python 代码字符串。"""
    # 里面直接调用我们的沙盒拦截与执行器
    is_success, result = execute_python_code(code)
    return result





@tool
def query_knowledge_base_tool(query: str) -> str:
    """当用户询问产品技术规格(如显存/CUDA)、适用场景、竞品对比话术、退款与促销政策等文本规则时，必须调用此工具。传入具体问题。"""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# 把工具打包成列表，一会塞给 Agent
tools_list = [query_sales_metrics_tool, execute_python_code_tool, query_knowledge_base_tool]

class LangChainSalesAgent:
    def __init__(self, max_context_turns):
        custom_http_client = httpx.Client(trust_env=False)
        self.llm = ChatOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL_NAME,
            temperature=0.1,
            http_client=custom_http_client
        )
        self.memory = ConversationBufferWindowMemory(
            k=max_context_turns,
            memory_key ="chat_history",
            return_messages = True
        )
        # prompt = ChatPromptTemplate.from_messages([
        #     ("system", """你是一个云服务销售数据分析Agent。
        #     核心工作流：
        #     1. 绝对不能使用 open() 凭空读取本地文件。
        #     2. 必须先调用 `query_sales_metrics_tool` 去获取真实的数据。
        #     3. 拿到数据后，必须调用 `execute_python_code_tool` 编写并在沙盒中执行 Python 代码计算结果。
        #     4. 如果代码执行报错或触发安全拦截，请根据报错信息反思并重新调用工具。"""),
        #     # 插入刚才定义的 memory 历史记录
        #     MessagesPlaceholder(variable_name="chat_history"),
        #     ("user", "{input}"),
        #     # LangChain 必备：存放 Agent "思考-调用工具-拿到结果" 这一连串中间过程的草稿本
        #     MessagesPlaceholder(variable_name="agent_scratchpad"),
        # ])
        base_prompt = get_sales_agent_prompt()
        prompt = base_prompt.partial(
            core_products="vGPU-RTX系列, 通用计算型",
            data_tool_name="query_sales_metrics_tool",
            code_tool_name="execute_python_code_tool",
            rag_tool_name="query_knowledge_base_tool",  # 注入 RAG 工具名
            current_date=datetime.date.today().strftime("%Y-%m-%d")
        )

        agent = create_tool_calling_agent(self.llm, tools_list, prompt)
        self.agent_executor = AgentExecutor(
            agent = agent,
            tools = tools_list,
            memory = self.memory,
            verbose = True,
            max_iterations = 6,
            handle_parsing_errors = True
        )

    def run(self, user_query):
        # print(f"\n[业务指令]: {user_query}")
        response = self.agent_executor.invoke({"input":user_query})

        # print("\n✅ [任务成功！最终业务结论]：\n", response["output"])
        return response["output"]