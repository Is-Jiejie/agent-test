import json

import openai
import httpx
import re


from config.settings import API_KEY, BASE_URL, MODEL_NAME, MAX_RETRIES
from sandbox.executor import execute_python_code
from tools.mcp_server import TestMCPServer
from agent.memory import MemoryManager

class QueryAgent:
    def __init__(self):
        # 初始化带代理穿透的客户端
        custom_http_client = httpx.Client(trust_env=False)
        self.client = openai.OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            http_client=custom_http_client,
        )
        self.model = MODEL_NAME

        self.mcp_server = TestMCPServer()
        self.tools = self.mcp_server.get_tool_schemas()

        self.memory = MemoryManager(max_turns=5)
        self.memory.set_system_prompt("""你是一个 QA 自动化测试 Agent。
核心工作流：
1. 必须先调用 get_api_documentation 工具获取接口定义。
2. 拿到接口信息后，编写 Python 测试代码（使用 requests 库向本地 mock 或真实环境发请求，或直接写 pytest 用例逻辑）。
3. 必须在代码中包含完整的 assert 逻辑。
4. 代码必须包裹在 ```python 和 ``` 之间。
5. 若沙盒执行报错，你必须读取报错堆栈，自主判断是用例错误还是发现了系统 Bug，如果是用例错误则修复代码后重试。""")
        # self.messages = [
        #     {
        #         "role": "system",
        #         "content": """你是一个云服务销售数据分析Agent。你的任务是解答用户的销售数据查询需求。
        #         核心工作流：
        #         1. 你绝对不能使用 open() 凭空读取本地文件。
        #         2. 当用户提问时，你必须先调用提供的 MCP 工具（如 query_sales_metrics）去获取真实的底层数据。
        #         3. 拿到工具返回的 JSON 数据后，你必须编写一段 Python 代码，将该数据硬编码在变量中并进行格式化处理，最后使用 print() 打印出面向用户的最终业务分析结论。
        #         4. 代码必须包裹在 ```python 和 ``` 之间。
        #         5. 如果触发了安全拦截或执行报错，请移除高危代码或修复错误后重新输出。"""
        #     }
        # ]

    def _extract_code(self, text):
        pattern = r"```python\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0] if matches else ""

    def run(self, user_query):
        print(f"\n[业务指令]:{user_query}")

        self.memory.add_message({"role": "user", "content": user_query})
        # self.messages.append({"role":"user", "content": user_query})

        step = 0
        while step <= MAX_RETRIES:
            step += 1
            print(f"\n--- 🔄 思考步长 (Step {step}/{MAX_RETRIES}) ---")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.memory.get_context(),
                tools=self.tools,
                temperature=0.1
            )
            message = response.choices[0].message

            if message.tool_calls:
                print("🛠️ Agent 决定调用外部 MCP 工具...")
                # 1. 必须先将大模型的工具调用请求原样加入上下文
                self.memory.add_message(message)

                # 2. 遍历执行所有它想调用的工具
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    # 解析大模型传过来的参数
                    arguments = json.loads(tool_call.function.arguments)

                    # 将参数丢给我们的 MCP Server 获取数据
                    tool_result = self.mcp_server.call_tool(function_name, arguments)
                    print(f"   └─ 成功获取底层数据 (长度: {len(tool_result)} 字符)")
                    # 3. 将拿到的真实数据以 tool 的身份追加到对话历史中
                    self.memory.add_message(
                        {
                            "role":"tool",
                            "tool_call_id":tool_call.id,
                            "name":function_name,
                            "content":tool_result
                        }
                    )
                # 工具执行完毕，直接进入下一轮循环，大模型看到数据后就会开始写代码了
                continue

            # 如果没有工具调用，说明大模型开始输出内容（期望是 Python 代码）
            agent_reply = message.content
            if agent_reply:
                self.memory.add_message({"role": "assistant", "content": agent_reply})
                code_to_run = self._extract_code(agent_reply)

                if not code_to_run:
                    print("❌ Agent 未输出标准格式的代码，提醒其重试。")
                    self.memory.add_message({"role": "user", "content": "请使用 ```python ... ``` 格式输出代码。"})
                    continue

                print("💻 [动态生成的处理代码]：\n", code_to_run)
                is_success, execution_result = execute_python_code(code_to_run)

                if is_success:
                    print("\n✅ [任务成功！最终业务结论]：\n", execution_result)
                    return execution_result
                else:
                    print(f"⚠️ [执行报错]：{execution_result}")
                    error_feedback = f"代码执行失败，报错如下：\n{execution_result}\n请分析错误原因并重写代码。"
                    self.memory.add_message({"role": "user", "content": error_feedback})
                    continue

        print("\n❌ 达到最大重试次数，任务中断。")
        return None
