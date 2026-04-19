import pandas as pd
import json
import os


class TestMCPServer:
    """
    模拟拉取 API 接口文档的 Tool Server
    """

    def __init__(self):
        # 模拟一个电商链路的接口文档字典（你可以后续替换为真实解析 Swagger/YAPI 的逻辑）
        self.api_docs = {
            "login": {
                "url": "http://mock-api.com/api/v1/user/login",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": {"username": "string", "password": "string(md5)"},
                "expected_response": {"status": 200, "data": {"token": "string"}}
            },
            "create_order": {
                "url": "http://mock-api.com/api/v1/order/create",
                "method": "POST",
                "headers": {"Authorization": "Bearer <token>"},
                "body": {"item_id": "int", "quantity": "int"},
                "expected_response": {"status": 201, "data": {"order_id": "string"}}
            }
        }

    def get_tool_schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_api_documentation",
                    "description": "根据业务模块或接口名称获取接口的详细定义文档（URL、参数、响应格式等）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_name": {
                                "type": "string",
                                "description": "需要查询的业务模块名称，例如 'login' 或 'create_order'"
                            }
                        },
                        "required": ["module_name"]
                    }
                }
            }
        ]

    def call_tool(self, tool_name, arguments):
        print(f"🔌 [MCP Server] 收到工具调用请求: {tool_name}, 参数: {arguments}")

        if tool_name == "get_api_documentation":
            module_name = arguments.get('module_name')
            if module_name in self.api_docs:
                return json.dumps({
                    "api_info": self.api_docs[module_name],
                    "tips": "请根据此接口定义使用 requests 库编写测试代码并执行"
                }, ensure_ascii=False)
            else:
                return json.dumps({"error": f"未找到模块 '{module_name}' 的接口文档，请检查模块名。"}, ensure_ascii=False)
        return json.dumps({"error": f"未注册此工具: {tool_name}"})

# --- 本地独立测试 MCP Server ---
if __name__ == "__main__":
    # 假设你在项目根目录下运行此文件，需确保 data/cloud_sales.csv 存在
    server = TestMCPServer(data_path="../data/cloud_sales.csv")

    print("1. 测试获取 Schema:")
    print(server.call_tool("get_sales_data_schema", {}))

    print("\n2. 测试查询具体指标:")
    print(server.call_tool("query_sales_metrics", {"instance_type": "vGPU-RTX4090"}))