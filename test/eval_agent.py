# tests/eval_harness.py
from agent.langchain_agent import LangChainSalesAgent
import time


def run_evaluation():
    agent = LangChainSalesAgent(max_context_turns=1)

    # 构建测试集：覆盖不同维度的能力
    test_cases = [
        {"type": "RAG检索测试", "query": "客户说友商价格更低，我该怎么用我们100Gbps的内网带宽优势回击？"},
        {"type": "MCP数据测试", "query": "帮我查一下 RTX3090 在北京的营收是多少。"},
        {"type": "混合推理测试",
         "query": "一个初创AI客户想微调 Llama-3，我应该给他推荐什么配置？基于推荐的配置，帮我查一下该配置当前在全网的总销量是多少？"}
    ]

    print("🚀 启动 Agent 自动化评测脚手架 (Harness Engineering)...\n")

    for i, test in enumerate(test_cases):
        print(f"[{i + 1}/{len(test_cases)}] 执行用例: {test['type']}")
        print(f"输入: {test['query']}")

        start_time = time.time()
        try:
            result = agent.run(test['query'])
            status = "✅ PASS"
        except Exception as e:
            result = str(e)
            status = "❌ FAIL"
        latency = time.time() - start_time

        print(f"结果: {status} (耗时: {latency:.2f}s)")
        print(f"输出片段: {result[:100]}...\n")
        print("-" * 50)


if __name__ == "__main__":
    run_evaluation()