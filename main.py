# main.py
# from agent.react_agent import QueryAgent
from agent.langchain_agent import LangChainSalesAgent

if __name__ == "__main__":
    # agent = QueryAgent()
    print("🚀 正在启动 智能测开(QA)助理引擎...")
    print("加载底层 MCP 工具与 AST 沙盒安全策略...")
    agent = LangChainSalesAgent(max_context_turns=3)

    print("\n" + "=" * 50)
    print("✅ 智能测试助理已就绪！")
    print("💡 提示：输入 '帮我写一个登录接口的自动化测试用例' 开始体验。")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("👨‍💻 研发/测试人员: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue

            print(f"🤖 Agent 查阅文档与思考中，请稍候...\n")
            result = agent.run(user_query=user_input)

            print("\n" + "-" * 50)
            print("✅ 【测试执行报告】:\n")
            print(result)
            print("-" * 50 + "\n")

        except Exception as e:
            print(f"\n❌ 系统异常: {str(e)}\n")

    # 测试用例：让它展示被 AST 拦截后的自我修复能力
    # test_query = "帮我查询一下 'vGPU-RTX4090' 的总营收和销量，并用一段 Python 代码帮我计算出它的平均客单价打印出来。"

    # print("========== 第一轮对话 ==========")
    # agent.run("帮我查询 'vGPU-RTX4090' 的总营收。")
    #
    # print("\n========== 第二轮对话 ==========")
    # agent.run("那 'vGPU-RTX3090' 呢？")
    #
    # print("\n========== 第三轮对话 (应该触发记忆压缩) ==========")
    # # 这一步测试 Agent 是否还“记得”前两轮查过的数据，并且由于超长触发了摘要
    # agent.run("请把这两种显卡的营收加起来，告诉我总和是多少？不需要重新调用工具了，直接根据你刚才查到的结果写代码计算。")