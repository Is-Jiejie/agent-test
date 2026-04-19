import openai
import httpx
import time
import json
import pandas as pd
from pyexpat.errors import messages
from torch.cuda.amp import custom_bwd

from config.settings import API_KEY, BASE_URL, MODEL_NAME

class MonitorAgent:
    def __init__(self):
        custom_http_client = httpx.Client(trust_env=False)
        self.client = openai.OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            http_client=custom_http_client,
        )
        self.model = MODEL_NAME

        # Monitor Agent 不需要长记忆，它每次只做单次判断 (Zero-shot)
        self.system_prompt = """你是一个专业的云服务价格与风控巡检Agent。
你的任务是审查系统最新传入的交易流水，判断是否存在异常。
【异常定义】：
1. 价格倒挂：GPU类实例（带有 'GPU' 字样）单价低于 100 元，或通用实例单价低于 10 元。
2. 刷单嫌疑：单笔订单销售量 (sales_volume) 超过 1000 个。

【输出格式要求】：
如果你发现异常，请严格输出一段 JSON 格式的预警简报，包含：{"status": "DANGER", "reason": "具体的异常原因", "action": "建议采取的阻断措施"}
如果没有发现任何异常，请严格只输出纯文本：NORMAL
"""
    def inspect_data(self, data_batch):
        """核心巡检逻辑：将最新数据批次喂给大模型进行语义与规则判断"""
        # 将数据转换为可读的 JSON 字符串传给模型
        data_str = json.dumps(data_batch, ensure_ascii=False)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"请检查以下最新的一批订单流水：\n{data_str}"}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1 # 风控需要极度严谨，温度降到极低
            )
            result = response.choices[0].message.content.strip()
            return result
        except Exception as e:
            return f"风控系统调用模型失败: {str(e)}"

    def run_daemon(self, mock_data_stream, interval_seconds=3):
        """模拟后台常驻进程 (Daemon)"""
        print("🛡️ [Monitor Agent] 后台风控巡检已启动...")

        for i,batch in enumerate(mock_data_stream):
            print(f"\n⏳ [第 {i + 1} 轮巡检] 正在扫描最新 {len(batch)} 条交易流水...")
            # 记录巡检耗时
            start_time = time.time()
            inspection_result = self.inspect_data(batch)
            cost_time = time.time() - start_time

            if "NORMAL" in inspection_result:
                print(f"✅ [通过] 数据正常。 (耗时: {cost_time:.2f}s)")
            else:
                print(f"🚨 [高危预警拦截] 发现异常交易！(耗时: {cost_time:.2f}s)")
                print(f"📩 [推送结构化预警简报]：\n{inspection_result}")

            # 模拟每隔一段时间扫描一次
            time.sleep(interval_seconds)


# --- 独立测试 Monitor Agent ---
if __name__ == "__main__":
    monitor = MonitorAgent()

    # 我们模拟一个实时数据流 (List of batches)
    # 第一批：正常数据
    batch_1 = [
        {"order_id": "A001", "instance_type": "通用型-8核32G", "sales_volume": 5, "revenue": 300},
        {"order_id": "A002", "instance_type": "vGPU-RTX3090", "sales_volume": 2, "revenue": 500}
    ]

    # 第二批：人为混入一条 "价格倒挂" 的恶意数据（RTX4090 竟然只卖 10 块钱！）
    batch_2 = [
        {"order_id": "A003", "instance_type": "计算型-16核64G", "sales_volume": 10, "revenue": 800},
        {"order_id": "A004", "instance_type": "vGPU-RTX4090", "sales_volume": 3, "revenue": 10}  # 异常！单价仅3.3元
    ]

    # 第三批：人为混入一条 "刷单嫌疑" 的恶意数据
    batch_3 = [
        {"order_id": "A005", "instance_type": "通用型-8核32G", "sales_volume": 5000, "revenue": 150000}  # 异常！单笔超1000台
    ]

    data_stream = [batch_1, batch_2, batch_3]

    # 启动后台巡检，模拟每 3 秒检查一次
    monitor.run_daemon(data_stream, interval_seconds=3)


