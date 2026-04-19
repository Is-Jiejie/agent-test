import openai
import httpx
from config.settings import API_KEY, BASE_URL, MODEL_NAME

class MemoryManager:
    def __init__(self, max_turns=5):
        """
        max_turns: 最大保留的近期对话轮数。超过此阈值将触发动态压缩。
        """
        self.max_turns = max_turns
        self.system_prompt = None
        self.history = [] # 存放近期的对话明细
        self.summary = "" # 存放久远历史的浓缩摘要

        #专门用于生成摘要的客户端
        custom_http_client = httpx.Client(trust_env=False)
        self.client = openai.OpenAI(
            api_key=API_KEY, base_url=BASE_URL, http_client=custom_http_client
        )
        self.model = MODEL_NAME

    def set_system_prompt(self, prompt_content):
        self.system_prompt = {"role": "system", "content": prompt_content}

    def add_message(self, message):
        """添加新消息，并检测是否需要压缩"""
        self.history.append(message)
        self._check_and_compress()

    def get_context(self):
        """组装发送给大模型的最终上下文"""
        context = []
        if self.system_prompt:
            context.append(self.system_prompt)
        if self.summary:
            # 将久远的记忆摘要作为前置条件喂给大模型
            context.append({"role": "system", "content": f"【历史记忆摘要】：{self.summary}"})

        context.extend(self.history)
        return context

    def _check_and_compress(self):
        """核心机制：动态滑动窗口"""
        # 简单按消息条数估算轮次 (User + Assistant + Tool 算作几条)
        # 如果历史消息超过了 max_turns * 2，触发压缩
        if len(self.history) > self.max_turns * 2:
            print(f"🗜️ [Memory] 上下文长度 ({len(self.history)}条) 超出阈值，准备动态压缩...")

            # 提取最前面的一半历史记录拿去压缩
            split_idx = len(self.history) // 2

            while split_idx < len(self.history):
                msg = self.history[split_idx]
                # 兼容字典和对象两种格式
                role = msg.get('role') if isinstance(msg, dict) else msg.role
                if role == 'user':
                    break
                split_idx += 1

            # 如果向后找不到 user 消息，说明当前正在密集的工具调用循环中。
            # 强行切断会导致“孤儿 Tool Call”，因此必须暂缓压缩，等本轮任务全跑完再说。
            if split_idx > len(self.history):
                print("⚠️ [Memory] 当前正在处理连续的工具调用链路，为保证上下文完整，暂缓压缩...")
                return


            to_compress = self.history[:split_idx]
            # 剩下的留作近期明细
            self.history = self.history[split_idx:]

            # 后台异步生成摘要（为 Demo 演示，这里采用同步调用）
            self._generate_summary(to_compress)

    def _generate_summary(self, old_messages):
        """调用大模型，将旧对话浓缩为摘要"""
        # 提取文本内容用于生成摘要
        chat_text = ""
        for m in old_messages:
            # 兼容字典格式和对象格式
            role = m.get('role', 'unknown') if isinstance(m, dict) else m.role
            content = m.get('content', '') if isinstance(m, dict) else m.content
            if content:
                chat_text += f"[{role}:{content}]"

        prompt = f"""请将以下历史对话进行极简摘要。保留核心业务信息（如查询过的具体参数、最终得出的关键数据等），过滤掉执行报错、重试等冗余过程。
旧摘要：{self.summary}
新增对话：
{chat_text}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"user", "content":prompt}],
                temperature=0.1
            )
            self.summary = response.choices[0].message.content.strip()
            print(f"🗜️ [Memory] 摘要压缩完成！最新摘要：{self.summary}")
        except Exception as e:
            print(f"⚠️ [Memory] 摘要压缩失败: {e}")