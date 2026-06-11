# 🤖 智能测试协同平台 (Multi-Agent Test Copilot)

> **一站式 AI 驱动的自动化测试用例生成系统**  
> 输入自然语言需求 → 拉取 API 契约 → 沙盒探路验证 → 生成全场景 YAML DSL 测试资产

---

## 📋 目录

- [1. 项目概述](#1-项目概述)
- [2. 核心能力](#2-核心能力)
- [3. 系统架构](#3-系统架构)
- [4. 项目目录结构](#4-项目目录结构)
- [5. 工作流程](#5-工作流程)
- [6. 快速开始](#6-快速开始)
- [7. 使用示例](#7-使用示例)
- [8. 生成的测试资产](#8-生成的测试资产)
- [9. 安全机制](#9-安全机制)
- [10. 配置说明](#10-配置说明)
- [11. 改动日志](#11-改动日志)

---

## 1. 项目概述

**智能测试协同平台**是一个基于 **多智能体（Multi-Agent）架构** 的自动化测试用例生成系统。它能够：

- 接收研发/测试人员用 **自然语言** 描述的测试需求
- 自动识别意图、检索历史经验、拉取 API 接口文档
- 在本地 **沙盒环境** 中探路验证接口的真实响应
- 按照 **私有测试框架 DSL 规范** 生成覆盖 **正向流程、必填缺失、边界值、异常输入** 四大维度的测试用例
- 将生成的测试资产固化到本地文件系统和 SQLite 经验库中

### 核心理念

| 理念 | 说明 |
|:---|:---|
| **单点探路，全面铺开** | 沙盒阶段只验证 1 条 Happy Path，生成阶段覆盖全部 4 个维度 |
| **意图分类守卫** | 闲聊/能力咨询类输入不进入核心流水线，节省 Token |
| **经验持续累积** | 成功的测试模板自动固化为可复用资产，越用越聪明 |

---

## 2. 核心能力

### 2.1 支持的测试类型

| 类型 | 说明 | 示例 |
|:---|:---|:---|
| **接口/API 测试** | 单接口功能验证 | "帮我写一个登录接口的测试用例" |
| **全链路/E2E 测试** | 多接口串联流程 | "帮我写一个购买物品的全链路测试用例" |
| **参数校验测试** | 必填字段、类型校验 | "测一下 /api/user/info 的异常参数" |

### 2.2 测试覆盖维度

每个 API 步骤生成的测试用例覆盖 **4 大维度**：

```
✅ [正向]   : 合法参数，预期 msg_code == 200
🚫 [必填缺失]: 必填字段置空/不传，预期返回参数校验错误
⚠️ [边界值] : 0/负数/超大值/超长字符串/SQL注入payload
💥 [异常输入]: 类型错误/不存在资源/鉴权缺陷
```

### 2.3 意图分类守卫

系统通过轻量级分类器对用户输入进行三分类：

| 分类 | 处理方式 | Token 消耗 |
|:---|:---|:---|
| `TEST_REQUEST` | → 进入完整流水线 | 正常 |
| `CAPABILITY_INQUIRY` | → 展示能力边界引导面板 | 极低（classify 仅 ~16 token） |
| `CHITCHAT` | → 礼貌重定向提示 | 极低（classify 仅 ~16 token） |

---

## 3. 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                        main.py (入口)                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              🔰 意图分类守卫 (Gate)                      ││
│  │          classify() → TEST_REQUEST / CAP / CHITCHAT      ││
│  └──────────────┬──────────────────────────────────────────┘│
│                 │ TEST_REQUEST                               │
│  ┌──────────────▼──────────────────────────────────────────┐│
│  │              第一阶段：PlannerAgent                       ││
│  │  意图识别 → 经验库检索 → USE_TEMPLATE / EXPLORE 策略      ││
│  └──────────────┬──────────────────────────────────────────┘│
│                 │                                            │
│  ┌──────────────▼──────────────────────────────────────────┐│
│  │              第二阶段：CoderAgent (ReAct 循环)            ││
│  │  ┌──────────────────────────────────────────────────┐   ││
│  │  │ 阶段一：Python 沙盒探路                           │   ││
│  │  │  get_api_docs → lint → execute_pytest → 看响应    │   ││
│  │  ├──────────────────────────────────────────────────┤   ││
│  │  │ 阶段二：DSL 翻译与固化                            │   ││
│  │  │  全场景 YAML 生成 → save_test_case → 报告输出     │   ││
│  │  └──────────────────────────────────────────────────┘   ││
│  └──────────────┬──────────────────────────────────────────┘│
│                 │                                            │
│  ┌──────────────▼──────────────────────────────────────────┐│
│  │           第三阶段：资产固化 & 报告                       ││
│  │  ExperienceDB.save_template() → 本地 YAML 文件 + SQLite  ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

### 多智能体角色

| Agent | 文件 | 职责 |
|:---|:---|:---|
| **PlannerAgent** | `agent/planner_agent.py` | 意图分类、经验检索、策略制定 |
| **CoderAgent** | `agent/coder_agent.py` | 文档拉取、沙盒探路、DSL 生成与固化 |
| **ReportAgent** | `agent/report_agent.py` | 测试结果爬取、Markdown 报告生成 |
| **MemoryManager** | `agent/memory.py` | 对话历史管理、长期经验沉淀 |
| **MonitorAgent** | `agent/monitor_agent.py` | 后台巡检守护（云价格/风控） |

### 沙盒安全三层漏斗

| 层 | 文件 | 检查内容 |
|:---|:---|:---|
| 🥇 漏斗一 | `sandbox/ast_linter.py` | Markdown 清洗 + AST 语法校验 |
| 🥈 漏斗二 | `sandbox/ast_scanner.py` | AST 遍历，拦截 os/sys/subprocess、open/exec/eval/compile |
| 🥉 漏斗三 | `sandbox/llm_reviewer.py` | LLM 语义审查：死循环、破坏性调用、SQL 注入、批量并发 |

---

## 4. 项目目录结构

```
AI_Test_Agent/
├── main.py                    # 🚪 系统入口，对话主循环 + 意图路由
├── pytest_executor.py         # 🏃 Pytest 子进程执行器
├── hello.py                   # 🔧 环境冒烟测试脚本
├── test_experience.db         # 🗄️ SQLite 经验库
│
├── agent/                     # 🧠 多智能体核心
│   ├── planner_agent.py       #   Planner：意图分类 + 经验检索
│   ├── coder_agent.py         #   Coder：沙盒探路 + DSL 生成
│   ├── report_agent.py        #   Report：Markdown 报告生成
│   ├── memory.py              #   Memory：上下文管理 + 经验沉淀
│   ├── react_agent.py         #   React：单 Agent ReAct 循环（旧版）
│   └── monitor_agent.py       #   Monitor：后台云价格/风控巡检
│
├── config/                    # ⚙️ 配置中心
│   ├── settings.py            #   API_KEY, BASE_URL, MODEL_NAME 等
│   └── prompts.py             #   CoderAgent 系统提示词模板
│
├── sandbox/                   # 🔒 三层安全漏斗
│   ├── ast_linter.py          #   漏斗一：语法检查 + Markdown 清洗
│   ├── ast_scanner.py         #   漏斗二：AST 危险调用拦截
│   ├── llm_reviewer.py        #   漏斗三：LLM 语义安全审查
│   └── executor.py            #   编排器：串联三个漏斗
│
├── tools/                     # 🔧 工具层
│   ├── mcp_server.py          #   Mock MCP Server（6 端电商 API 文档）
│   └── db_manager.py          #   ExperienceDB（SQLite 模板存取）
│
├── tests/auto_generated/      # 📦 生成的测试资产
│   ├── login.yaml             #   登录 - 7 个场景
│   ├── purchase_full_chain.yaml     #   购买全链路 - 5 步骤
│   └── purchase_full_chain_test.yaml # 购买全链路 - 32 个场景 ⭐
│
├── reports/                   # 📊 测试执行报告
│   └── report_*.md
│
└── test/                      # 🧪 项目自身测试
    └── eval_agent.py          #   Agent 评估工具
```

---

## 5. 工作流程

### 完整流水线

```
用户输入: "帮我写一个购买物品的全链路测试用例"
   │
   ├─ [1] 🔰 意图分类守卫
   │      classify() → TEST_REQUEST
   │
   ├─ [2] 🧠 Planner: 意图识别 + 经验检索
   │      提取意图 "购买物品" → 查询 SQLite 经验库
   │      → 无匹配模板 → strategy = EXPLORE
   │
   ├─ [3] 📖 Coder 阶段一：拉取 API 文档
   │      get_api_documentation_tool("getProductList,commitOrder,doPay")
   │
   ├─ [4] 🔬 Coder 阶段一：沙盒探路
   │      lint_code_tool(code) → ✅
   │      execute_local_pytest_tool(code) → 看到真实 JSON 响应结构
   │
   ├─ [5] 📝 Coder 阶段二：全场景 DSL 生成
   │      为每个 API 步骤生成 4 个维度的测试用例：
   │        [正向]   getProductList → msg_code 200
   │        [边界值]  getProductList page=0, page=-1
   │        [正向]   commitOrder → msg_code 200
   │        [必填缺失] commitOrder address=""
   │        [边界值]  commitOrder quantity=0, quantity=99999
   │        [异常输入] commitOrder product_id=不存在
   │        ...
   │      save_test_case_tool("purchase_full_chain.yaml", yaml) → 固化
   │
   └─ [6] 💾 经验固化
          保存 YAML 到 tests/auto_generated/
          同步写入 SQLite 经验库
          输出 Markdown 执行大盘报告
```

### 沙盒探路的"故意报错"机制

```python
# 沙盒会屏蔽 print()，用这个方法看响应：
response = requests.post(url, json=data, headers=headers)
assert False, f"我要看响应: {response.text}"
# ☝ 这行断言必败，但报错信息会带着 JSON 响应一起吐出来
# 看完响应结构后，把断言改成 assert response.json()['msg_code'] == 200
```

---

## 6. 快速开始

### 环境要求

- Python 3.10+
- 可用的 LLM API（兼容 OpenAI 接口）
- 本地 Mock API 服务（默认 `http://127.0.0.1:8787`）
- `uv` 包管理器（用于 `uv run pytest`）

### 安装步骤

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd AI_Test_Agent

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install langchain langchain-openai langchain-classic openai httpx pydantic

# 4. 配置 API 密钥
# 编辑 config/settings.py，填入你的 API_KEY 和 BASE_URL

# 5. 启动系统
python main.py
```

### 配置 LLM 后端

编辑 [config/settings.py](config/settings.py)：

```python
API_KEY = "your-api-key-here"          # 你的 API Key
BASE_URL = "https://your-llm-api.com/v1/"  # API 地址
MODEL_NAME = "DeepSeek-V4-Flash"       # 使用的模型名
MAX_RETRIES = 6                         # 最大重试次数
```

---

## 7. 使用示例

### 基本对话

```
🚀 智能测试协同平台 (Multi-Agent Copilot) 已启动！
💡 提示：您可以连续输入测试需求。
   [示例 1] 帮我写一个登录接口的自动化测试用例
   [示例 2] 帮我写一个购买物品的全链路测试用例
🚪 输入 'exit', 'quit' 或 'q' 退出系统。
============================================================
⚙️ 正在初始化 Planner 与 Coder 引擎...
✅ 引擎初始化完成，随时待命！

👨‍💻 研发/测试人员: 帮我写一个登录接口的自动化测试用例
🔍 [Gate] 正在进行意图分类...
🏷️  [Gate] 分类结果: TEST_REQUEST
🧠 [Planner] 正在进行意图识别与历史经验检索...
📋 [Planner] 识别意图: 用户登录 | 执行策略: EXPLORE
💻 [Coder] 正在接管任务，准备拉取契约并进行沙盒验证...
... (沙盒探路 + DSL 生成) ...

==================== 测试执行报告 ====================
# 📊 智能 Agent 自动化测试全链路执行大盘
| 🎯 任务/文件模块 | 总用例数 | 🟩 通过数 | 🟥 失败数 | ⚡ 运行状态 |
| :---: | :---: | :---: | :---: | :---: |
| login.yaml | 7 | 7 | 0 | ✅ SUCCESS |
======================================================

👨‍💻 研发/测试人员: 今天天气怎么样
🔍 [Gate] 正在进行意图分类...
🏷️  [Gate] 分类结果: CHITCHAT
⚠️ 检测到您的输入与【自动化测试用例生成】无关。
👋 我是专门用于生成自动化测试用例的智能体...
🔄 系统已就绪，请重新输入测试需求。

👨‍💻 研发/测试人员: 你能做什么
🔍 [Gate] 正在进行意图分类...
🏷️  [Gate] 分类结果: CAPABILITY_INQUIRY
╔══════════════════════════════════════════════════╗
║      🧪 智能测试协同平台 — 能力边界说明          ║
║  ✅ 我能做什么：接口/API 测试、全链路E2E测试...  ║
║  📝 示例输入...                                  ║
╚══════════════════════════════════════════════════╝
```

### 更多示例输入

| 输入 | 预期行为 |
|:---|:---|
| `帮我写一个登录接口的自动化测试用例` | 生成单接口多场景 YAML |
| `帮我写一个购买物品的全链路测试用例` | 生成商品→下单→支付全链路 YAML |
| `生成退款流程的端到端测试` | 生成退款全链路 YAML |
| `帮我测一下 /api/user/info 接口的异常参数` | 针对单个接口生成异常参数测试 |
| `你能做什么` | 展示能力边界引导面板（不进入流水线） |
| `今天天气怎么样` | 礼貌重定向提示（不消耗额外 Token） |

---

## 8. 生成的测试资产

### YAML DSL 格式

生成的 YAML 文件严格遵循私有测试框架规范：

```yaml
baseInfo:
  api_name: "商品浏览到提交订单全链路测试"
  header:
    Content-Type: "application/json"

testCase:
  # ── 步骤1：获取商品列表 ──
  - case_name: "[正向] 查询商品列表"
    url: "/dar/product/list"
    method: POST
    json:
      page: 1
      page_size: 10
    validation:
      - eq: { "msg_code": 200 }
    extract:
      first_product_id: "$.data.products[0].product_id"

  - case_name: "[边界值] 分页参数 page=0"
    url: "/dar/product/list"
    method: POST
    json:
      page: 0
      page_size: 10
    validation:
      - eq: { "msg_code": -1 }

  - case_name: "[边界值] 分页参数 page=-1"
    url: "/dar/product/list"
    method: POST
    json:
      page: -1
      page_size: 10
    validation:
      - eq: { "msg_code": -1 }

  # ── 步骤2：提交订单 ──
  - case_name: "[正向] 提交订单"
    url: "/dar/order/commit"
    method: POST
    json:
      product_id: "${get_extract_data(first_product_id)}"
      quantity: 1
      address: "北京市朝阳区测试地址100号"
    validation:
      - eq: { "msg_code": 200 }

  - case_name: "[必填缺失] 提交订单-缺少address"
    url: "/dar/order/commit"
    method: POST
    json:
      product_id: "${get_extract_data(first_product_id)}"
      quantity: 1
    validation:
      - eq: { "msg_code": -1 }

  - case_name: "[边界值] 提交订单-quantity为0"
    url: "/dar/order/commit"
    method: POST
    json:
      product_id: "${get_extract_data(first_product_id)}"
      quantity: 0
      address: "北京市朝阳区测试地址100号"
    validation:
      - eq: { "msg_code": -1 }
```

### 跨接口数据传递

```yaml
# 步骤 A：提取数据
  - case_name: "[正向] 查询商品列表"
    extract:
      first_product_id: "$.data.products[0].product_id"   # JsonPath 提取

# 步骤 B：引用数据
  - case_name: "[正向] 提交订单"
    json:
      product_id: "${get_extract_data(first_product_id)}"  # 闭包函数引用
```

### 已生成的测试资产统计

| 文件名 | 用例数 | 覆盖类型 |
|:---|:---:|:---|
| `purchase_full_chain_test.yaml` | 32 | 全链路 4 维度 |
| `login.yaml` | 7 | 登录 4 维度 |
| `purchase_full_chain.yaml` | 5 | 5 步骤全链路 |
| `product_to_order_full_chain.yaml` | 3 | 3 步骤全链路 |
| `purchase_full_link.yaml` | 2 | 双步骤链路 |
| `test_login_api.yaml` | 2 | 登录正向+异常 |
| *其他* | 3 | 单场景 |

---

## 9. 安全机制

### 三层漏斗防御

所有 LLM 生成的 Python 代码在沙盒执行前，必须通过三层安全检查：

```
LLM 生成的代码
   │
   ├─ 🥇 漏斗一：AST Linter (sandbox/ast_linter.py)
   │     · 清洗 Markdown 代码围栏 (```python / ```py / ```)
   │     · AST 语法合法性校验
   │     └─ 失败 → ❌ 拒绝执行
   │
   ├─ 🥈 漏斗二：AST Scanner (sandbox/ast_scanner.py)
   │     · 遍历 AST 节点树
   │     · 拦截危险模块：os, sys, subprocess, shutil, socket, ctypes
   │     · 拦截危险函数：open(), exec(), eval(), compile(), __import__()
   │     └─ 命中 → ❌ 抛出 SecurityException
   │
   └─ 🥉 漏斗三：LLM Reviewer (sandbox/llm_reviewer.py)
         · 语义级安全审查（6 类高危行为）
         · 结构化输出 → 判定 safe / unsafe + 理由
         └─ unsafe → ❌ 拒绝执行
```

### Agent 执行层防护

| 防护项 | 配置 | 位置 |
|:---|:---|:---|
| 最大迭代次数 | 25 轮 | `coder_agent.py` |
| 硬超时 | 300 秒 (5 分钟) | `coder_agent.py` |
| 早期停止策略 | `generate`（触顶时生成最终输出） | `coder_agent.py` |
| 沙盒探路迭代预算 | 最多 6 轮 | `prompts.py` |
| Tool return_direct | `save_test_case_tool` 调用后立即终止 | `coder_agent.py` |

---

## 10. 配置说明

### config/settings.py

```python
API_KEY = "sk-xxx"                              # LLM API 密钥
BASE_URL = "https://llmapi.paratera.com/v1/"    # LLM API 地址
MODEL_NAME = "DeepSeek-V4-Flash"                # 使用的模型
MAX_RETRIES = 6                                  # 最大重试次数
REMOTE_PROJECT_ROOT = "/root/root/Test-Automation-Framework"  # 远端测试框架路径
REMOTE_TEST_DIR = "..."                          # 远端测试用例存放路径
```

### config/prompts.py

包含 CoderAgent 的系统提示词模板，定义了：
- **核心一**：探路态与交付态的时空隔离（沙盒含 login，YAML 不含 login）
- **核心二**：沙盒破壁与降级策略（故意报错看响应 / 404 立即降级 / 迭代预算约束）
- **核心三**：全场景测试用例覆盖（4 维度 + 硬性指标）
- **双轨工作流**：阶段一（Python 探路）→ 阶段二（DSL 翻译）
- **DSL 规范**：baseInfo + testCase 结构、JsonPath 提取、闭包函数引用

---

## 11. 改动日志

### v1.0 — 初始版本
- 多智能体架构（Planner + Coder + Report）
- ReAct 循环 + 工具调用
- 双轨工作流：Python 沙盒探路 → YAML DSL 生成
- SQLite 经验库模板复用
- 三层安全漏斗防御

### v1.1 — Robust 意图分类守卫
- 新增 `PlannerAgent.classify()` 轻量级分类器（max_tokens=16）
- 三种分类路由：`TEST_REQUEST` / `CAPABILITY_INQUIRY` / `CHITCHAT`
- 非测试请求不进入 CoderAgent 流水线，节省 Token
- 能力咨询 → 展示引导面板；闲聊 → 礼貌重定向

### v1.2 — 全场景测试覆盖
- 新增**核心三**：4 维度测试覆盖规范
  - ✅ 正向流程 | 🚫 必填缺失 | ⚠️ 边界值 | 💥 异常输入
- 硬性指标：每 API 步骤 ≥ 1 正向 + 2 异常/边界，全链路 ≥ 5 条
- case_name 引入场景前缀：`[正向]` / `[必填缺失]` / `[边界值]` / `[异常输入]`

### v1.3 — 迭代健壮性增强
- `max_iterations` 从 15 提升至 25
- `early_stopping_method` 从 `"force"` 改为 `"generate"`
- 新增 `max_execution_time = 300s` 硬超时
- 沙盒探路迭代预算约束：最多 6 轮，连续失败 3 次即逃逸
- 输出兜底：检测 "Agent stopped" 残留文本，提取中间步骤摘要

---

## 📄 License

本项目仅供学习和内部使用。

---

*本报告由 AI 辅助生成 · 最后更新：2026-06-11*
