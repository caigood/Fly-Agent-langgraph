# Fly-Agent-langgraph

基于 LangGraph 的多智能体航空客服系统，实现了专业化的任务分工和人工确认机制。

## 功能特性

- **多智能体架构**：主助手 + 4 个专门助手（航班、酒店、租车、旅行）
- **任务委派机制**：主助手自动识别用户意图并委派给专门助手
- **RAG 检索增强**：基于向量相似度的政策文档检索，支持语义化查询
- **数据库查询**：SQLite 数据库存储航班、酒店、租车等业务数据，支持结构化查询
- **Human-in-the-Loop**：敏感操作（预订、取消等）执行前自动中断，等待用户确认
- **对话状态管理**：追踪当前对话位置，支持助手间切换

## 系统架构

```
START → fetch_user_info → primary_assistant
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
enter_update_flight       enter_book_hotel       enter_book_car_rental
        │                         │                         │
        ▼                         ▼                         ▼
update_flight            book_hotel             book_car_rental
        │                         │                         │
        ▼                         ▼                         ▼
(safe/sensitive)       (safe/sensitive)        (safe/sensitive)
        │                         │                         │
        └─────────────────────────┴─────────────────────────┘
                                  │
                                  ▼
                          leave_skill (返回主助手)
```

## 技术栈

- **LangGraph**: 多智能体工作流编排
- **LangChain**: LLM 应用框架
- **SQLite**: 数据存储
- **Pydantic**: 数据验证

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据库

项目需要数据库文件才能运行：

```bash
# 解压数据库文件
unzip travel2.sqlite.zip

# 或者从原始来源下载
# 数据来源：https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env` 并填写配置：

```bash
cp .env.example .env
```

### 4. 运行程序

```bash
python main.py
```

## 项目结构

```
Fly-Agent-langgraph/
├── main.py                 # 程序入口
├── requirements.txt        # 依赖列表
├── .env.example           # 环境变量示例
├── travel2.sqlite.zip     # 数据库文件（压缩）
└── src/
    ├── agents/            # Agent 定义
    │   ├── assistant.py   # Agent 基类
    │   └── state.py       # 状态定义
    ├── graph/             # 图构建
    │   └── agents/        # 各助手实现
    │       ├── graph.py   # 主图构建
    │       ├── primary_assistant.py
    │       ├── flight_assistant.py
    │       ├── hotel_assistant.py
    │       ├── car_rental_assistant.py
    │       ├── excursion_assistant.py
    │       └── delegation.py
    ├── tools/             # 工具定义
    └── utils/             # 工具函数
```

## 核心概念

### 1. RAG 检索增强生成

系统使用 RAG 技术实现政策文档的语义检索：

- **向量化存储**：使用 OpenAI Embedding 模型将政策文档转换为向量
- **相似度搜索**：基于余弦相似度检索最相关的文档片段
- **应用场景**：查询退改签政策、行李规定、登机流程等

示例：
```
用户：我可以免费改签吗？
系统：通过 RAG 检索相关政策文档 → 返回准确的改签规定
```

### 2. 数据库查询

系统使用 SQLite 存储和管理业务数据：

- **航班数据**：航班信息、时刻表、座位情况
- **用户数据**：机票预订、座位分配
- **酒店数据**：酒店信息、房间预订
- **租车数据**：租车公司、车辆预订

查询方式：
- 结构化 SQL 查询（精确匹配）
- 支持多条件筛选（时间、地点等）

### 3. 任务委派

主助手通过调用委派工具（如 `ToFlightBookingAssistant`）将任务转交给专门助手。

### 4. 入口节点

每个专门助手都有一个入口节点，负责：
- 发送切换消息
- 更新对话状态

### 5. 路由函数

决定下一步执行哪个节点：
- 检测是否需要返回主助手
- 区分安全工具和敏感工具

### 6. Human-in-the-Loop

敏感操作执行前自动中断，等待用户确认：
- 用户确认：继续执行
- 用户拒绝：告知助手调整策略

## 许可证

MIT License
