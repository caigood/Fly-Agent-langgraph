"""
图结构概览：
================

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
                                  │
                                  ▼
                          primary_assistant

Human in the Loop：
===================
敏感操作（预订、取消等）执行前自动中断，等待用户确认。
"""

import os

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from src.agents.assistant import Assistant
from src.agents.state import State
from src.tools import fetch_user_flight_information
from src.utils.helpers import create_tool_node_with_fallback

from .delegation import create_entry_node, pop_dialog_state
from .primary_assistant import create_runnable as create_primary_runnable
from .primary_assistant import route as route_primary
from .primary_assistant import TOOLS as PRIMARY_TOOLS
from .flight_assistant import create_runnable as create_flight_runnable
from .flight_assistant import route as route_flight
from .flight_assistant import SAFE_TOOLS as FLIGHT_SAFE_TOOLS
from .flight_assistant import SENSITIVE_TOOLS as FLIGHT_SENSITIVE_TOOLS
from .hotel_assistant import create_runnable as create_hotel_runnable
from .hotel_assistant import route as route_hotel
from .hotel_assistant import SAFE_TOOLS as HOTEL_SAFE_TOOLS
from .hotel_assistant import SENSITIVE_TOOLS as HOTEL_SENSITIVE_TOOLS
from .car_rental_assistant import create_runnable as create_car_rental_runnable
from .car_rental_assistant import route as route_car_rental
from .car_rental_assistant import SAFE_TOOLS as CAR_RENTAL_SAFE_TOOLS
from .car_rental_assistant import SENSITIVE_TOOLS as CAR_RENTAL_SENSITIVE_TOOLS
from .excursion_assistant import create_runnable as create_excursion_runnable
from .excursion_assistant import route as route_excursion
from .excursion_assistant import SAFE_TOOLS as EXCURSION_SAFE_TOOLS
from .excursion_assistant import SENSITIVE_TOOLS as EXCURSION_SENSITIVE_TOOLS


def create_graph():
    """
    图结构包含：
    1. 入口节点：获取用户信息
    2. 主助手：路由到各个专门助手
    3. 四个专门助手：航班、酒店、租车、旅行
    4. 返回机制：专门助手完成后返回主助手
    5. 中断点：敏感操作前等待用户确认
    
    返回：
        编译好的 LangGraph 图对象
    """
    
    # ==================== 初始化 LLM 和各助手 ====================
    
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "qwen-plus"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # 创建各助手的可运行对象（包含提示词 + LLM + 工具绑定）
    primary_runnable = create_primary_runnable(llm)         # 主助手
    flight_runnable = create_flight_runnable(llm)           # 航班助手
    hotel_runnable = create_hotel_runnable(llm)             # 酒店助手
    car_rental_runnable = create_car_rental_runnable(llm)   # 租车助手
    excursion_runnable = create_excursion_runnable(llm)     # 旅行助手

    # 创建图构建器
    builder = StateGraph(State)

    # ==================== 第1步：入口节点 ====================
    # 获取用户航班信息，作为对话上下文
    
    def user_info(state: State):
        """获取用户信息节点"""
        # 从工具中获取用户航班信息  执行工具函数 （查数据库）
        return {"user_info": fetch_user_flight_information.invoke({})}

    builder.add_node("fetch_user_info", user_info)
    builder.add_edge(START, "fetch_user_info")

    # ==================== 第2步：主助手 ====================
    # 核心调度中心，负责识别意图并委派给专门助手
    # Assistant 给 primary_runnable 的链加了 包装 ，处理边界情况！
    builder.add_node("primary_assistant", Assistant(primary_runnable))
    # 创建一个"工具执行器"：
    # - 正常：执行工具（如搜索航班）
    # - 异常：捕获错误，返回友好提示（不崩溃）
    builder.add_node("primary_assistant_tools", create_tool_node_with_fallback(PRIMARY_TOOLS))
    
    # 主助手的路由：可以委派给4个专门助手，或执行普通工具
    builder.add_conditional_edges(
        "primary_assistant",
        # 主助手的路由函数，根据 LLM 输出判断下一步，从文件最上面import引入并改了别名
        route_primary,
        # 路由表
        [
            "enter_update_flight",      # 委派给航班助手
            "enter_book_car_rental",    # 委派给租车助手
            "enter_book_hotel",         # 委派给酒店助手
            "enter_book_excursion",     # 委派给旅行助手
            "primary_assistant_tools",  # 执行普通工具（搜索等）
            END,                        # 结束对话
        ],
    )
    builder.add_edge("fetch_user_info", "primary_assistant")
    builder.add_edge("primary_assistant_tools", "primary_assistant")

    # ==================== 第3步：航班助手子图 ====================
    # 处理航班查询、改签、取消
    
    # 3.1 入口节点：切换到航班助手
    # enter_update_flight（入口）
    # ├── 发送系统消息（告诉专门助手"你来了"）
    # ├── 更新对话状态（记录当前在哪个助手）
    # └── 清理上下文（可选）

    # update_flight（助手）
    # ├── 接收用户消息
    # ├── LLM 思考
    # └── 决定调用工具或返回
    builder.add_node(
        "enter_update_flight",
        create_entry_node("航班更新助手", "update_flight"),
    )
    builder.add_edge("enter_update_flight", "update_flight")
    
    # 3.2 航班助手主体
    builder.add_node("update_flight", Assistant(flight_runnable))
    
    # 3.3 航班助手的工具节点
    builder.add_node(
        "update_flight_safe_tools",           # 安全工具：查询类
        create_tool_node_with_fallback(FLIGHT_SAFE_TOOLS),
    )
    builder.add_node(
        "update_flight_sensitive_tools",      # 敏感工具：修改类（需确认）
        create_tool_node_with_fallback(FLIGHT_SENSITIVE_TOOLS),
    )
    
    # 3.4 航班助手的边
    builder.add_edge("update_flight_safe_tools", "update_flight")      # 安全工具执行完返回助手
    builder.add_edge("update_flight_sensitive_tools", "update_flight") # 敏感工具执行完返回助手
    
    # 3.5 航班助手的路由
    builder.add_conditional_edges(
        "update_flight",
        route_flight,
        [
            "update_flight_safe_tools",      # 执行安全工具
            "update_flight_sensitive_tools", # 执行敏感工具（会触发中断）
            "leave_skill",                   # 返回主助手
            END,                             # 结束对话
        ],
    )

    # ==================== 第4步：租车助手子图 ====================
    # 处理租车查询、预订
    
    # 4.1 入口节点
    builder.add_node(
        "enter_book_car_rental",
        create_entry_node("租车助手", "book_car_rental"),
    )
    builder.add_edge("enter_book_car_rental", "book_car_rental")
    
    # 4.2 租车助手主体
    builder.add_node("book_car_rental", Assistant(car_rental_runnable))
    
    # 4.3 租车助手的工具节点
    builder.add_node(
        "book_car_rental_safe_tools",
        create_tool_node_with_fallback(CAR_RENTAL_SAFE_TOOLS),
    )
    builder.add_node(
        "book_car_rental_sensitive_tools",
        create_tool_node_with_fallback(CAR_RENTAL_SENSITIVE_TOOLS),
    )
    
    # 4.4 租车助手的边
    builder.add_edge("book_car_rental_safe_tools", "book_car_rental")
    builder.add_edge("book_car_rental_sensitive_tools", "book_car_rental")
    
    # 4.5 租车助手的路由
    builder.add_conditional_edges(
        "book_car_rental",
        route_car_rental,
        [
            "book_car_rental_safe_tools",
            "book_car_rental_sensitive_tools",
            "leave_skill",
            END,
        ],
    )

    # ==================== 第5步：酒店助手子图 ====================
    # 处理酒店查询、预订
    
    # 5.1 入口节点
    builder.add_node(
        "enter_book_hotel",
        create_entry_node("酒店预订助手", "book_hotel"),
    )
    builder.add_edge("enter_book_hotel", "book_hotel")
    
    # 5.2 酒店助手主体
    builder.add_node("book_hotel", Assistant(hotel_runnable))
    
    # 5.3 酒店助手的工具节点
    builder.add_node(
        "book_hotel_safe_tools",
        create_tool_node_with_fallback(HOTEL_SAFE_TOOLS),
    )
    builder.add_node(
        "book_hotel_sensitive_tools",
        create_tool_node_with_fallback(HOTEL_SENSITIVE_TOOLS),
    )
    
    # 5.4 酒店助手的边
    builder.add_edge("book_hotel_safe_tools", "book_hotel")
    builder.add_edge("book_hotel_sensitive_tools", "book_hotel")
    
    # 5.5 酒店助手的路由
    builder.add_conditional_edges(
        "book_hotel",
        route_hotel,
        [
            "leave_skill",
            "book_hotel_safe_tools",
            "book_hotel_sensitive_tools",
            END,
        ],
    )

    # ==================== 第6步：旅行助手子图 ====================
    # 处理旅行推荐、预订
    
    # 6.1 入口节点
    builder.add_node(
        "enter_book_excursion",
        create_entry_node("旅行推荐助手", "book_excursion"),
    )
    builder.add_edge("enter_book_excursion", "book_excursion")
    
    # 6.2 旅行助手主体
    builder.add_node("book_excursion", Assistant(excursion_runnable))
    
    # 6.3 旅行助手的工具节点
    builder.add_node(
        "book_excursion_safe_tools",
        create_tool_node_with_fallback(EXCURSION_SAFE_TOOLS),
    )
    builder.add_node(
        "book_excursion_sensitive_tools",
        create_tool_node_with_fallback(EXCURSION_SENSITIVE_TOOLS),
    )
    
    # 6.4 旅行助手的边
    builder.add_edge("book_excursion_safe_tools", "book_excursion")
    builder.add_edge("book_excursion_sensitive_tools", "book_excursion")
    
    # 6.5 旅行助手的路由
    builder.add_conditional_edges(
        "book_excursion",
        route_excursion,
        [
            "leave_skill",
            "book_excursion_safe_tools",
            "book_excursion_sensitive_tools",
            END,
        ],
    )

    # ==================== 第7步：返回机制 ====================
    # 专门助手完成后，通过 leave_skill 返回主助手
    
    builder.add_node("leave_skill", pop_dialog_state)
    builder.add_edge("leave_skill", "primary_assistant")

    # ==================== 第8步：编译图 ====================
    # 添加中断点：敏感操作前等待用户确认
    
    memory = InMemorySaver()
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=[
            # 在这些节点执行前中断，等待用户确认
            "update_flight_sensitive_tools",     # 航班改签/取消前
            "book_car_rental_sensitive_tools",   # 租车预订前
            "book_hotel_sensitive_tools",        # 酒店预订前
            "book_excursion_sensitive_tools",    # 旅行预订前
        ],
    )
    return graph
