"""
Part 4: 多 Agent 专业分工架构
============================

文件结构（按 Agent 组织）：
-------------------------
- primary_assistant.py  - 主助手（路由和简单查询）
- flight_assistant.py   - 航班助手（改签和取消）
- hotel_assistant.py    - 酒店助手（预订、更新、取消）
- car_rental_assistant.py - 租车助手（预订、更新、取消）
- excursion_assistant.py  - 旅行助手（推荐和预订）
- delegation.py         - 任务委派类和入口节点
- graph.py              - 图构建和组装

核心概念：
---------
1. 任务委派：主助手通过委派工具将任务转交专门助手
2. Human in the Loop：敏感操作前自动中断等待确认
3. 对话状态管理：dialog_state 追踪当前助手
"""

from .graph import create_graph
from .delegation import (
    ToFlightBookingAssistant,
    ToBookCarRental,
    ToHotelBookingAssistant,
    ToBookExcursion,
    create_entry_node,
    pop_dialog_state,
)

__all__ = [
    "create_graph",
    "ToFlightBookingAssistant",
    "ToBookCarRental",
    "ToHotelBookingAssistant",
    "ToBookExcursion",
    "create_entry_node",
    "pop_dialog_state",
]
