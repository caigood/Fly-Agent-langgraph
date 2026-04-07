"""
图模块（Graph Module）
======================

多 Agent 专业分工架构：
- 主助手（Primary Assistant）- 负责路由和简单查询
- 航班助手（Flight Assistant）- 处理航班改签/取消
- 酒店助手（Hotel Assistant）- 处理酒店预订
- 租车助手（Car Rental Assistant）- 处理租车预订
- 旅行助手（Excursion Assistant）- 处理旅行推荐

使用方式：
    from src.graph import create_graph
    graph = create_graph()
"""

from .agents import create_graph

__all__ = ["create_graph"]
