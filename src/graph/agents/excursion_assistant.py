"""
旅行助手（Excursion Assistant）
===============================

专门处理旅行推荐和游览预订。

职责：
1. 搜索旅行推荐
2. 预订游览项目
3. 更新游览预订
4. 取消游览预订

工具分类：
- 安全工具：search_trip_recommendations（只读查询）
- 敏感工具：book_excursion, update_excursion, cancel_excursion（需要用户确认）
"""

from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from langgraph.prebuilt import tools_condition

from src.tools import (
    search_trip_recommendations,
    book_excursion,
    update_excursion,
    cancel_excursion,
)
from src.agents.assistant import CompleteOrEscalate


SYSTEM_PROMPT = """你是专门处理旅行推荐的助手。
当用户需要帮助预订推荐旅行时，主助手会将任务委派给你。
根据用户偏好搜索可用旅行推荐，并向客户确认预订详情。
如果需要更多信息或客户改变主意，将任务升级回主助手。
搜索时要坚持，如果第一次搜索没有结果，请扩大查询范围。
记住，预订在成功调用相应工具后才算完成。

当前时间: {time}。

如果用户需要帮助，但没有合适的工具，请调用 CompleteOrEscalate 将对话交还给主助手。不要浪费用户的时间。不要编造无效的工具或函数。

以下情况应该调用 CompleteOrEscalate:
 - '算了，我自己单独预订吧'
 - '我需要在那边安排交通'
 - '哦等等，我还没订机票，我先订那个'
 - '旅行预订确认！'"""


SAFE_TOOLS = [search_trip_recommendations]
SENSITIVE_TOOLS = [book_excursion, update_excursion, cancel_excursion]
ALL_TOOLS = SAFE_TOOLS + SENSITIVE_TOOLS


def get_prompt():
    """获取旅行助手的提示词模板"""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{messages}"),
    ]).partial(time=datetime.now)


def create_runnable(llm):
    """创建旅行助手的可执行链"""
    return get_prompt() | llm.bind_tools(ALL_TOOLS + [CompleteOrEscalate])


def route(state):
    """
    旅行助手路由函数。
    
    决策流程：
    1. 没有工具调用 → 结束
    2. CompleteOrEscalate → 返回主助手
    3. 安全工具 → 执行安全工具节点
    4. 敏感工具 → 执行敏感工具节点（会被中断等待确认）
    """
    route = tools_condition(state)
    if route == END:
        return END
        
    tool_calls = state["messages"][-1].tool_calls
        # __name__ 是 Python 的内置属性，返回CompleteOrEscalate的类名，在这里就是"CompleteOrEscalate"
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
        
    safe_toolnames = [t.name for t in SAFE_TOOLS]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "book_excursion_safe_tools"
        
    return "book_excursion_sensitive_tools"
