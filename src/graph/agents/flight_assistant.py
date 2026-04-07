"""
航班助手（Flight Booking Assistant）
=====================================

专门处理航班改签和取消。

职责：
1. 搜索可用航班
2. 更新航班预订
3. 取消航班预订

工具分类：
- 安全工具：search_flights（只读查询）
- 敏感工具：update_ticket_to_new_flight, cancel_ticket（需要用户确认）
"""

from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from langgraph.prebuilt import tools_condition

from src.tools import (
    search_flights,
    update_ticket_to_new_flight,
    cancel_ticket,
)
from src.agents.assistant import CompleteOrEscalate


SYSTEM_PROMPT = """你是专门处理航班更新的助手。
当用户需要帮助更新预订时，主助手会将任务委派给你。
向客户确认更新后的航班详情，并告知任何额外费用。
搜索时要坚持，如果第一次搜索没有结果，请扩大查询范围。
如果需要更多信息或客户改变主意，将任务升级回主助手。
记住，预订在成功调用相应工具后才算完成。

当前用户航班信息:
<航班>
{user_info}
</航班>

当前时间: {time}。

如果用户需要帮助，但没有合适的工具，请调用 CompleteOrEscalate 将对话交还给主助手。不要浪费用户的时间。不要编造无效的工具或函数。"""


SAFE_TOOLS = [search_flights]
SENSITIVE_TOOLS = [update_ticket_to_new_flight, cancel_ticket]
ALL_TOOLS = SAFE_TOOLS + SENSITIVE_TOOLS


def get_prompt():
    """获取航班助手的提示词模板"""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{messages}"),
    ]).partial(time=datetime.now)


def create_runnable(llm):
    """创建航班助手的可执行链"""
    return get_prompt() | llm.bind_tools(ALL_TOOLS + [CompleteOrEscalate])


def route(state):
    """
    航班助手路由函数。
    
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
    
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
        
    #得到安全工具的名称列表，safe_toolnames是一个字符串列表，每个元素是安全工具的名称字符串
    safe_toolnames = [t.name for t in SAFE_TOOLS]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "update_flight_safe_tools"
    # 返回的字符串都是图中节点的名字
    return "update_flight_sensitive_tools"
