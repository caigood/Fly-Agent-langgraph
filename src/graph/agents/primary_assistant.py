"""
主助手（Primary Assistant）
============================

这是整个系统的入口和核心路由器。

职责：
1. 回答简单查询（航班信息、政策查询）
2. 将复杂任务委派给专门助手

委派工具：
- ToFlightBookingAssistant → 航班助手
- ToBookCarRental → 租车助手
- ToHotelBookingAssistant → 酒店助手
- ToBookExcursion → 旅行助手
"""

from datetime import datetime
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from langgraph.prebuilt import tools_condition

from src.tools import (
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
)
from .delegation import (
    ToFlightBookingAssistant,
    ToBookCarRental,
    ToHotelBookingAssistant,
    ToBookExcursion,
)


SYSTEM_PROMPT = """你是瑞士航空的客服助手。
                你的主要职责是搜索航班信息和公司政策来回答客户问题。
                如果客户要求更新或取消航班、预订租车、预订酒店或获取旅行推荐，
                请通过调用相应的工具将任务委派给专门助手。你自己无法进行这些类型的更改。
                只有专门助手才能处理这些请求。
                向专门助手提供用户到目前为止提供的所有相关信息。
                搜索时要坚持，如果第一次搜索没有结果，请扩大查询范围。
                如果搜索结果为空，在放弃之前先扩大搜索范围。

                当前用户航班信息:
                <航班>
                {user_info}
                </航班>

                当前时间: {time}。"""


def get_prompt():
    """获取主助手的提示词模板"""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{messages}"),
    ]).partial(time=datetime.now)


TOOLS = [
    TavilySearch(max_results=1),            # 网络搜索工具 - 查询实时信息
    fetch_user_flight_information,          # 获取用户航班信息 - 查询当前预订
    search_flights,                         # 搜索航班 - 查找可用航班
    lookup_policy,                          # 查询政策 - 搜索公司政策文档
    ToFlightBookingAssistant,               # 委派工具 - 转交航班助手处理改签/取消
    ToBookCarRental,                        # 委派工具 - 转交租车助手处理租车预订
    ToHotelBookingAssistant,                # 委派工具 - 转交酒店助手处理酒店预订
    ToBookExcursion,                        # 委派工具 - 转交旅行助手处理推荐/预订
]


def create_runnable(llm):
    """创建主助手的可执行链"""
    # LCEL 管道语法：prompt | llm_with_tools
    # 纯命令式等价写法：
    #   prompt_template = get_prompt()                    # 1. 获取提示词模板
    #   llm_with_tools = llm.bind_tools(TOOLS)            # 2. LLM 绑定工具
    #   messages = prompt_template.format_messages(...)   # 3. 模板生成消息列表
    #   result = llm_with_tools.invoke(messages)          # 4. 调用 LLM，返回 AIMessage
    return get_prompt() | llm.bind_tools(TOOLS)
    # 实际内部执行流程
    # 1. get_prompt() 接收输入
    # prompt_value = get_prompt().invoke({"messages": [...], "user_info": "..."})
    # 返回 ChatPromptValue（已填充数据的提示词入{user_info}） 
    # 注意，上面的这个get_prompt().invoke，不是调用大模型的那个invoke，
    # 而是调用ChatPromptTemplate的invoke，只是用来填充提示词模板中的变量，下面的步骤2才是调用大模型的invoke。
    # 2. 传递给 LLM
    # result = llm_with_tools.invoke(prompt_value)
    # 返回 AIMessage

def route(state):
    """
    主助手路由函数 - 整个系统的核心路由逻辑。
    
    决策流程：
    1. 没有工具调用 → 结束对话
    2. 委派工具 → 进入对应专门助手入口
    3. 其他工具 → 执行普通工具节点
    
    委派工具检测：
    - ToFlightBookingAssistant → enter_update_flight
    - ToBookCarRental → enter_book_car_rental
    - ToHotelBookingAssistant → enter_book_hotel
    - ToBookExcursion → enter_book_excursion
    """
    # 1. 先用 tools_condition 检查是否有工具调用
    #    如果没有工具调用，返回 字符串"__end__"，对话结束
    #    如果有工具调用，返回字符串"tools" ,只是一个普通的字符串,不是列表。
    #    下面的END是变量，本质是字符串"__end__"，只是为了方便阅读，所以用END表示，所以能和返回值有下划线的end比较
    route = tools_condition(state)
    if route == END:
        return END
    
    # 2. 获取 LLM 的 tool_calls（LLM 决定调用哪些工具）
    #    state["messages"][-1] 是最后一条消息（AI 的回复消息）
    #    tool_calls 是一维数组，元素是字典，name 是字典的 key
    tool_calls = state["messages"][-1].tool_calls
    
    # 3. 检查是否是委派工具（触发 Agent 切换）
    #    根据 tool_calls[0]["name"] 判断 LLM 想调用哪个工具
    if tool_calls:
        # 委派给航班助手（处理改签/取消）
        #__name__是Python 内置属性，返回类的名称字符串
        if tool_calls[0]["name"] == ToFlightBookingAssistant.__name__:
            return "enter_update_flight"
        # 委派给租车助手
        elif tool_calls[0]["name"] == ToBookCarRental.__name__:
            return "enter_book_car_rental"
        # 委派给酒店助手
        elif tool_calls[0]["name"] == ToHotelBookingAssistant.__name__:
            return "enter_book_hotel"
        # 委派给旅行助手
        elif tool_calls[0]["name"] == ToBookExcursion.__name__:
            return "enter_book_excursion"
    
    # 4. 普通工具调用（搜索航班、查询政策等）
    #    执行工具节点，然后返回主助手继续对话
    return "primary_assistant_tools"
