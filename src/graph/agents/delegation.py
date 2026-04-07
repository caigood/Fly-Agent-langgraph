"""
任务委派（Task Delegation）
===========================

定义主助手如何将任务委派给专门助手。

委派机制：
1. 主助手调用委派工具（如 ToFlightBookingAssistant）
2. 路由函数检测到委派工具调用
3. 跳转到对应专门助手的入口节点
4. 入口节点发送切换消息，更新 dialog_state

委派类：
- ToFlightBookingAssistant → 航班助手
- ToBookCarRental → 租车助手
- ToHotelBookingAssistant → 酒店助手
- ToBookExcursion → 旅行助手
"""

from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from src.agents.state import State


class ToFlightBookingAssistant(BaseModel):
    """
    将任务转交给专门处理航班更新和取消的助手。
    
    当主助手识别到用户需要更新或取消航班时，会调用这个工具。
    这不是真正的工具调用，而是触发 Agent 切换的信号。
    """
    request: str = Field(
        description="航班助手在继续之前需要澄清的任何后续问题。"
    )


class ToBookCarRental(BaseModel):
    """
    将任务转交给专门处理租车预订的助手。
    
    包含租车所需的关键信息：地点、日期等。
    主助手会从对话中提取这些信息并传递给租车助手。
    """
    #Field 是 Pydantic 的函数，用于给模型字段添加元数据
    # 是给字段加"说明书"和"约束条件"的！
    location: str = Field(
        description="用户想要租车的地点。"
    )
    start_date: str = Field(description="租车的开始日期。")
    end_date: str = Field(description="租车的结束日期。")
    request: str = Field(
        description="用户关于租车的任何额外信息或要求。"
    )
    #上面的是所需参数说明，下面是参数示意，是one-shot
    class Config:
        json_schema_extra = {
            "example": {
                "location": "巴塞尔",
                "start_date": "2023-07-01",
                "end_date": "2023-07-05",
                "request": "我需要一辆自动挡的紧凑型车。",
            }
        }


class ToHotelBookingAssistant(BaseModel):
    """
    将任务转交给专门处理酒店预订的助手。
    
    包含酒店预订所需的关键信息：地点、入住/退房日期等。
    """
    location: str = Field(
        description="用户想要预订酒店的地点。"
    )
    checkin_date: str = Field(description="酒店的入住日期。")
    checkout_date: str = Field(description="酒店的退房日期。")
    request: str = Field(
        description="用户关于酒店预订的任何额外信息或要求。"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "苏黎世",
                "checkin_date": "2023-08-15",
                "checkout_date": "2023-08-20",
                "request": "我更喜欢市中心附近有景观房间的酒店。",
            }
        }


class ToBookExcursion(BaseModel):
    """
    将任务转交给专门处理旅行推荐和游览预订的助手。
    
    包含旅行推荐所需的关键信息：地点、用户偏好等。
    """
    location: str = Field(
        description="用户想要预订推荐旅行的地点。"
    )
    request: str = Field(
        description="用户关于旅行推荐的任何额外信息或要求。"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "卢塞恩",
                "request": "用户对户外活动和风景感兴趣。",
            }
        }


def create_entry_node(assistant_name: str, new_dialog_state: str):
    """
    创建一个入口节点，用于切换到专门助手。
    
    参数：
        assistant_name: 助手名称（用于提示消息）
        new_dialog_state: 新的对话状态（如 "update_flight", "book_hotel" 等）
    
    返回：
        一个节点函数，会被添加到图中
    
    工作原理：
        1. 获取上一个消息中的 tool_call_id
        2. 创建一个 ToolMessage，告诉专门助手"现在轮到你了"
        3. 更新 dialog_state，标记当前在哪个助手
        4. 本质是因为这个切换助手的工具，是人为制造的假工具，而langgraph需要调用工具后返回ToolMessage响应，这里返回调用的响应
    """
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"现在切换到{assistant_name}。请回顾上面的对话。"
                    f"用户的意图尚未满足。使用可用的工具来帮助用户。"
                    f"记住，你是{assistant_name}，"
                    "预订、更新或其他操作在成功调用相应工具后才算完成。"
                    "如果用户改变主意或需要其他帮助，调用 CompleteOrEscalate 函数让主助手接管。"
                    "不要提及你是谁 - 直接作为助手代理行事。",
                    #  OpenAI/通义千问 等模型在返回 tool_calls 时，会自动分配唯一 ID
                    #  一个对话中 同一个工具被调用两次，那么他们的tool_call_id不同
                    #  tool_call_id是工具调用的身份证号
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node


def pop_dialog_state(state: State) -> dict:
    """
    退出专门助手，返回主助手。
    
    工作原理：
        1. 清空 dialog_state
        2. 告诉主助手专门助手已完成任务
    """
    messages = []
    if state["messages"][-1].tool_calls:
        messages.append(
            ToolMessage(
                content="任务已完成，现在返回主助手。"
                "请回顾对话历史，看看是否还有其他需要帮助的地方。",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }
