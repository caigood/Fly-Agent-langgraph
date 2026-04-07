from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig

from src.agents.state import State


class Assistant:
    """
    助手包装器类 - 封装 LLM 调用逻辑。

    作用：
        1. 调用 LLM 生成回复
        2. 处理 LLM 返回空内容的情况（自动重试）
        3. 返回标准化的消息格式

    使用方式：
        assistant = Assistant(runnable)  # runnable = 提示词 + LLM + 工具
        builder.add_node("node_name", assistant)  # LangGraph 会自动调用 __call__
    """

    def __init__(self, runnable: Runnable):
        """
        初始化助手。

        参数：
            runnable: 可运行的链（提示词模板 + LLM + 工具绑定）
                     例如：get_prompt() | llm.bind_tools(tools)
        """
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        """
        执行助手逻辑 - 被 LangGraph 自动调用。

        参数：
            state: 当前状态，包含 messages、user_info 等
            config: 运行配置

        返回：
            {"messages": result} - 把 LLM 的回复加入消息历史

        逻辑流程：
            1. 调用 runnable.invoke(state) 获取 LLM 回复
            2. 检查回复是否有效：
               - 有 tool_calls（工具调用）→ 有效，跳出循环
               - 有 content（文本内容）→ 有效，跳出循环
               - 空回复 → 无效，添加提示"请给出实际回复"，重新调用
            3. 返回结果
        """
        while True:
            # 调用 LLM（执行提示词 + 模型推理）
            result = self.runnable.invoke(state)

            # 检查是否是空回复（没有工具调用，且没有文本内容）
            # 这种情况通常发生在 LLM "卡住" 或回复格式异常时
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                # 空回复：添加系统提示，要求重新生成
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
                # 继续循环，重新调用 LLM
            else:
                # 有效回复：跳出循环
                break

        # 返回 LLM 的回复，LangGraph 会自动合并到 state["messages"]
        return {"messages": result}


from pydantic import BaseModel, Field


class CompleteOrEscalate(BaseModel):
    """一个工具，用于标记当前任务已完成，和/或将对话控制权交还给主助手，主助手可以根据用户需求重新路由对话。"""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }
