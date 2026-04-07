from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """
    对话栈更新函数 - 用于管理多 Agent 对话状态。

    这是一个自定义的 Reducer 函数，配合 Annotated 使用：
        dialog_state: Annotated[list[str], update_dialog_stack]

    功能：
        实现栈（Stack）的入栈和出栈操作，追踪当前在哪个 Agent 对话。

    参数：
        left: 当前的对话栈列表，如 ["assistant", "update_flight"]
        right: 操作指令，可以是：
            - None: 不操作，返回原栈
            - "pop": 出栈，移除最后一个元素（返回主助手）
            - 其他字符串: 入栈，添加到列表末尾（进入子助手）

    返回：
        更新后的对话栈列表

    示例：
        >>> update_dialog_stack(["assistant"], "update_flight")
        ["assistant", "update_flight"]  # 进入航班助手

        >>> update_dialog_stack(["assistant", "update_flight"], "pop")
        ["assistant"]  # 返回主助手

        >>> update_dialog_stack(["assistant", "update_flight"], None)
        ["assistant", "update_flight"]  # 无变化

        场景：多层嵌套
        dialog_state: ["assistant", "book_hotel", "update_flight"]
        节点返回: {"dialog_state": "pop"}
            ↓
        update_dialog_stack(
            left=["assistant", "book_hotel", "update_flight"],
            right="pop"
        )
            ↓
        返回: ["assistant", "book_hotel"]
    """
    if right is None:
        # 没有新操作，保持原栈不变
        return left
    if right == "pop":
        # 出栈：完成任务，返回上一层（主助手）
        return left[:-1]
    # 入栈：进入新的子助手
    return left + [right]


class State(TypedDict):
    # 存储对话历史消息
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str     #存储用户航班信息（从数据库查询）
    dialog_state: Annotated[
        list[                           # 1. 外层：列表类型
            Literal[                    # 2. 中层：Literal限定dialog_state 里面的list列表里只能是这些值之一或多个
                "assistant",            #    - 主助手
                "update_flight",        #    - 航班助手
                "book_car_rental",      #    - 租车助手
                "book_hotel",           #    - 酒店助手
                "book_excursion",       #    - 旅行助手
            ]
        ],
        update_dialog_stack,           # 3. Reducer：栈操作
    ]
