"""
航空客服助手 - 主入口文件
==============================

这个文件是整个项目的入口点，负责：
1. 加载环境变量和配置
2. 管理对话循环
3. 处理 Human in the Loop（人工确认机制）
"""

import getpass
import os
import uuid

from dotenv import load_dotenv

from src.graph import create_graph
from src.utils.helpers import _print_event, update_dates
from src.config import local_file

load_dotenv()


def _set_env(var: str):
    """检查环境变量是否存在，如果不存在则提示用户输入。"""
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


def main():
    _set_env("TAVILY_API_KEY")
    update_dates(local_file)

    print("\n" + "=" * 50)
    print("瑞士航空客服助手 - 多 Agent 专业分工版")
    print("=" * 50)

    graph = create_graph()
    #全球唯一的随机ID，用于标识当前对话
    thread_id = str(uuid.uuid4())
    #config 是给框架和工具用的，不传给大模型！
    config = {
        "configurable": {
            "passenger_id": "3442 587242",    # 乘客ID，用于查询数据库
            "thread_id": thread_id,           # 对话线程ID，用于标识当前对话
        }
    }

    print("\n对话已开始！输入 'quit' 或 'exit' 退出\n")

    _printed = set()
    
    while True:
        try:
            user_input = input("你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break
        
        if user_input.lower() in ["quit", "exit", "退出", "q"]:
            print("再见！")
            break
        
        if not user_input:
            continue

        events = graph.stream(
            {"messages": ("user", user_input)}, config, stream_mode="values"
        )
        for event in events:
            _print_event(event, _printed)
        
        snapshot = graph.get_state(config)
        
        while snapshot.next:
            try:
                approval = input(
                    "\n是否批准上述操作？输入 'y' 继续；"
                    "否则请说明您要求的修改: "
                )
            except (KeyboardInterrupt, EOFError):
                approval = "y"
            
            if approval.strip().lower() == "y":
                graph.invoke(None, config)
            else:
                from langchain_core.messages import ToolMessage
                graph.invoke(
                    {
                        "messages": [
                            ToolMessage(
                                tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                content=f"用户拒绝了操作。原因：'{approval}'。请根据用户的输入继续协助。",
                            )
                        ]
                    },
                    config,
                )
            snapshot = graph.get_state(config)


if __name__ == "__main__":
    main()
