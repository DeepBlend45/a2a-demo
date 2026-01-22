import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

 
# a2a-with-library/supervisor/__main__.py
import logging

import asyncio

# LangGraph A2A Client
from langgraph_a2a_client import A2AClientToolProvider

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create A2A client tool provider with known agent URLs
provider = A2AClientToolProvider(
    known_agent_urls=[
        "http://127.0.0.1:9000",  # Currency Agent
    ]
)

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=provider.tools,
    system_prompt="You are a team supervisor managing a currency agent and a weather information agent.",
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {
                    "allowed_decisions": ["approve", "reject"]
                },
                "search_web": False,
            }
        ),
    ]
)
async def main():
    thread_id = "techorus-demo"
    user_input = input("メッセージを入力してください: ")
    config = {"configurable": {"thread_id": thread_id}}

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config
    )

    while result.get("__interrupt__"):
        interrupts = result["__interrupt__"]
        print("\n=== ツール実行の承認が必要です ===")

        for interrupt in interrupts:
            action_requests = interrupt.value.get("action_requests", [])
            for action in action_requests:
                print(f"\nツール: {action['name']}")
                print(f"引数: {action['args']}")

        decision = input("\n承認しますか？ (approve/reject): ").strip().lower()

        if decision == "approve":
            result = await agent.ainvoke(
                Command(resume={"decisions": [{"type": "approve", "message": "ユーザーが許可しました"}]}),
                config
            )
        elif decision == "reject":
            result = await agent.ainvoke(
                Command(resume={"decisions": [{"type": "reject", "message": "ユーザーが拒否しました"}]}),
                config
            )
        else:
            print("無効な入力です。approve または reject を入力してください。")
            continue

    print("\n=== 応答 ===")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())