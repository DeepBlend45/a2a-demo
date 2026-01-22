# a2a-with-library/sub_agents/weather_agent.py
import logging

from a2a.types import AgentSkill

# LangGraph A2A Server
from langgraph_a2a_server import A2AServer

# LangChain/LangGraph
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = '0.0.0.0'
PORT = 20000


@tool
def get_weather(city_name: str) -> str:
    """Get the current weather."""
    return f"{city_name} is Sunny"


# Create LangGraph agent
agent = create_agent(
    model=init_chat_model(model='gpt-4.1-nano'),
    tools=[get_weather],
    checkpointer=InMemorySaver(),
)

# Create A2A server with the agent
server = A2AServer(
    graph=agent,
    description='Fetch the current weather for a location',
    host=HOST,
    port=PORT,
    skills=[
        AgentSkill(
            id='get_weather',
            name='Get Weather',
            description='Fetch the current weather for a location',
            tags=['weather', 'location'],
            examples=['What is the weather like in New York?', 'Tell me the weather in San Francisco'],
        )
    ],
)

# Start the server
if __name__ == '__main__':
    server.serve()