import os
from strands import Agent, tool
from fastapi import FastAPI
import uvicorn
import logging
from strands.multiagent.a2a import A2AServer
from strands.models.openai import OpenAIModel
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

model = OpenAIModel(
    client_args={"api_key": os.environ["OPENAI_API_KEY"]},
    model_id="gpt-4o-mini",
    params={"temperature": 0.2},
)

@tool
def get_exchange_rate():
    """Get the exchange rate between USD and JPY."""
    return "1 USD = 147円"

# Create a Strands agent
strands_agent = Agent(
    name="Carrency Agent",
    model=model,
    description="A carrency agent that can exchange rate between USD and JPY.",
    tools=[get_exchange_rate],
    callback_handler=None
)

# Create A2A server (streaming enabled by default)
a2a_server = A2AServer(agent=strands_agent)

# # Start the server
# a2a_server.serve()

app = FastAPI()

@app.get("/ping")
def ping():
    return {"status": "healthy"}

app.mount("/", a2a_server.to_fastapi_app())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)