import asyncio
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console

async def main():
    # 1. Initialize the local Ollama client
    # Pointing to default Ollama port 11434
    model_client = OllamaChatCompletionClient(
        model="llama3.1:8b",
        base_url="http://localhost:11434",
    )

    # 2. Create a simple agent
    agent = AssistantAgent(
        name="test_agent",
        model_client=model_client,
        system_message="You are a helpful assistant confirming the system works."
    )

    # 3. Run a quick query
    print("Sending message to Llama-8b...")
    await Console(agent.run_stream(task="Say 'The GPU is ready'"))

if __name__ == "__main__":
    asyncio.run(main())