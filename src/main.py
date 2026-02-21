from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from schemas import AgentResponse

llm = ChatGroq(model="openai/gpt-oss-120b")
tools = [TavilySearch(max_results=3)]
agent = create_agent(
        model=llm, 
        tools=tools,
    )

def main():
    print("Asking the agent...")
    result = agent.invoke(
        {
            "messages": [{
                    "role": "user",
                    "content": "search for the top 3 latest news on deep space research and provide the sources",
                }
            ],
        }
    )
    print(f"Agent response: {result}")

if __name__ == "__main__":
    main()