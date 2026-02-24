import os

from datapizza.agents import Agent
from datapizza.clients.google import GoogleClient
from datapizza.tools import tool
from dotenv import load_dotenv
from prompts.data_engineer import SYSTEM_PROMPT as DATA_ENGINEER_PROMPT
from prompts.data_scientist import SYSTEM_PROMPT as DATA_SCIENTIST_PROMPT
from prompts.tech_lead import SYSTEM_PROMPT as TECH_LEAD_PROMPT
from tools.execute_code import execute_code

load_dotenv()

client = GoogleClient(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.5-flash",
    system_prompt="You are a helpful AI assistant.",
)


code_interpreter_state = {}

@tool
def code_interpreter(code: str, agent_name: str) -> str:
    """Use this tool to execute the Python code you have written.
    - if it tells you code is not safe, return this information as a response
    - if it gives you errors due to code execution, correct the code
    - always end with a print statement as verification output    
    """
    
    print(f"Executing code for: {agent_name}")
    print(f"Code: {code}")
    state = code_interpreter_state.get(f"{agent_name}_state", {})
    code_interpreter_state[f"{agent_name}_state"] = state
    
    return execute_code(code, state)


tech_lead = Agent(
    name="tech_lead",
    client=client,
    system_prompt=TECH_LEAD_PROMPT,
    stateless=False,  # this means that the agent will keep the state of the conversation, otherwise it would be same as calling it for the first time every time (i.e. it forgets previous interactions)
)


# no need for stateless=False since they are agents-as-a-tool (so they can forget what happened before)

data_engineer = Agent(
    name="data_engineer",
    client=client,
    system_prompt=DATA_ENGINEER_PROMPT,
    tools=[code_interpreter],
) 

data_scientist = Agent(
    name="data_scientist",
    client=client,
    system_prompt=DATA_SCIENTIST_PROMPT,
    tools=[code_interpreter],
)


tech_lead.can_call([data_engineer, data_scientist])  # tech lead can invoke other agents


while True:
    user_input = input("Input: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    response = tech_lead.run(user_input)
    print(response.text)