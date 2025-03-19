from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_ibm import WatsonxLLM
import os

os.environ[""] = "API KEY"
os.environ["Serper_API_KEY"]="Serper_API_KEY"

parameters = {"decoding_method": "greedy", "max_new_tokens": 500}

lllm = WatsonxLLM(
    model_id="ibm/granite-20b-chat-v1",
    url="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    project_id="",
    parameters=parameters,
)

function_calling_llm = WatsonxLLM(
    model_id="ibm/merlin-20b-chat-v1",
    url="XXXXXXXXXXXXXXXXXXXXXXXX",
    project_id="",
    parameters=parameters,
)

search = SerperDevTool()

researcher = Agent(
    role="Researcher",
    goal="Research the topic",
    backstory="You are a researcher",
    llm=lllm,
    tools=[search],
    verbose=1,
    allow_delegation=False,
    function_calling_llm=function_calling_llm,
)

task1 = Task(
    description="Search the internet and find 5 examples of promising AI research.",
    expected_output="A detailed bullet point summary on each of the topics. Each bullet point should cover the topic, background and why the innovation is useful.",
    output_file="task1output.txt",
    agent=researcher,
)

writer = Agent(
    role="Writer",
    goal="Write a detailed report on the topic",
    backstory="You are a writer",
    llm=lllm,
    verbose=1,
    allow_delegation=False,
    function_calling_llm=function_calling_llm,
)

task2 = Task(
    description="Write a detailed report on the topic.",
    expected_output="A detailed report on the topic.",
    output_file="task2output.txt",
    agent=writer,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2,
)
print(crew.run())