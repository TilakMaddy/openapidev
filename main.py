from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType

import openai
import os
import classifier
import pandas as pd

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# set PARTIAL to -1 if you want a complete result
PARTIAL = 2 

def generate_csv():
    global PARTIAL
    
    ai = classifier.DebugBuildClassifier(filename='build.debug.output')
    ai.run(PARTIAL)

def dump_excel_sheet():
    results = Path.cwd() / "results.csv"
    assert(results.exists())
    pd.read_csv("results.csv").to_excel("hints.xlsx", index=None, header=True)

def qaloop():
    agent = create_csv_agent(
        OpenAI(temperature=0),
        "results.csv",
        verbose=True,
    )

    while True:
        print(agent.run(input()))


def main():
    for f in [
        generate_csv,
        dump_excel_sheet,
        qaloop
    ]:
        f()

if __name__ == '__main__':
    main()