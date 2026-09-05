#%%
# SETUP
import os
import sys
import warnings
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter3_llm_evals"
repo = "ARENA_3.0"
branch = "main"

# Install dependencies
try:
    import inspect_ai
except:
    %pip install openai>=1.58.1 anthropic inspect_ai tabulate wikipedia jaxtyping python-dotenv datasets

# Get root directory, handling 3 different cases: (1) Colab, (2) notebook not in ARENA repo, (3) notebook in ARENA repo
root = (
    "/content"
    if IN_COLAB
    else "/root"
    if repo not in os.getcwd()
    else str(next(p for p in Path.cwd().parents if p.name == repo))
)

if Path(root).exists() and not Path(f"{root}/{chapter}").exists():
    if not IN_COLAB:
        !sudo apt-get install unzip
        %pip install jupyter ipython --upgrade

    if not os.path.exists(f"{root}/{chapter}"):
        !wget -P {root} https://github.com/callummcdougall/ARENA_3.0/archive/refs/heads/{branch}.zip
        !unzip {root}/{branch}.zip '{repo}-{branch}/{chapter}/exercises/*' -d {root}
        !mv {root}/{repo}-{branch}/{chapter} {root}/{chapter}
        !rm {root}/{branch}.zip
        !rmdir {root}/{repo}-{branch}

if IN_COLAB:
    from google.colab import output, userdata
    !pip install pyngrok
    from pyngrok import ngrok
    import threading
    import time

    for key in ["OPENAI", "ANTHROPIC"]:
        try:
            os.environ[f"{key}_API_KEY"] = userdata.get(f"{key}_API_KEY")
        except:
            warnings.warn(
                f"You don't have a '{key}_API_KEY' variable set in the secrets tab of your google colab. You have to set one, or calls to the {key} API won't work."
            )

# Handles running code in an ipynb
if "__file__" not in globals() and "__vsc_ipynb_file__" in globals():
    __file__ = globals()["__vsc_ipynb_file__"]

if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")

os.chdir(f"{root}/{chapter}/exercises")

import os
import random
import re
import sys
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Any, Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Dataset, Sample, example_dataset, hf_dataset, json_dataset
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Scorer, Target, answer, match, model_graded_fact, scorer
from inspect_ai.solver import (
    Choices,
    Generate,
    Solver,
    TaskState,
    chain,
    chain_of_thought,
    generate,
    self_critique,
    solver,
)
from openai import OpenAI

# Make sure exercises are in the path
chapter = "chapter3_llm_evals"
section = "part3_running_evals_with_inspect"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part3_running_evals_with_inspect.tests as tests

MAIN = __name__ == "__main__"
#%%
load_dotenv()
assert os.getenv("OPENROUTER_API_KEY") is not None, "You must set your OpenRouter API key - see instructions in dropdown"

openrouter_client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
# %%
dataset = example_dataset("theory_of_mind")
pprint(dataset.samples[0].__dict__)
# %%
# helper functions
def record_to_sample(record: dict):
    """
    Turns a dataset record into a sample object, map fields of record into fields of sample object.
    """
    input = [ChatMessageUser(content=record["question"])]
    has_system_prompt = record.get("system", "") != ""
    if has_system_prompt:
        input.insert(0, ChatMessageSystem(content=record["system"]))

    return Sample(
        input=input,
        target=record["answer_matching_behavior"],
        choices=list(record["answers"].values()),
        metadata={
            "labels": list(record["answers"].keys()),
            "behavior_category": record["behavior_category"],
            "system_prompt": has_system_prompt,
        },
    )


# %%
@task
def theory_of_mind():
    return Task(
        dataset=example_dataset("theory_of_mind"),
        solver=[chain_of_thought(), generate(), self_critique(model="openrouter/openai/gpt-4o-mini")],
        scorer=model_graded_fact(model="openrouter/openai/gpt-4o-mini"),
    )

log = eval(theory_of_mind(), model="openrouter/openai/gpt-4o-mini", limit=10, log_dir=str(section_dir / "logs"))
