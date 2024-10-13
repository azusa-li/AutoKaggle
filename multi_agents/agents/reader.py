from typing import Dict, Any
import json
import re
import logging
import sys 
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sys.path.append('..')
sys.path.append('../..')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_base import Agent
from utils import read_file, PREFIX_MULTI_AGENTS
from llm import LLM
from state import State
from prompts.prompt_base import *
from prompts.prompt_reader import *

class Reader(Agent):
    def __init__(self, model: str, type: str):
        super().__init__(
            role="reader",
            description="You are good at reading document and summarizing information.",
            model=model,
            type=type
        )
    
    def _execute(self, state: State, role_prompt: str) -> Dict[str, Any]:
        path_to_overview = f'{PREFIX_MULTI_AGENTS}/competition/{state.competition}/overview.txt'
        overview = read_file(path_to_overview)
        history = []
        round = 0
        # Understand Background 读取overview.txt，生成competition_info.txt
        if len(state.memory) == 1: # 如果之前没有memory，说明是第一次执行
            history.append({"role": "system", "content": f"{role_prompt}{self.description}"})
            # pdb.set_trace()
            while True:
                if round == 0:
                    task = PROMPT_READER_TASK
                    input = PROMPT_READER.format(phases_in_context=state.context, task=task)
                elif round == 1: 
                    input = f"\n#############\n# OVERVIEW #\n{overview}"
                    input += self._read_data(state, num_lines=1)
                elif round == 2: 
                    reader_mid_reply = raw_reply
                    input = PROMPT_READER_ROUND2
                elif round == 3: 
                    break
                raw_reply, history = self.llm.generate(input, history, max_tokens=4096)
                round += 1
        else: # 如果之前有memory，拼接之前memory中reader的结果作为experience
            self.description = "You are good at reading document and summarizing information." \
                            "You have advanced reasoning abilities and can improve your answers through reflection."
            experience_with_suggestion = self._gather_experience_with_suggestion(state)
            history.append({"role": "system", "content": f"{role_prompt} {self.description}"})
            while True:
                if round == 0:
                    task = PROMPT_READER_TASK
                    input = PROMPT_READER_WITH_EXPERIENCE_ROUND0.format(phases_in_context=state.context, task=task, experience_with_suggestion=experience_with_suggestion)
                elif round == 1: 
                    input = f"# OVERVIEW #\n{overview}\n############# "
                    input += self._read_data(state, num_lines=1)
                elif round == 2:
                    reader_mid_reply = raw_reply
                    input = PROMPT_READER_WITH_EXPERIENCE_ROUND2
                elif round == 3: 
                    break
                raw_reply, history = self.llm.generate(input, history, max_tokens=4096)
                round += 1
        result = raw_reply
        reply = self._parse_markdown(raw_reply)

        summary = reply 

        # 保存history
        with open(f'{state.restore_dir}/{self.role}_history.json', 'w') as f:
            json.dump(history, f, indent=4)
        with open(f'{state.competition_dir}/competition_info.txt', 'w') as f:
            f.write(summary)
        with open(f'{state.restore_dir}/{self.role}_reply.txt', 'w') as f:
            f.write(raw_reply)
        with open(f'{state.restore_dir}/{self.role}_mid_reply.txt', 'w') as f:
            f.write(reader_mid_reply)
        input_used_in_review = f"   <overview>\n{overview}\n    </overview>"

        print(f"State {state.phase} - Agent {self.role} finishes working.")
        return {
            self.role: {
                "history": history,
                "role": self.role,
                "description": self.description,
                "task": PROMPT_READER_TASK,
                "input": input_used_in_review,
                "summary": summary,
                "result": result
            }
        }