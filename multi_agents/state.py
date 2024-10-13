import os
import sys
import json
import pdb

sys.path.append('..')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict, Any, Optional
from utils import PREFIX_MULTI_AGENTS, load_config
from prompts.prompt_base import PHASES_IN_CONTEXT_PREFIX

class State:
    def __init__(self, phase: str, competition: str, message: str = "There is no message."):
        self.phase = phase
        self.competition = competition
        self.message = message
        self.memory: List[Dict[str, Any]] = [{}]
        self.current_step = 0
        self.score = 0
        self.finished = False
        
        config = load_config(f'{PREFIX_MULTI_AGENTS}/config.json')
        self.agents = config['phase_to_agents'][self.phase]
        self.phase_to_directory = config['phase_to_directory']
        self.phase_to_unit_tests = config['phase_to_unit_tests']
        self.rulebook_parameters = config['rulebook_parameters']
        self.phase_to_ml_tools = config['phase_to_ml_tools']
        
        self.function_to_schema = load_config(f'{PREFIX_MULTI_AGENTS}/function_to_schema.json')
        
        self.competition_dir = f'{PREFIX_MULTI_AGENTS}/competition/{self.competition}'
        self.dir_name = self.phase_to_directory[self.phase]
        self.restore_dir = f'{self.competition_dir}/{self.dir_name}'
        self.ml_tools = self.phase_to_ml_tools[self.phase]
        self.context = ""

    def __str__(self) -> str:
        return f"State: {self.phase}, Current Step: {self.current_step}, Current Agent: {self.agents[self.current_step]}, Finished: {self.finished}"

    def make_context(self) -> None:
        self.context = PHASES_IN_CONTEXT_PREFIX.replace("# {competition_name}", self.competition)
        phases = load_config(f'{PREFIX_MULTI_AGENTS}/config.json')['phases']
        self.context += "\n".join(f"{i+1}. {phase}" for i, phase in enumerate(phases))

    def get_state_info(self) -> str:
        if self.phase == 'Data Cleaning':
            return "In Data Cleaning, you already have file `train.csv' and `test.csv`. In this phase, the overall goal is to do data cleaning to them to get `cleaned_train.csv` and `cleaned_test.csv`"
        elif self.phase == 'Feature Engineering':
            return "After Data Cleaning, you already have file `cleaned_train.csv' and `cleaned_test.csv`. In this phase, the overall goal is to do feature engineering to them to get `processed_train.csv` and `processed_test.csv`"
        elif self.phase == 'Model Building, Validation, and Prediction':
            return ("In this phase, you have `processed_train.csv` and `processed_test.csv`. "
                    "Before training the model:\n"
                    "1. For the training set, separate the target column as y.\n"
                    "2. Remove the target column and any non-numeric columns (e.g., String-type columns) that cannot be used in model training from the training set.\n"
                    "Before making predictions:\n"
                    "1. For the test set, remove the same columns that were removed from the training set (except the target column, which is not present in the test set).\n"
                    "2. Ensure consistency between the columns used in training and prediction.\n"
                    "Due to computational resource limitations, you are allowed to train a maximum of **three** models")
        return ""

    def get_current_agent(self) -> str:
        return self.agents[self.current_step % len(self.agents)]

    def generate_rules(self) -> str:
        if self.rulebook_parameters[self.phase]['status']:
            default_rules = self.rulebook_parameters[self.phase]['default_rules_with_parameters']
            rules = self._format_rules(default_rules)
        else:
            rules = "There is no rule for this stage."
        
        with open(f'{self.restore_dir}/user_rules.txt', 'w') as f:
            f.write(rules)
        return rules

    def _format_rules(self, default_rules: Dict[str, List[Any]]) -> str:
        formatted_rules = []
        for key, values in default_rules.items():
            if sum(values[0]) == 0:
                continue
            formatted_rules.append(f"If you need to {key}, please follow the following rules:")
            formatted_rules.extend([f"- {rule[0].format(placeholder=rule[1])}" for i, rule in enumerate(values[1:]) if values[0][i] == 1])
            formatted_rules.append("")
        return "\n".join(formatted_rules)

    # 创建当前State的目录
    def make_dir(self) -> None:
        path_to_dir = f'{self.competition_dir}/{self.dir_name}'
        os.makedirs(path_to_dir, exist_ok=True)
        if 'eda' in self.dir_name:
            os.makedirs(f'{path_to_dir}/images', exist_ok=True)
        self.restore_dir = path_to_dir

    def get_previous_phase(self, type: str = "code") -> Any:
        phases = load_config(f'{PREFIX_MULTI_AGENTS}/config.json')['phases']
        current_phase_index = phases.index(self.phase)
        
        if type == 'code':
            return self._get_previous_phase_for_code(phases, current_phase_index)
        elif type == 'plan':
            return phases[:current_phase_index]
        elif type == 'report':
            return phases[current_phase_index - 1]
        else:
            raise ValueError(f"Unknown type: {type}")

    def _get_previous_phase_for_code(self, phases: List[str], current_phase_index: int) -> str:
        if self.phase == 'Data Cleaning':
            return 'Understand Background'
        elif self.phase == 'Feature Engineering':
            return 'Data Cleaning'
        else:
            return phases[current_phase_index - 1]

    def get_next_phase(self) -> Optional[str]:
        phases = load_config(f'{PREFIX_MULTI_AGENTS}/config.json')['phases']
        current_phase_index = phases.index(self.phase)
        return phases[current_phase_index + 1] if current_phase_index < len(phases) - 1 else None

    def update_memory(self, memory: Dict[str, Any]) -> None:
        print(f"{self.agents[self.current_step]} updates internal memory in Phase: {self.phase}.")
        self.memory[-1].update(memory)

    def restore_memory(self) -> None:
        with open(f'{self.restore_dir}/memory.json', 'w') as f:
            json.dump(self.memory, f, indent=4)
        print(f"Memory in Phase: {self.phase} is restored.")

    def restore_report(self) -> None:
        report = self.memory[-1].get('summarizer', {}).get('report', '')
        if len(report) > 0:
            with open(f'{self.restore_dir}/report.txt', 'w') as f:
                f.write(report)
            print(f"Report in Phase: {self.phase} is restored.")
        else:
            print(f"No report in Phase: {self.phase} to restore.")

    def next_step(self) -> None:
        self.current_step += 1

    def set_score(self) -> None:
        final_score = self.memory[-1]['reviewer']['score']
        self.score = sum(float(score) for score in final_score.values())/len(final_score)

    def check_finished(self) -> bool:
        self.finished = self.current_step == len(self.agents)
        return self.finished