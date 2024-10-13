from typing import Dict, Any, Tuple
import json
import re
import logging
import sys 
import os
import copy
import subprocess
import shutil
import pdb

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
from prompts.prompt_developer import *
from tools import *

class Developer(Agent):
    def __init__(self, model: str, type: str):  
        super().__init__(
            role="developer",
            description="You are skilled at writing and implementing code according to plan.",
            model=model,
            type=type
        )
        self.all_error_messages = []

    def _is_previous_code(self, state: State) -> Tuple[bool, str, str]:
        previous_phase = state.get_previous_phase()
        previous_dir_name = state.phase_to_directory[previous_phase]
        path_to_previous_code = f'{state.competition_dir}/{previous_dir_name}/{previous_dir_name}_code.py'
        path_to_previous_run_code = f'{state.competition_dir}/{previous_dir_name}/{previous_dir_name}_run_code.py'
        path_to_last_phase_code = f'{state.competition_dir}/{previous_dir_name}/single_phase_code.txt'
        return os.path.exists(path_to_previous_code), path_to_previous_code, path_to_previous_run_code, path_to_last_phase_code

    def _delete_output_in_code(self, state: State, previous_code) -> str:
        previous_run_code = copy.deepcopy(previous_code) # 深拷贝 防止修改原始数据
        keywords = ('sns.', '.plot', '.hist', '.plt')

        # 第一次扫描：识别 for 循环，整段替换
        for_loop_list = []
        in_for_loop = False
        # pdb.set_trace()
        for i, line in enumerate(previous_run_code):
            if line.startswith('    for'):
                tmp_loop = []
                indent = line[:len(line) - len(line.lstrip())]  # 获取缩进部分
                tmp_loop.append(i) # 记录for循环的起始行
                # print(i)
                in_for_loop = True
            elif in_for_loop and line.startswith(indent) and not line.startswith('    '+indent) and len(line.strip()) > 0:
                tmp_loop.append(i) # 记录for循环的终止行
                # print(i)
                in_for_loop = False
                for_loop_list.append(tmp_loop)

        # if len(for_loop_list) > 0:
        #     logger.info(f"for_loop_list: {for_loop_list}")

        # 逆序替换 for 循环为 '    pass'
        for start, end in for_loop_list[::-1]:
            loop_code = "\n".join(previous_run_code[start:end])
            if any(keyword in loop_code for keyword in keywords):  # 如果 for 循环中包含关键字
                previous_run_code[start:end] = ['    pass\n']  # 替换相应行

        # 第二次扫描：替换 print 和 plt.show / plt.save 行，保留缩进
        # pdb.set_trace()
        start_signs = ('print', 'plt')
        for i, line in enumerate(previous_run_code):
            stripped_line = line.lstrip()
            if stripped_line.startswith(start_signs) or any(keyword in stripped_line for keyword in keywords):
                indent = line[:len(line) - len(stripped_line)]  # 获取缩进部分
                previous_run_code[i] = indent + 'pass\n'
        
        # 第三次扫描：合并连续的 pass 行
        new_code = []
        pass_found = False
        
        for line in previous_run_code:
            if line.strip() == 'pass':
                if not pass_found:  # 第一次遇到 pass
                    new_code.append(line)
                    pass_found = True
            else:
                new_code.append(line)
                pass_found = False
        
        return new_code

    def _generate_code_file(self, state: State, raw_reply) -> Tuple[bool, str, str]:
        is_previous_code, path_to_previous_code, _, _ = self._is_previous_code(state)
        if is_previous_code:
            with open(path_to_previous_code, 'r', encoding='utf-8') as f_1:
                previous_code = f_1.readlines()
                previous_code = previous_code[:-2] # 删除最后两行
                previous_code = previous_code[9:] # 删除前9行
            previous_run_code = self._delete_output_in_code(state, previous_code) # 删除output
        else:
            previous_code = []
            previous_run_code = []
        # code with output
        path_to_code = f'{state.restore_dir}/{state.dir_name}_code.py'
        path_to_run_code = f'{state.restore_dir}/{state.dir_name}_run_code.py'

        no_code_flag = False
        # Extract code from the file
        pattern = r"```python(.*?)```"
        matches = re.findall(pattern, raw_reply, re.DOTALL)
        code_lines = []
        # pdb.set_trace()
        for match in matches:
            code_lines.extend(match.split('\n'))
        
        # Enclose the code in a function
        code_lines = [f"    {line}\n" for line in code_lines]
        if len(code_lines) == 0:
            logging.error("No code found in the reply.")
            # pdb.set_trace()
            no_code_flag = True
            return True, "no code", "no code"
        
        with open(f'{state.restore_dir}/single_phase_code.txt', 'w') as f: # 保存单步的code
            f.write("\n".join(matches))

        prefix_in_code_file = [line + '\n' for line in PREFIX_IN_CODE_FILE.split('\n')]        
        code_with_output_lines = prefix_in_code_file + previous_code + code_lines
        run_code_lines = prefix_in_code_file + previous_run_code + code_lines

        # Write the code to a python file
        with open(path_to_code, 'w', encoding='utf-8') as f_w:
            f_w.write("".join(code_with_output_lines))
            f_w.write('\n\nif __name__ == "__main__":\n    generated_code_function()')
        # Write the run code to a python file
        with open(path_to_run_code, 'w', encoding='utf-8') as f_w:
            f_w.write("".join(run_code_lines))
            f_w.write('\n\nif __name__ == "__main__":\n    generated_code_function()')
        
        return no_code_flag, path_to_code, path_to_run_code

    def _run_code(self, state: State, no_code_flag: bool, path_to_run_code: str) -> str:
        # Delete previous images files
        if 'eda' in state.restore_dir:
            images_dir = f'{state.restore_dir}/images/'
            for filename in os.listdir(images_dir):
                image_path = os.path.join(images_dir, filename)
                try:
                    if os.path.isfile(image_path) or os.path.islink(image_path):
                        os.remove(image_path)  # Delete file
                    elif os.path.isdir(image_path):
                        shutil.rmtree(image_path)  # Delete directory
                except Exception as e:
                    logger.info(f"Failed to delete {image_path}. Reason: {e}")
            logger.info(f"All files in directory '{images_dir}' have been deleted successfully.")

        # Run the code
        timeout_flag = False
        error_flag = False
        path_to_error = f'{state.restore_dir}/{state.dir_name}_error.txt'
        path_to_output = f'{state.restore_dir}/{state.dir_name}_output.txt'

        if no_code_flag:
            with open(path_to_error, 'w') as f:
                f.write("No code found in the reply.")
            with open(path_to_output, 'w') as f:
                f.write("")
            return True

        result = {}
        # timeout
        if 'Analysis' in state.phase:
            timeout = 600
            timeout_info = "Your code is running out of time, please consider resource availability and reduce the number of data analysis plots drawn."
        elif 'Model' in state.phase:
            timeout = 1200
            timeout_info = "Your code is running out of time, please consider resource availability and try fewer models."
        else:
            timeout = 300
            timeout_info = "Your code is running out of time, please consider resource availability or other factors."
        try:
            result = subprocess.run(['python3', '-W', 'ignore', path_to_run_code], 
                                    capture_output=True, text=True, timeout=timeout, 
                                    preexec_fn=os.setsid)
        except subprocess.TimeoutExpired:
            logger.info("Code execution timed out.")
            self.all_error_messages.append(timeout_info)
            with open(path_to_error, 'w') as f:
                f.write(timeout_info)
            with open(path_to_output, 'w') as f:
                f.write("")
            error_flag = True
        except subprocess.CalledProcessError as e:
            if e.returncode < 0:
                # Negative return codes usually indicate termination by a signal
                logger.info(f"Process was killed by signal {-e.returncode}")
                error_message = f"Process was terminated by the operating system (signal {-e.returncode})"
            else:
                logger.info(f"Process exited with non-zero status: {e.returncode}")
                error_message = f"Process exited with status {e.returncode}: {e.stderr}"
            self.all_error_messages.append(error_message)
            with open(path_to_error, 'w') as f:
                f.write(error_message+"\nI suggest you use logging module to record the information, which can help you find the reason why operation system terminated your process.\nOne possible reason is When working with dataframe-type data, you perform multiplication operations on different types of data.")
            error_flag = True
        else:
            if result.returncode != 0:
                logger.info(f"Process exited with non-zero status: {result.returncode}")
                error_message = f"Process exited with status {result.returncode}: {result.stderr}"
                self.all_error_messages.append(error_message)
                with open(path_to_error, 'w') as f:
                    f.write(error_message)
                error_flag = True
            else:
                logger.info("Code executed successfully without errors.")
                self.all_error_messages = []
                try:
                    os.remove(path_to_error)
                    logger.info(f"File '{path_to_error}' has been deleted successfully.")
                except FileNotFoundError:
                    logger.info(f"File '{path_to_error}' doesn't exist, you don't need to delete it.")

        # Write the output to a file
        if result and hasattr(result, 'stdout'):
            with open(path_to_output, 'w') as f:
                f.write(result.stdout)
        else:
            with open(path_to_output, 'w') as f:
                f.write("")

        # 在这里如果error信息太长 可以读取做去重
        return error_flag

    def _conduct_unit_test(self, state: State) -> None:
        test_tool = TestTool(memory=None, model='gpt-4o', type='api')
        not_pass_flag = False
        not_pass_tests = test_tool.execute_tests(state) # [(test1_number, test1_information), ...] 全通过返回[]
        logger.info(f"There are {len(not_pass_tests)} not pass tests.")
        not_pass_information = ""
        if not_pass_tests:
            not_pass_flag = True
            logger.info("Unit tests failed.")
            for test_flag, test_number, test_information in not_pass_tests:
                logger.info(f"Test {test_number}: {test_information}")
                not_pass_information += f"\n## TEST CASE NUMBER {test_number} ##\n{test_information}"
            # print("Not pass information: ", not_pass_information)
        else:
            not_pass_information = ""
            logger.info("All unit tests passed.")
            try:
                # Delete error file.
                path_to_not_pass_info = f'{state.restore_dir}/{state.dir_name}_not_pass_information.txt'
                os.remove(path_to_not_pass_info)
                logger.info(f"File '{path_to_not_pass_info}' has been deleted successfully.")
            except FileNotFoundError:
                logger.info(f"File '{path_to_not_pass_info}' doesn't exist, you don't need to delete it.")
        return not_pass_flag, not_pass_information

    def _debug_code(self, state: State, error_flag: bool, not_pass_flag: bool, not_pass_information: str, raw_reply: str) -> str:
        # prepare debug information, and then debug
        is_previous_code, path_to_previous_code, _, path_to_last_phase_code = self._is_previous_code(state)
        if is_previous_code:
            previous_code = read_file(path_to_last_phase_code)
        else:
            previous_code = "There is no code file in the previous phase."
        # Extract code from the file
        pattern = r"```python(.*?)```"
        matches = re.findall(pattern, raw_reply, re.DOTALL)
        code_lines = []
        for match in matches:
            code_lines.extend(match.split('\n'))
        wrong_code = "\n".join(code_lines) # 这里的code是有错误的
        # 读取error和output
        path_to_error = f'{state.restore_dir}/{state.dir_name}_error.txt'
        path_to_output = f'{state.restore_dir}/{state.dir_name}_output.txt'
        if os.path.exists(path_to_error):
            error_messages = read_file(path_to_error)
        else:
            error_messages = "There is no error message in the previous phase."
        if state.phase in ['Feature Engineering', 'Model Building, Validation, and Prediction']:
            output_messages = read_file(path_to_output)
        else:
            output_messages = ""

        logger.info("Start debugging the code.")
        debug_tool = DebugTool(model='gpt-4o', type='api')
        if error_flag:
            tools, tool_names = self._get_tools(state)
            reply, single_round_debug_history = debug_tool.debug_code_with_error(state, copy.deepcopy(self.all_error_messages), output_messages, previous_code, wrong_code, error_messages, tools, tool_names)
        elif not_pass_flag:
            reply, single_round_debug_history = debug_tool.debug_code_with_no_pass_test(state, output_messages, previous_code, wrong_code, not_pass_information)

        return reply, single_round_debug_history

    def _generate_prompt_round0(self, state: State) -> str:
        prompt_round1 = ""
        # 读取上一个阶段的code
        is_previous_code, path_to_previous_code, _, path_to_last_phase_code = self._is_previous_code(state)
        if is_previous_code:
            previous_code = read_file(path_to_last_phase_code)
        else:
            previous_code = "There is no code file in the previous phase."
        prompt_round1 += f"\n#############\n# CODE FROM PREVIOUS PHASE #\n{previous_code}"
        prompt_round1 += self._read_data(state, num_lines=1)
        tools, tool_names = self._get_tools(state)
        if len(tool_names) > 0:
            prompt_round1 += PROMPT_AVAILABLE_TOOLS.format(tools=tools, tool_names=tool_names)
        else:
            prompt_round1 += "# AVAILABLE TOOLS #\nThere is no pre-defined tools in this phase. You can use the functions from public libraries such as Pandas, NumPy, Scikit-learn, etc.\n"

        return prompt_round1

    def _execute(self, state: State, role_prompt: str) -> Dict[str, Any]:
        # 实现开发和调试功能
        history = []
        debug_history = []
        test_history = []
        self.all_error_messages = []
        round = 0
        test_round = 0
        max_tries = 5
        error_flag = False
        not_pass_flag = False
        no_code_flag = False
        retry_flag = False
        not_pass_information = ""
        restore_path = state.restore_dir
        competition_path = state.competition_dir
        task = PROMPT_DEVELOPER_TASK
        constraints = PROMPT_DEVELOPER_CONSTRAINTS.format(restore_path=restore_path, competition_path=competition_path, phase_name=state.phase)
        with open(f'{state.competition_dir}/competition_info.txt', 'r') as f:
            competition_info = f.read()
        plan = state.memory[-1]["planner"]["plan"] # 这里plan的格式是markdown

        if len(state.memory) == 1: # 如果之前没有memory，说明是第一次执行
            history.append({"role": "system", "content": f"{role_prompt}{self.description}\n when you are writing code, you should follow the plan and the following constraints.\n{constraints}"})
        else:
            self.description = "You are skilled at writing and implementing code according to plan." \
                            "You have advanced reasoning abilities and can improve your answers through reflection."
            experience_with_suggestion = self._gather_experience_with_suggestion(state)
            history.append({"role": "system", "content": f"{role_prompt}{self.description}\n when you are writing code, you should follow the plan and the following constraints.\n{constraints}"})

        while round <= max_tries:
            if round == 0 or retry_flag or no_code_flag:
                if len(state.memory) == 1:
                    input = PROMPT_DEVELOPER.format(phases_in_context=state.context, phase_name=state.phase, state_info=state.get_state_info(), competition_info=competition_info, plan=plan,task=task)
                    if retry_flag or no_code_flag: # Reset history to initial system message if retrying or no code was generated
                        history = history[:1]
                    raw_reply, history = self.llm.generate(input, history, max_tokens=4096)
                    prompt_round0 = self._generate_prompt_round0(state)
                    input = prompt_round0
                    raw_reply, history = self.llm.generate(input, history, max_tokens=4096)
                else:
                    input = PROMPT_DEVELOPER_WITH_EXPERIENCE_ROUND0_0.format(phases_in_context=state.context, phase_name=state.phase, state_info=state.get_state_info(), competition_info=competition_info, plan=plan, task=task, experience_with_suggestion=experience_with_suggestion)
                    raw_reply, history = self.llm.generate(input, history, max_tokens=4096)
                    prompt_round0 = self._generate_prompt_round0(state)
                    input = prompt_round0
                    raw_reply, history = self.llm.generate(input, history, max_tokens=4096)
                    with open(f'{state.restore_dir}/{self.role}_first_mid_reply.txt', 'w') as f:
                        f.write(raw_reply)
                    input = PROMPT_DEVELOPER_WITH_EXPERIENCE_ROUND0_2
                    raw_reply, history = self.llm.generate(input, history, max_tokens=4096)
                if retry_flag:
                    self.all_error_messages = [] # retry后清空error messages
                    logger.info("The developer asks for help when debugging the code. Regenerating the code.")
                    with open(f'{state.restore_dir}/{self.role}_retry_reply.txt', 'w') as f:
                        f.write(raw_reply)
                elif no_code_flag:
                    self.all_error_messages = [] # 清空error messages
                    logger.info("Last reply has no code. Regenerating the code.")
                    with open(f'{state.restore_dir}/{self.role}_no_code_reply.txt', 'w') as f:
                        f.write(raw_reply)
                else:
                    with open(f'{state.restore_dir}/{self.role}_first_reply.txt', 'w') as f:
                        f.write(raw_reply)
                retry_flag = False
            elif round >= 1:
                if error_flag and round < max_tries: # 如果最后一轮还有错，就不再debug
                    # 每一轮单独debug
                    raw_reply, single_round_debug_history = self._debug_code(state, error_flag, not_pass_flag, not_pass_information, raw_reply) # 这里我先把develop和debug解耦 后续便于加上retrieve history然后debug
                    debug_history.append(single_round_debug_history) # 没必要深拷贝
                    if raw_reply == "HELP":
                        logger.info("The developer asks for help when debugging the code. Regenerating the code.")
                        retry_flag = True
                elif not error_flag: # 如果没有错误
                    # 进行unit test 如果根据unit test修改后的code存在问题 
                    while test_round < 2*max_tries and not error_flag:
                        logger.info(f"Start the {test_round+1}-th unit test.")
                        not_pass_flag, not_pass_information = self._conduct_unit_test(state)
                        if not_pass_flag: # 如果unit test不通过 进行debug test 假设run code又出错 就要重新debug code
                            raw_reply, single_round_test_history = self._debug_code(state, error_flag, not_pass_flag, not_pass_information, raw_reply)
                            test_history.append(single_round_test_history)
                            no_code_flag, _, path_to_run_code = self._generate_code_file(state, raw_reply) # 这里后面再重复一遍生成文件和运行代码也没事 因为这里不涉及llm生成代码 raw_reply是一样的
                            error_flag = self._run_code(state, no_code_flag, path_to_run_code)
                        else:
                            break
                        test_round += 1
                    if not not_pass_flag or test_round == max_tries:
                        break
                else:
                    break
            logger.info(f"The {round+1}-th try.")
            if retry_flag:
                round -= 1
            else:
                no_code_flag, _, path_to_run_code = self._generate_code_file(state, raw_reply)
                error_flag = self._run_code(state, no_code_flag, path_to_run_code)
            round += 1

        # 保存history
        with open(f'{state.restore_dir}/{self.role}_history.json', 'w') as f:
            json.dump(history, f, indent=4)
        with open(f'{state.restore_dir}/debug_history.json', 'w') as f:
            json.dump(debug_history, f, indent=4)
        with open(f'{state.restore_dir}/test_history.json', 'w') as f:
            json.dump(test_history, f, indent=4)

        execution_flag = True
        if os.path.exists(f'{state.restore_dir}/{state.dir_name}_error.txt'):
            execution_flag = False
            logger.info(f"State {state.phase} - Agent {self.role} finishes working with error.")
        else:
            if not_pass_flag:
                execution_flag = False
                logger.info(f"State {state.phase} - Agent {self.role} finishes working with not pass tests.")
                with open(f'{state.restore_dir}/{state.dir_name}_not_pass_information.txt', 'w') as f:
                    f.write(not_pass_information)
            else:
                logger.info(f"State {state.phase} - Agent {self.role} finishes working.")

        input_used_in_review = f"   <competition_info>\n{competition_info}\n    </competition_info>\n   <plan>\n{plan}\n    </plan>"
        return {
            self.role: {
                "history": history,
                "role": self.role,
                "description": self.description,
                "task": task,
                "input": input_used_in_review,
                "result": raw_reply,
                "status": execution_flag
            }
        }
    