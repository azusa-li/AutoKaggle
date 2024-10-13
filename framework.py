import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multi_agents.state import State
from multi_agents.sop import SOP
from utils import PREFIX_MULTI_AGENTS, load_config
import pdb
import argparse
import logging

import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SOP for a competition.')
    parser.add_argument('--competition', type=str, default='titanic', help='Competition name')
    args = parser.parse_args()
    competition = args.competition

    sop = SOP(competition)
    start_state = State(phase="Understand Background", competition=competition)
    # start_state = State(phase="Preliminary Exploratory Data Analysis", competition=competition)
    # start_state = State(phase="Data Cleaning", competition=competition)
    # start_state = State(phase="In-depth Exploratory Data Analysis", competition=competition)
    # start_state = State(phase="Feature Engineering", competition=competition)
    # start_state = State(phase="Model Building, Validation, and Prediction", competition=competition)
    start_message = ""
    new_state = start_state

    # 配置根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(f"{PREFIX_MULTI_AGENTS}/competition/{competition}/{competition}.log")
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 将处理器添加到根logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    root_logger.info(f"Start SOP for competition: {competition}")
    while True:
        current_state = new_state
        state_info, new_state = sop.step(state=current_state)
        if state_info == 'Fail':
            logging.error("Failed to update state.")
            exit()
        if state_info == 'Complete':
            logging.info(f"Competition {competition} SOP is completed.")
            break  
