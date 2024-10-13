from strong_baseline.understand_background import understand_background
from strong_baseline.preliminary_eda import preliminary_eda, get_insight_pre_eda
from strong_baseline.data_cleaning import data_cleaning
from strong_baseline.deep_eda import deep_eda, get_insight_deep_eda
from strong_baseline.feature_engineering import feature_engineering
from strong_baseline.model_build_predict import model_build_predict
from utils import SEPERATOR_TEMPLATE, PREFIX_STRONG_BASELINE
import os
import time
import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the strong baseline for the data science competition.')
    parser.add_argument('--competition', type=str, default='titanic', help='The competition name.')
    parser.add_argument('--start_times', type=int, default=0, help='The start times of the pipeline.')
    parser.add_argument('--end_times', type=int, default=10, help='The end times of the pipeline.')
    parser.add_argument('--store_history', type=bool, default=True, help='Whether to store the chat history.')
    args = parser.parse_args()

    competition = args.competition
    store_history = args.store_history
    run_times = args.end_times
    run_range = range(args.start_times, args.end_times)
    success_times = 0

    start_time = time.time()   
    for i in tqdm.tqdm(run_range):
        try:
            print(f"Run {i+1} of {run_times}.")

            path_to_competition = f'{PREFIX_STRONG_BASELINE}/{competition}'
            assert os.path.exists(path_to_competition), f"Path {path_to_competition} does not exist."
            path_to_overview = f'{path_to_competition}/overview.txt'
            assert os.path.exists(path_to_overview), f"Path {path_to_overview} does not exist."

            path_to_competition_times = f'{path_to_competition}/submission_{i}'
            if not os.path.exists(path_to_competition_times):
                os.mkdir(path_to_competition_times)
            dir_list = ['pre_eda', 'data_cleaning', 'deep_eda', 'feature_engineering', 'model_build_predict']
            for dir_name in dir_list:
                path_to_dir = f'{path_to_competition_times}/{dir_name}'
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)
                if 'eda' in dir_name:
                    if not os.path.exists(f'{path_to_dir}/images'):
                        os.makedirs(f'{path_to_dir}/images')

            print(f"Starting the pipeline for {competition}.")
            print(SEPERATOR_TEMPLATE.format(step_name='Step 1: Background Understanding'))
            competition_info = understand_background(competition, i)

            print(SEPERATOR_TEMPLATE.format(step_name='Step 2: Preliminary EDA'))
            pre_eda_flag = preliminary_eda(competition, competition_info, i, store_history)

            if not pre_eda_flag:
                # exit()
                continue
            pre_eda_insight = get_insight_pre_eda(competition, i)

            print(SEPERATOR_TEMPLATE.format(step_name='Step 3: Data Cleaning'))
            data_cleaning_flag = data_cleaning(competition, competition_info, i, store_history)
            if not data_cleaning_flag:
                # exit()
                continue

            print(SEPERATOR_TEMPLATE.format(step_name='Step 4: In-depth EDA'))
            deep_eda_flag = deep_eda(competition, competition_info, i, store_history)
            if not deep_eda_flag:
                # exit()
                continue 
            deep_eda_insight = get_insight_deep_eda(competition, i)

            print(SEPERATOR_TEMPLATE.format(step_name='Step 5: Feature Engineering'))
            feature_engineering_flag = feature_engineering(competition, competition_info, i, store_history)
            if not feature_engineering_flag:
                # exit()
                continue

            print(SEPERATOR_TEMPLATE.format(step_name='Step 6: Model Building, Validation, Prediction'))
            model_build_predict_flag = model_build_predict(competition, competition_info, i, store_history)
            if not model_build_predict_flag:
                # exit()
                continue

            print("Pipeline completed successfully.")

            success_times += 1
        except Exception as e:
            print(f"Error: {e}")
            print("Submission {i} failed.")
            continue

    end_time = time.time()
    print(f"Average time elapsed for each run: {(end_time-start_time)/run_times} seconds.")
    print(f"Pipeline run {run_times} times, success {success_times} times.")