import json
import sys
import os
import time
from tqdm import tqdm

from strong_baseline.prompt import STEPS_IN_CONTEXT_TEMPLATE
from utils import read_file, extract_and_run_code, multi_chat, PREFIX_WEAK_BASELINE

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

PROMPT_WEAK_BASELINE = '''
# CONTEXT #
{steps_in_context}
This is the {i}-th time I ask you to complete the competition.
#############
# OVERVIEW #
{overview}
#############
# TASK #
I will provide you with [OVERVIEW] and all the files you need ({files_str}). Please complete this competition for me. You must write the code and give proper explanations, but do not run the code.
#############
# CONSTRAINTS #
- All the files you need are in the '{prefix}/{competition}' folder, including training data, testing data, and other files.
- Always save the result in the '{prefix}/{competition}/submission_{i}/' folder.
- Always validate and process data types during data handling. Before calculating the correlation matrix, make sure the dataset exclusively contains numeric data. If any non-numeric data is present, handle it appropriately by either removing or processing them.
- Always apply the same modifications to both the training and test sets.
    - Note that the test dataset does not have the target variable.
- Always copy the DataFrame before processing it and use the copy to process.
- Always write some `assert` statements to check the correctness of the code.
#############
# RESPONSE: BLOCK (CODE & EXPLANATION) #
BLOCK 1:
CODE
EXPLANATION
BLOCK 2:
CODE
EXPLANATION
...
#############
# START ANALYSIS #
If you understand, let's work this out in a step-by-step way.
'''

if __name__ == '__main__':
    # competition = 'house_prices'
    # competition = 'titanic'
    competition = 'predict_future_sales'
    competition_name = competition.replace('_', ' ')
    prefix = PREFIX_WEAK_BASELINE

    run_times = 10
    success_times = 0
    start_time = time.time()

    for i in tqdm(range(run_times)):
        path_to_competition_step = f'{prefix}/{competition}/submission_{i}'
        if not os.path.exists(path_to_competition_step):
            os.mkdir(path_to_competition_step)

        with open(f'/mnt/d/PythonProjects/AutoKaggleMaster/competition_to_files.json', 'r') as f:
            data = json.load(f)

        files = data[competition]
        file_str = ', '.join(files)
        # print(file_str)

        steps_in_context = STEPS_IN_CONTEXT_TEMPLATE.format(competition_name=competition_name)
        overview = read_file(f'{prefix}/{competition}/overview.txt')
        prompt = PROMPT_WEAK_BASELINE.format(steps_in_context=steps_in_context, i=i, overview=overview, files_str=file_str, prefix=prefix, competition=competition)
        reply, _ = multi_chat(prompt)

        with open(f'{path_to_competition_step}/prompt.txt', 'w') as f:
            f.write(prompt)

        with open(f'{path_to_competition_step}/submission_{i}_code.txt', 'w') as f:
            f.write(reply)

        fail_flag = extract_and_run_code(competition, path_to_competition_step)

        if not fail_flag and os.path.exists(f'{path_to_competition_step}/submission.csv'):
            success_times += 1

    end_time = time.time()
    print(f"Average time elapsed for each run: {(end_time-start_time)/run_times} seconds.")
    print(f"Weak Baseline run {run_times} times, success {success_times} times.")