import os
from api_handler import APIHandler, APISettings
import base64
import subprocess
import pdb
import re
import shutil
import json

DIR = os.path.dirname(os.path.abspath(__file__))
PREFIX_STRONG_BASELINE = f'{DIR}/strong_baseline/competition'
PREFIX_WEAK_BASELINE = f'{DIR}/weak_baseline/competition'
PREFIX_MULTI_AGENTS = f'{DIR}/multi_agents'
SEPERATOR_TEMPLATE = '-----------------------------------{step_name}-----------------------------------'

def load_config(file_path: str):
    assert file_path.endswith('json'), "The configuration file should be in JSON format."
    with open(file_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def read_file(file_path: str):
    """
    Read the content of a file and return it as a string.
    """
    if file_path.endswith('txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    if file_path.endswith('csv'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    
def multi_chat(model: APIHandler, prompt, history=None, max_tokens=4096):
    """
    Multi-round chat with the assistant.
    """
    if history is None:
        history = []

    messages = history + [{'role': 'user', 'content': prompt}]

    settings = APISettings(max_tokens=max_tokens)
    reply = model.get_output(messages=messages, settings=settings)
    history.append({'role': 'user', 'content': prompt})
    history.append({'role': 'assistant', 'content': reply})
    
    return reply, history

def read_image(prompt, image_path):
    """
    Read the image and return the response.
    """
    # encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    # Getting the base64 string
    base64_image = encode_image(image_path)
    api_handler = APIHandler('gpt-4o')
    messages=[
        {"role": "system", "content": "You are a professional data analyst."},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        }
    ]
    settings = APISettings(max_tokens=4096)
    reply = api_handler.get_output(messages=messages, settings=settings, response_type='image')
    return reply

def extract_and_run_code(competition, path_to_competition_step):
    """
    Extract code from a markdown file, write it to a python file, run the python file, and capture the output.
    """
    step = path_to_competition_step.split('/')[-1]

    print(f"Extracting code from markdown file and running it for the competition '{competition}' and step '{step}'.")

    # Define paths
    txt_file_path = f'{path_to_competition_step}/{step}_code.txt'
    output_code_path = f'{path_to_competition_step}/{step}_generated.py'
    output_result_path = f'{path_to_competition_step}/{step}_output.txt'

    # Read the content of the file
    content = read_file(txt_file_path)
    
    # Extract code from the file
    pattern = r"```python(.*?)```\n"
    matches = re.findall(pattern, content, re.DOTALL)
    code_lines = []
    # pdb.set_trace()
    for match in matches:
        code_lines.extend(match.split('\n'))

    # pdb.set_trace()    
    # Enclose the code in a function
    code_lines = [f"    {line}\n" for line in code_lines]
    code_lines = ["def generated_code_function():\n"] + code_lines
    
    # Write the code to a python file
    # pdb.set_trace()
    with open(output_code_path, 'w') as file:
        file.write(''.join(code_lines))
        file.write('\n\nif __name__ == "__main__":\n    generated_code_function()')
    
    # Delete all files in the images directory
    if 'eda' in step:
        images_dir = f'{path_to_competition_step}/images/'
        for filename in os.listdir(images_dir):
            file_path = os.path.join(images_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)  # Delete file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Delete directory
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        print(f"All files in directory '{images_dir}' have been deleted successfully.")

    # Run the code and capture the output
    result = subprocess.run(['python3', output_code_path], capture_output=True, text=True)
    
    # Check if there was an error during code execution
    error_flag = False
    output_error_path = f'{path_to_competition_step}/{step}_error.txt'
    if result.stderr:
        print("Error encountered during code execution.")
        with open(output_error_path, 'w') as file:
            file.write(result.stderr)
        error_flag = True
    else:
        print("Code executed successfully without errors.")
        try:
            # Delete error file.
            os.remove(output_error_path)
            print(f"File '{output_error_path}' has been deleted successfully.")
        except FileNotFoundError:
            print(f"File '{output_error_path}' doesn't exist, you don't need to delete it.")

    # Write the output to a file
    with open(output_result_path, 'w') as file:
        file.write(result.stdout)

    return error_flag

if __name__ == '__main__':
    print(DIR)
    print(PREFIX_STRONG_BASELINE)
    print(PREFIX_WEAK_BASELINE)