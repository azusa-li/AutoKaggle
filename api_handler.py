import os
import time
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

import openai
import httpx

# Constants
DIR = os.path.dirname(os.path.abspath(__file__))
API_KEY_FILE = f'{DIR}/api_key.txt'
MAX_ATTEMPTS = 5
RETRY_DELAY = 30

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# 配置 httpx 的日志级别
logging.getLogger("httpx").setLevel(logging.WARNING)

# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

@dataclass
class APISettings:
    max_tokens: int
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None

    @property
    def timeout(self) -> int:
        return (self.max_tokens // 1000 + 1) * 30

def load_api_config() -> Tuple[str, Optional[str]]:
    try:
        with open(API_KEY_FILE, 'r') as f:
            api_config = f.readlines()
        api_key = api_config[0].strip()
        base_url = api_config[1].strip() if len(api_config) > 1 else None
        return api_key, base_url
    except FileNotFoundError:
        raise ValueError(f"API key file not found: {API_KEY_FILE}")

def generate_response(client: openai.OpenAI, engine: str, messages: List[Dict[str, str]], 
                      settings: APISettings, response_type: str) -> Any:
    logger.info(f"Generating response for engine: {engine}")
    start_time = time.time()
    
    try:
        if response_type == 'text':
            response = client.chat.completions.create(
                messages=messages,
                model=engine,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                top_p=settings.top_p,
                frequency_penalty=settings.frequency_penalty,
                presence_penalty=settings.presence_penalty,
                stop=settings.stop,
                timeout=settings.timeout,
            )
        elif response_type == 'image':
            response = client.chat.completions.create(
                messages=messages,
                model=engine,
                temperature=settings.temperature,
                timeout=settings.timeout,
            )
        else:
            raise ValueError(f"Unsupported response type: {response_type}")
    except Exception as e:
        logging.error(f"Error during API call: {e}")
        raise
    
    logger.info(f"Response generated in {time.time() - start_time:.2f} seconds")
    return response

class APIHandler:
    def __init__(self, model: str):
        self.engine = model
        self.api_key, self.base_url = load_api_config()
        self.client = openai.OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url, 
            http_client=httpx.Client(verify=False)
        )

    def _save_long_message(self, messages: List[Dict[str, str]]):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"long_message_{timestamp}.txt"
        with open(f'multi_agents/competition/{filename}', 'w', encoding='utf-8') as f:
            for message in messages:
                f.write(f"Role: {message['role']}\n")
                f.write(f"Content: {message['content']}\n\n")
        logger.info(f"Long message saved to {filename}")

    def _truncate_messages(self, messages: List[Dict[str, str]], max_length: int = 1000000) -> List[Dict[str, str]]:
        """Truncate messages to fit within the maximum length."""
        truncated = []
        current_length = 0
        for message in reversed(messages):
            message_length = len(message['content'])
            if current_length + message_length <= max_length:
                truncated.insert(0, message)
                current_length += message_length
            else:
                available_length = max_length - current_length
                if available_length > 100:  # Ensure we have enough space for a meaningful truncation
                    truncated_content = message['content'][:available_length-3] + "..."
                    truncated.insert(0, {"role": message['role'], "content": truncated_content})
                break
        return truncated

    def get_output(self, messages: List[Dict[str, str]], settings: APISettings, response_type: str = 'text') -> str:
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = generate_response(self.client, self.engine, messages, settings, response_type)
                if response.choices and response.choices[0].message and hasattr(response.choices[0].message, 'content'):
                    return response.choices[0].message.content
                else:
                    return "Error: Wrong response format."
            except openai.BadRequestError as error:
                error_message = str(error)
                if "string too long" in error_message or "maximum context length" in error_message:
                    logging.error(f"Message too long. Attempting to truncate.")
                    self._save_long_message(messages)
                    messages = self._truncate_messages(messages)
                    continue
                else:
                    logging.error(f'Attempt {attempt + 1} of {MAX_ATTEMPTS} failed with error: {error}')
            except (TimeoutError, openai.APIError, openai.APIConnectionError, openai.RateLimitError) as error:
                logging.error(f'Attempt {attempt + 1} of {MAX_ATTEMPTS} failed with error: {error}')
            
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                return f"Error: Max attempts reached. Last error: {error}"

if __name__ == '__main__':
    handler = APIHandler('gpt-4o')
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How are you today?"}
    ]
    settings = APISettings(max_tokens=50)
    output_text = handler.get_output(messages=messages, settings=settings)
    print(output_text)