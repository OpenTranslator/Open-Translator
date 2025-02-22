# %%
from typing import List, Optional, Dict, Any
import pandas as pd
import google.generativeai as genai
from pydantic import BaseModel, validator
from tqdm import tqdm
import time
import logging
from dataclasses import dataclass
from enum import Enum
import json

from datasets import load_dataset
from abc import ABC, abstractmethod
from enum import Enum

import asyncio
import os
import pathlib
import pandas as pd

import argparse

import argparse
import sys
from datetime import datetime

# %%
# Dataset name
dataset_name = "simplescaling/s1K-1.1"

# Output CSV file name
output_csv_file = "data/s1K-1.1.csv"

api_key = os.getenv('GEMINI_API_KEY')

# %%
# Download and load the dataset
try:
    dataset = load_dataset(dataset_name, split="train")  # Assuming you want the 'train' split
    print(f"Dataset '{dataset_name}' loaded successfully.")
except ConnectionError as e:
    print(f"Error: Could not connect to the internet to download the dataset.\n{e}")
    exit()
except ValueError as e:
    print(f"Error: Could not find the dataset or the specified split.\n{e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the dataset:\n{e}")
    exit()

# %%
class TranslationMode(Enum):
    INITIAL = "initial"
    FIXING = "fixing"
    VALIDATION = "validation"

# %%
@dataclass
class TranslationPrompts:
    INITIAL_PROMPT = """You are an unwavering, highly precise, and literal translator tasked with converting English technical documents to Bangla. Your sole responsibility is to provide an exact Bangla translation of each English language sentence, one sentence at a time, strictly following the rules given below. There is absolutely no summarization, generalization, paraphrasing or modification of the content whatsoever.

Key Rules:
1. Strict Sentence-by-Sentence Translation: You MUST translate the English source material sentence by sentence. Finish translating one sentence completely before moving on to the next.
2. Exclusive Bangla Conversion: ALL translatable English words MUST be converted to their closest, most accurate Bangla equivalent.
3. Preserve Alphanumeric, Equations, and Code: DO NOT translate, alter, or modify alphanumeric strings, mathematical equations, variables, or code snippets. These MUST remain in English (e.g., C1, C2, pF, sqrt, (a + b)^2).
4. Avoid Unnecessary English Terms: Translate even conversational English such as "Okay", "Therefore" to their Bangla equivalents.
5. Maintain Original Structure: The format must precisely mirror the source's format.
6. Grammatical Accuracy: Ensure translated Bangla sentences are grammatically correct and fluent.

Examples:

Input 1:
"Okay, here's an example: The area A of a circle is given by the equation A = πr^2. Therefore, the area is dependent on the radius r. Additionally, we have C1 = 2000 pF and C2 = 3000 pF. Well, this can be shown using E = mc^2"

Output 1:
"ঠিক আছে, এখানে একটি উদাহরণ দেওয়া হলো: বৃত্তের ক্ষেত্রফল A নিম্নলিখিত সমীকরণ দ্বারা নির্ধারিত হয়: A = πr^2। সুতরাং, ক্ষেত্রফল ব্যাসার্ধ r-এর উপর নির্ভরশীল। এছাড়াও, আমাদের কাছে C1 = 2000 pF এবং C2 = 3000 pF। ঠিক আছে, এটি E = mc^2 ব্যবহার করে দেখানো যেতে পারে।"

Input 2:
"If a > b, then f(a) should be greater than f(b). This is because the code will result in g = h + i"

Output 2:
"যদি a > b হয়, তাহলে f(a) অবশ্যই f(b)-এর চেয়ে বড় হওয়া উচিত। এটি এই কারণে যে, কোডের ফলে g = h + i হবে।"

Now, translate the following text into Bangla, following the above rules and examples:
"""

    FIXING_PROMPT = """You are an expert Bangla translator fixing a partially translated text. Identify and translate remaining English words to Bangla while strictly following these rules:

1. Process sentence by sentence
2. Convert ALL remaining English to accurate Bangla equivalents
3. Preserve technical elements exactly as they are: alphanumeric strings, equations, variables, code snippets
4. Maintain original formatting precisely
5. Ensure grammatical fluency in Bangla

Examples:

Input 1:
"আচ্ছা, এখানে একটি উদাহরণ: The area A of a circle is given by the equation A = πr^2। সুতরাং, ক্ষেত্রফল radius r এর উপর নির্ভরশীল। উপরন্তু, আমাদের C1 = 2000 pF এবং C2 = 3000 pF আছে। আচ্ছা, This can be shown using E = mc^2"

Output 1:
"আচ্ছা, এখানে একটি উদাহরণ: বৃত্তের ক্ষেত্রফল A সমীকরণ A = πr^2 দ্বারা প্রদত্ত। সুতরাং, ক্ষেত্রফল ব্যাসার্ধ r এর উপর নির্ভরশীল। উপরন্তু, আমাদের C1 = 2000 pF এবং C2 = 3000 pF আছে। আচ্ছা, এটা E = mc^2 ব্যবহার করে দেখানো যেতে পারে"

Input 2:
"যদি a > b হয়, তাহলে f(a), f(b) থেকে বড় হওয়া উচিত। This is because the code will result in g = h + i"

Output 2:
"যদি a > b হয়, তাহলে f(a), f(b) থেকে বড় হওয়া উচিত। কারণ কোডটি g = h + i তে পরিণত হবে।"

Now, fix any remaining untranslated English text in the following, using the above examples as reference:
"""

    VALIDATION_PROMPT = """You are a translation quality assessment expert. You are given an original English text and its Bangla translation. Your task is to assess the quality of the translation and provide a summary, focusing on:

1. Complete conversion of non-technical English to Bangla
2. Proper preservation of technical terms, equations, and code
3. Grammar and fluency in Bangla
4. Formatting consistency

Example Assessment:

Original: "Okay, here's an example: The area A of a circle is given by the equation A = πr^2. Therefore, the area is dependent on the radius r. Additionally, we have C1 = 2000 pF and C2 = 3000 pF. Well, this can be shown using E = mc^2"
Translation: "আচ্ছা, এখানে একটি উদাহরণ: The area A of a circle is given by the equation A = πr^2। সুতরাং, ক্ষেত্রফল radius r এর উপর নির্ভরশীল। উপরন্তু, আমাদের C1 = 2000 pF এবং C2 = 3000 pF আছে। আচ্ছা, This can be shown using E = mc^2"
Assessment: The translation is partially complete but needs improvement. Issues found:
1. Several English phrases remain untranslated: "The area A of a circle is given by the equation", "radius", "This can be shown using"
2. Technical terms (A, πr^2, C1, C2, pF, E = mc^2) are correctly preserved
3. Translated portions have good grammar
4. Formatting is consistent

Please assess the following translation:

Original: {original}
Translation: {translation}

Provide your assessment:
"""

# %%
class ModelType(Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class BaseModelWrapper(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass

class GeminiWrapper(BaseModelWrapper):
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    async def generate(self, prompt: str) -> str:
        # Gemini's generate_content is synchronous
        try:
            response = self.model.generate_content(prompt)
            if hasattr(response, 'text'):
                return response.text
            return ''.join(part.text for part in response.parts)
        except Exception as e:
            logging.error(f"Gemini generation error: {e}")
            raise

class OpenAIWrapper(BaseModelWrapper):
    def __init__(self, api_key: str):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o"

    async def generate(self, prompt: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI generation error: {e}")
            raise

class AnthropicWrapper(BaseModelWrapper):
    def __init__(self, api_key: str):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = "claude-3.5-sonnet"

    async def generate(self, prompt: str) -> str:
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"Anthropic generation error: {e}")
            raise

# %%
class TranslationResult(BaseModel):
    text: str
    quality_score: Optional[float] = None
    validation_notes: Optional[str] = None
    attempts: List[str] = []

    @validator('quality_score')
    def validate_score(cls, v):
        if v is not None and not (0 <= v <= 1):
            raise ValueError("Quality score must be between 0 and 1")
        return v

# %%
class BatchTranslationResult(BaseModel):
    translations: Dict[str, str]
    validation_notes: Dict[str, str]
    attempts: List[Dict[str, str]] = []

class TranslationAgent:
    def __init__(
        self,
        model_type: ModelType,
        api_key: str = None,
        batch_size: int = 1
    ):
        self.model_type = model_type  # Set model_type first
        self.setup_logging()  # Now setup_logging can access self.model_type
        self.model_wrapper = self.setup_model(model_type, api_key)
        self.prompts = TranslationPrompts()
        self.batch_size = batch_size
        self.logger.info(f"Initialized TranslationAgent with {model_type.value} model")

    def setup_logging(self):
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Create timestamp for log file
        timestamp = time.strftime("%d%b%Y_%I_%M%p")
        
        # Setup file handler with model type and timestamp
        log_file = f'logs/translation_{self.model_type.value}_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),  # Write to file
                logging.StreamHandler()         # Also write to console
            ]
        )
        
        self.logger = logging.getLogger(__name__)

    def setup_model(self, model_type: ModelType, api_key: str = None) -> BaseModelWrapper:
        if model_type == ModelType.GEMINI:
            api_key = api_key or os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("Gemini API key not found")
            return GeminiWrapper(api_key)
        elif model_type == ModelType.OPENAI:
            api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found")
            return OpenAIWrapper(api_key)
        elif model_type == ModelType.ANTHROPIC:
            api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key not found")
            return AnthropicWrapper(api_key)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def get_batch_prompt(self, mode: TranslationMode, columns_data: Dict[str, str] = None, **kwargs) -> str:
        base_prompt = self.prompts.INITIAL_PROMPT if mode == TranslationMode.INITIAL else self.prompts.FIXING_PROMPT
        
        prompt = f"""{base_prompt}

Please translate the following columns and return them in a strict JSON format.
Example format:
{{
    "column1": "translated text 1",
    "column2": "translated text 2"
}}

Columns to translate:
"""
        
        for col, text in columns_data.items():
            prompt += f"\n{col}:\n{text}\n"
        
        prompt += "\nProvide ONLY the JSON response with translations, no additional text."
        return prompt

    def extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from response text, handling common formatting issues."""
        # Try to find JSON-like structure
        try:
            # First try direct JSON parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            try:
                # Look for content between curly braces
                start = response_text.find('{')
                end = response_text.rfind('}')
                if start != -1 and end != -1:
                    json_str = response_text[start:end+1]
                    return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                pass
            
            # If still failing, try to construct JSON from the response
            try:
                constructed_json = {}
                lines = response_text.split('\n')
                current_key = None
                current_value = []
                
                for line in lines:
                    line = line.strip()
                    if ':' in line and not current_key:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            if current_key and current_value:
                                constructed_json[current_key] = ' '.join(current_value).strip()
                            current_key = parts[0].strip().strip('"\'')
                            current_value = [parts[1].strip()]
                    elif line and current_key:
                        current_value.append(line)
                
                if current_key and current_value:
                    constructed_json[current_key] = ' '.join(current_value).strip()
                
                return constructed_json
            except Exception as e:
                self.logger.error(f"Failed to construct JSON from response: {e}")
                return {}

    async def translate_row(self, row: pd.Series, columns: List[str], current_row: int = 0, total_rows: int = 0) -> BatchTranslationResult:
        if not columns:
            return BatchTranslationResult(translations={}, validation_notes={})

        translations = {}
        validation_notes = {}
        attempts = []

        for column in columns:
            if pd.isna(row[column]):
                continue

            try:
                prompt = f"""{self.prompts.INITIAL_PROMPT}

    Please translate the following text from English to Bangla.
    The text is from the column: {column}

    Text to translate:
    {str(row[column])}

    Provide ONLY the translated text, without any additional comments or formatting."""

                start_time = time.time()
                self.logger.info(f"Row {current_row}/{total_rows} - Starting translation for column {column} at {time.strftime('%H:%M:%S')}")

                self.logger.info(f"Row {current_row}/{total_rows} - Column {column}\nPrompt:\n{str(row[column])}")
                
                response = await self.model_wrapper.generate(prompt)
                
                end_time = time.time()
                duration = end_time - start_time
                self.logger.info(f"Row {current_row}/{total_rows} - Translation of column {column} completed in {duration:.2f} seconds")

                # Store the translation
                translations[column] = response.strip()
                validation_notes[column] = "Translation successful"
                self.logger.info(f"Row {current_row}/{total_rows} - Result:\n{str(translations[column])}")

            except Exception as e:
                self.logger.error(f"Row {current_row}/{total_rows} - Translation failed for column {column}: {e}")
                translations[column] = str(row[column])  # Keep original text
                validation_notes[column] = f"Error: {str(e)}"

        attempts.append(translations.copy())

        return BatchTranslationResult(
            translations=translations,
            validation_notes=validation_notes,
            attempts=attempts
        )
    
    async def translate_row2(self, row: pd.Series, columns: List[str]) -> BatchTranslationResult:
        if not columns:
            return BatchTranslationResult(translations={}, validation_notes={})

        columns_data = {col: str(row[col]) for col in columns if pd.notna(row[col])}
        
        try:
            prompt = self.get_batch_prompt(TranslationMode.INITIAL, columns_data=columns_data)
            
            start_time = time.time()
            self.logger.info(f"Starting translation at {time.strftime('%H:%M:%S')}")
            
            # Use the model wrapper's generate method
            response = await self.model_wrapper.generate(prompt)
            
            end_time = time.time()
            duration = end_time - start_time
            self.logger.info(f"Translation completed in {duration:.2f} seconds")
            
            translation_dict = self.extract_json_from_response(response)
            
            if not translation_dict:
                self.logger.warning("Failed to parse translation response")
                return BatchTranslationResult(
                    translations={},
                    validation_notes={"error": "Failed to parse response"},
                    attempts=[]
                )
            
            return BatchTranslationResult(
                translations=translation_dict,
                validation_notes={},
                attempts=[translation_dict]
            )
            
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            return BatchTranslationResult(
                translations={},
                validation_notes={"error": str(e)},
                attempts=[]
            )

    async def translate_dataframe(
        self,
        df: pd.DataFrame,
        columns: List[str],
        save_dir: str = "checkpoints",
        save_interval: int = 10,
        total_rows: Optional[int] = None
    ) -> pd.DataFrame:
        translated_df = df.copy()
        total_rows = len(df) if total_rows is None else total_rows
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Counter for tracking processed rows
        rows_processed = 0
        
        for start_idx in tqdm(range(0, len(df), self.batch_size)):
            batch_df = df.iloc[start_idx:start_idx + self.batch_size]
            
            for idx, row in batch_df.iterrows():
                try:
                    current_row = start_idx + batch_df.index.get_loc(idx) + 1
                    self.logger.info(f"Starting row {current_row}/{total_rows}")
                    
                    translation_result = await self.translate_row(
                        row, 
                        columns, 
                        current_row=current_row,
                        total_rows=total_rows
                    )
                    
                    for column in columns:
                        if column in translation_result.translations:
                            translated_df.at[idx, f"{column}_translated"] = translation_result.translations[column]
                            
                    rows_processed += 1
                    
                    # Save checkpoint every save_interval rows
                    if rows_processed % save_interval == 0:
                        # Generate timestamp
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        
                        # Create filenames with timestamp
                        checkpoint_csv = os.path.join(save_dir, f"translation_checkpoint_{timestamp}.csv")
                        checkpoint_pkl = os.path.join(save_dir, f"translation_checkpoint_{timestamp}.pkl")
                        
                        # Also save a latest version that gets overwritten
                        latest_csv = os.path.join(save_dir, "translation_checkpoint_latest.csv")
                        latest_pkl = os.path.join(save_dir, "translation_checkpoint_latest.pkl")
                        
                        self.logger.info(f"Row {current_row}/{total_rows} - Saving checkpoint after {rows_processed} rows")
                        
                        # Save timestamped versions
                        translated_df.to_csv(checkpoint_csv, index=False)
                        translated_df.to_pickle(checkpoint_pkl)
                        
                        # Save latest versions
                        translated_df.to_csv(latest_csv, index=False)
                        translated_df.to_pickle(latest_pkl)
                        
                except Exception as e:
                    self.logger.error(f"Failed to translate row {current_row}/{total_rows}: {e}")
                    continue
                
        return translated_df
# %%
class TeeStream:
    def __init__(self, stdout, file_stream):
        self.stdout = stdout
        self.file = file_stream
        
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()
        
    def flush(self):
        self.stdout.flush()
        self.file.flush()

async def main(model_type=ModelType.GEMINI):
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%d%b%Y_%I_%M%p")
    log_filename = f"translation_{model_type.value}_{timestamp}.txt"
    
    # Open log file and redirect output
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        # Save original stdout and redirect to both console and file
        original_stdout = sys.stdout
        sys.stdout = TeeStream(original_stdout, log_file)
        
        try:
            print(f"Starting translation with {model_type.value} model at {timestamp}")
            print("-" * 80)
            
            # Initialize the agent
            agent = TranslationAgent(
                model_type=model_type,
                batch_size=1  # Process one row at a time
            )

            # Get project root directory from current file path
            current_file = pathlib.Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            data_dir = str(project_root) + "/translation/src/s1/data"
            data_file = data_dir + "/" + "s1K-1.1.csv"
            df = pd.read_csv(data_file)
            
            # Specify columns to translate
            columns_to_translate = ['solution', 'question', 'cot_type', 'source_type', 'metadata',
                                'deepseek_thinking_trajectory', 'deepseek_attempt']

            # Translate
            translated_df = await agent.translate_dataframe(
                df=df,
                columns=columns_to_translate,
                save_dir="checkpoints",
                save_interval=10,
                total_rows=len(df)  # Pass total number of rows
            )
            
            # Save final results
            translated_df.to_csv("translated_output.csv", index=False)
            print("\nTranslation completed successfully!")
            
        except Exception as e:
            print(f"\nTranslation failed: {e}")
        
        finally:
            # Restore original stdout
            sys.stdout = original_stdout
            print(f"\nLog file saved as: {log_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run translation with specified model')
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['gemini', 'openai', 'anthropic'],
        default='gemini',
        help='Model to use for translation (default: gemini)'
    )
    
    args = parser.parse_args()
    
    # Convert string argument to ModelType enum
    model_map = {
        'gemini': ModelType.GEMINI,
        'openai': ModelType.OPENAI,
        'anthropic': ModelType.ANTHROPIC
    }
    
    selected_model = model_map[args.model]
    
    asyncio.run(main(selected_model))
# %%
