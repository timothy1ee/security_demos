import os
from dotenv import load_dotenv
from langfuse.openai import AsyncOpenAI
import json
from datetime import datetime
from prompts import ASSESSMENT_PROMPT, SYSTEM_PROMPT, CLASS_CONTEXT
from student_record import read_student_record, write_student_record, format_student_record, parse_student_record
import pandas as pd
import asyncio

# Load environment variables
load_dotenv()

class TAModel:
    def __init__(self):
        self.configurations = {
            "mistral_7B_instruct": {
                "endpoint_url": os.getenv("MISTRAL_7B_INSTRUCT_ENDPOINT"),
                "api_key": os.getenv("RUNPOD_API_KEY"),
                "model": "mistralai/Mistral-7B-Instruct-v0.2"
            },
            "openai_gpt-4": {
                "endpoint_url": os.getenv("OPENAI_ENDPOINT"),
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": "gpt-4o-mini"
            }
        }

        # Choose configuration
        self.config_key = "openai_gpt-4"
        self.config = self.configurations[self.config_key]

        # Initialize the LangFuse AsyncOpenAI client
        self.client = AsyncOpenAI(api_key=self.config["api_key"], base_url=self.config["endpoint_url"])

        self.gen_kwargs = {
            "model": self.config["model"],
            "temperature": 0.2,
            "max_tokens": 1000
        }

        self.ENABLE_SYSTEM_PROMPT = True
        self.ENABLE_CLASS_CONTEXT = True

    def get_latest_user_message(self, message_history):
        for message in reversed(message_history):
            if message['role'] == 'user':
                return message['content']
        return None

    async def assess_message(self, message_history):
        file_path = "student_record.md"
        markdown_content = read_student_record(file_path)
        parsed_record = parse_student_record(markdown_content)

        latest_message = self.get_latest_user_message(message_history)

        filtered_history = [msg for msg in message_history if msg['role'] != 'system']

        history_str = json.dumps(filtered_history, indent=4)
        alerts_str = json.dumps(parsed_record.get("Alerts", []), indent=4)
        knowledge_str = json.dumps(parsed_record.get("Knowledge", {}), indent=4)
        
        current_date = datetime.now().strftime('%Y-%m-%d')

        filled_prompt = ASSESSMENT_PROMPT.format(
            latest_message=latest_message,
            history=history_str,
            existing_alerts=alerts_str,
            existing_knowledge=knowledge_str,
            current_date=current_date
        )
        if self.ENABLE_CLASS_CONTEXT:
            filled_prompt += "\n" + CLASS_CONTEXT

        response = await self.client.chat.completions.create(messages=[{"role": "system", "content": filled_prompt}], **self.gen_kwargs)

        assessment_output = response.choices[0].message.content.strip()

        new_alerts, knowledge_updates = self.parse_assessment_output(assessment_output)

        parsed_record["Alerts"].extend(new_alerts)
        for update in knowledge_updates:
            topic = update["topic"]
            note = update["note"]
            parsed_record["Knowledge"][topic] = note

        updated_content = format_student_record(
            parsed_record["Student Information"],
            parsed_record["Alerts"],
            parsed_record["Knowledge"]
        )
        write_student_record(file_path, updated_content)

    def parse_assessment_output(self, output):
        try:
            parsed_output = json.loads(output)
            new_alerts = parsed_output.get("new_alerts", [])
            knowledge_updates = parsed_output.get("knowledge_updates", [])
            return new_alerts, knowledge_updates
        except json.JSONDecodeError as e:
            print("Failed to parse assessment output:", e)
            return [], []

    def insert_system_message(self, message_history):
        if self.ENABLE_SYSTEM_PROMPT and (not message_history or message_history[0].get("role") != "system"):
            system_prompt_content = SYSTEM_PROMPT
            if self.ENABLE_CLASS_CONTEXT:
                system_prompt_content += "\n" + CLASS_CONTEXT
            message_history.insert(0, {"role": "system", "content": system_prompt_content})

    async def get_response_stream(self, message_history):
        self.insert_system_message(message_history)
        stream = await self.client.chat.completions.create(messages=message_history, stream=True, **self.gen_kwargs)
        async for part in stream:
            if token := part.choices[0].delta.content or "":
                yield token

ta_model = TAModel()

# Function for Giskard compatibility
def model_predict(df: pd.DataFrame) -> list:
    async def process_messages(messages):
        responses = []
        for message in messages:
            message_history = [{"role": "user", "content": message}]
            full_response = ""
            async for token in ta_model.get_response_stream(message_history):
                full_response += token
            responses.append(full_response)
        return responses

    return asyncio.run(process_messages(df["question"]))
