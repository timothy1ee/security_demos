import chainlit as cl
import asyncio
from ta_model import ta_model
from aimon import AnalyzeProd, Application, Model
from prompts import SYSTEM_PROMPT
import os

aimon_config = {
                "instruction_adherence": {"detector_name": "default"}, 
                "toxicity": {"detector_name": "default"},
                # Other supported detectors
                #"hallucination": {"detector_name": "default"},
                #"conciseness": {"detector_name": "default"},
                #"completeness": {"detector_name": "default"},
                #"context_classification": {"detector_name": "default"}
                }

# This is an async implementation 
# Note: it is being revamped to be bundled into the Detect decorator 
analyze_prod = AnalyzeProd(
    # Your application name
    Application("gpt4o_secure_app_oct_2024"), 
    # A model object to track on the UI
    Model("my_gpt40_model", "GPT-4o"), 
    # These are the values returned from your LLM function being monitored
    values_returned=["context", "generated_text", "instructions"],
    # This is the AIMon API key
    api_key=os.getenv("AIMON_API_KEY"),
    config=aimon_config
)

@analyze_prod
def scan_for_compliance(context, generated_text, instructions):
    return context, generated_text, instructions

@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})

    response_message = cl.Message(content="")
    await response_message.send()

    # Stream response from the TA
    async for token in ta_model.get_response_stream(message_history):
        await response_message.stream_token(token)

    # Assess the latest message and update student_record.md
    asyncio.create_task(ta_model.assess_message(message_history))

    # Recap the most recent 5 messages
    context = "<recent_messages>\n"
    user_assistant_messages = [msg for msg in message_history if msg['role'] in ['user', 'assistant']]
    recent_messages = user_assistant_messages[-5:]
    
    for msg in recent_messages:
        role = "user" if msg['role'] == 'user' else "assistant"
        context += f"  <message role='{role}'>\n    {msg['content']}\n  </message>\n"
    
    context += "</recent_messages>"
    _, _, _, aimon_res = scan_for_compliance(context, context, SYSTEM_PROMPT)
    print(aimon_res)
    
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)
    
    await response_message.update()

if __name__ == "__main__":
    cl.main()
