import chainlit as cl
import asyncio
from ta_model import ta_model

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

    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)
    
    await response_message.update()

if __name__ == "__main__":
    cl.main()
