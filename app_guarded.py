import chainlit as cl
import asyncio
from ta_model import ta_model
from llm_guard import scan_prompt, scan_output
from llm_guard.input_scanners import Toxicity, PromptInjection
from llm_guard.output_scanners import NoRefusal
import json

# Initialize input scanners
input_scanners = [Toxicity(), PromptInjection()]

# Initialize output scanners
output_scanners = [NoRefusal()]

@cl.on_message
async def on_message(message: cl.Message):
    try:
        # Scan the input message
        sanitized_prompt, results_valid, results_score = scan_prompt(input_scanners, message.content)
        
        # Format and print the input scan results
        input_scan_results = {
            "original_prompt": message.content,
            "sanitized_prompt": sanitized_prompt,
            "scanners_results": {}
        }
        
        print("Debug: results_valid =", results_valid)
        print("Debug: results_score =", results_score)
        
        for scanner_name, is_valid in results_valid.items():
            print(f"Debug: Processing scanner {scanner_name}")
            input_scan_results["scanners_results"][scanner_name] = {
                "is_valid": is_valid,
                "score": results_score.get(scanner_name, None)
            }
        
        print("Input Scan Results:")
        print(json.dumps(input_scan_results, indent=2))
    except Exception as e:
        print(f"Error during input scanning: {str(e)}")
        sanitized_prompt = message.content  # Use original message if scanning fails
    
    # Continue with the original logic
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": sanitized_prompt})

    response_message = cl.Message(content="")
    await response_message.send()

    # Stream response from the TA
    full_response = ""
    async for token in ta_model.get_response_stream(message_history):
        await response_message.stream_token(token)
        full_response += token

    # Scan the output response
    try:
        _, output_results_valid, output_results_score = scan_output(
            output_scanners, sanitized_prompt, full_response
        )
        
        if 'NoRefusal' in output_results_valid and 'NoRefusal' in output_results_score:
            is_valid = output_results_valid['NoRefusal']
            score = output_results_score['NoRefusal']
            
            if is_valid:
                print("Output Scan Result: No refusal detected. The response is valid.")
            else:
                print("Output Scan Result: WARNING - Potential refusal detected.")
            
            print(f"NoRefusal Score: {score:.2f}")
        else:
            print("Output Scan Result: NoRefusal check results are unavailable.")
        
    except Exception as e:
        print(f"Error during output scanning: {str(e)}")

    # Assess the latest message and update student_record.md
    asyncio.create_task(ta_model.assess_message(message_history))

    message_history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("message_history", message_history)
    
    await response_message.update()

if __name__ == "__main__":
    cl.main()
