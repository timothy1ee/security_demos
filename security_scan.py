import giskard
from ta_model import model_predict
from langfuse.openai import OpenAI
from giskard.llm.client.openai import OpenAIClient

giskard.llm.set_llm_api("openai")
oc = OpenAIClient(model="gpt-4o-mini", client=OpenAI())
giskard.llm.set_default_client(oc)

# Create a Giskard model
giskard_model = giskard.Model(
    model=model_predict,
    model_type="text_generation",
    name="Computer Science TA Bot",
    description="This model acts as a computer science TA, answering questions and providing guidance on programming topics.",
    feature_names=["question"],
)

# Perform the security scan
scan_results = giskard.scan(giskard_model)
scan_results.to_html("scan_results.html")

print("Security scan completed. Results saved to 'scan_results.html'.")
