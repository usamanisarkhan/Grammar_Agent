import gradio as gr
from transformers import pipeline

# Load grammar correction model
grammar_model = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")

# AI agent logic
def grammar_agent(sentence):
    if not sentence.strip():
        return "Please enter a sentence."
    if len(sentence.split()) < 2:
        return "Try a longer sentence for meaningful correction."
    
    result = grammar_model(f"grammar: {sentence}", max_length=128, clean_up_tokenization_spaces=True)
    corrected = result[0]['generated_text']
    
    if corrected == sentence:
        return "✅ Looks great! No correction needed."
    else:
        return f"✏️ Corrected: {corrected}"

# Gradio UI
iface = gr.Interface(
    fn=grammar_agent,
    inputs=gr.Textbox(lines=3, placeholder="Enter a sentence with grammar issues..."),
    outputs="text",
    title="🧠 Grammar-Correcting AI Agent",
    description="Enter a sentence and get grammatically correct suggestions using a T5 model."
)

iface.launch()
