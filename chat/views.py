from django.shortcuts import render
from django.http import JsonResponse
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

# === Load your local model ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml_model")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.eval()


# === Response generator ===
def get_ai_response(user_input, max_length=250):
    try:
        prompt = f"Question: {user_input}\nAnswer:"
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.7
            )

        full_output = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract only the answer portion (remove "Question: ... Answer:" if present)
        answer_start = full_output.find("Answer:")
        if answer_start != -1:
            response = full_output[answer_start + len("Answer:"):].strip()
        else:
            response = full_output.strip()

        return response
    except Exception as e:
        return f"Error: {str(e)}"


# === Views ===
def chat_page(request):
    return render(request, 'chat/chat.html')


def ai_response(request):
    user_input = request.GET.get("message", "")
    response_text = get_ai_response(user_input)
    return JsonResponse({"response": response_text})
