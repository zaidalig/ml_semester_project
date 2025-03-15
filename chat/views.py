from django.shortcuts import render
from django.http import JsonResponse
from transformers import pipeline

# Load the text-generation model (This runs locally)
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")


def get_ai_response(user_input):
    try:
        response = generator(
            user_input,
            max_length=250,  # Reduce length for faster responses
            num_return_sequences=1,
            truncation=True,  # Explicitly enable truncation
            temperature=0.7  # Control randomness
        )
        return response[0]['generated_text']
    except Exception as e:
        return f"Error: {str(e)}"


def chat_page(request):
    return render(request, 'chat/chat.html')


def ai_response(request):
    user_input = request.GET.get("message", "")
    response_text = get_ai_response(user_input)
    return JsonResponse({"response": response_text})
