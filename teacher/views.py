from transformers import pipeline
from django.http import JsonResponse
import speech_recognition as sr
import pyttsx3
from django.shortcuts import render

# Load AI Models
chatbot_pipeline = pipeline("text-generation", model="gpt2")
lesson_pipeline = pipeline("text2text-generation", model="t5-small")

# ✅ AI Chatbot (Free GPT-2 Model)
def ai_chatbot(request):
    query = request.GET.get("query", "Explain Python basics")
    response = chatbot_pipeline(query, max_length=100)[0]['generated_text']
    return JsonResponse({"response": response})

def home_view(request):
    return render(request, "teacher/home.html")  # Make sure to create home.html

# ✅ Lesson Generator (Free T5 Model)
def generate_lesson(request):
    topic = request.GET.get("topic", "Python Basics")
    lesson = lesson_pipeline(f"Explain {topic}")[0]['generated_text']
    return JsonResponse({"lesson": lesson})

# ✅ Quiz Generator (Using a Simple Question Template)
def generate_quiz(request):
    topic = request.GET.get("topic", "Python")
    question = f"What is the purpose of {topic} in programming?"
    options = ["A) To make the code slower", "B) To improve performance", "C) To confuse developers", "D) To increase memory usage"]
    correct_answer = "B"

    return JsonResponse({"quiz": question, "options": options, "answer": correct_answer})

# ✅ Speech-to-Text (STT) - Uses Google Speech API
def speech_to_text(request):
    recognizer = sr.Recognizer()
    with sr.AudioFile("audio.wav") as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
    return JsonResponse({"text": text})

# ✅ Text-to-Speech (TTS) - Uses `pyttsx3`
def text_to_speech(request):
    text = request.GET.get("text", "Hello, this is an AI Teacher.")
    engine = pyttsx3.init()
    engine.save_to_file(text, "output.mp3")
    engine.runAndWait()
    return JsonResponse({"status": "Audio generated"})
