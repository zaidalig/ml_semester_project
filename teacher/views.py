from transformers import pipeline
from django.http import JsonResponse
import speech_recognition as sr
import pyttsx3
from django.shortcuts import render
import os
from django.contrib.auth.decorators import login_required
# ✅ Use the fine-tuned GPT-2 model saved in /ml_model
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_model")

chatbot_pipeline = pipeline("text-generation", model=MODEL_DIR)
lesson_pipeline = pipeline("text2text-generation", model="t5-small")  # Still using default T5


# ✅ AI Chatbot (Custom-trained GPT-2 on ML dataset)
 # Ensure user is logged in

 
@login_required  # Ensure user is logged in
def ai_chatbot(request):
    query = request.GET.get("query", "Explain supervised learning")
    result = chatbot_pipeline(query, max_length=100, do_sample=True, temperature=0.7)
    response = result[0]['generated_text']
    return JsonResponse({"response": response})

# ✅ Home view
 # Ensure user is logged in


@login_required  # Ensure user is logged in
def home_view(request):
    return render(request, "teacher/home.html")  # Make sure this template exists


# ✅ Lesson Generator using T5
 # Ensure user is logged in
def generate_lesson(request):
    topic = request.GET.get("topic", "Python Basics")
    lesson = lesson_pipeline(f"Explain {topic}")[0]['generated_text']
    return JsonResponse({"lesson": lesson})


# ✅ Quiz Generator (static template)

def generate_quiz(request):
    topic = request.GET.get("topic", "Python")
    question = f"What is the purpose of {topic} in programming?"
    options = ["A) To make the code slower", "B) To improve performance", "C) To confuse developers", "D) To increase memory usage"]
    correct_answer = "B"
    return JsonResponse({"quiz": question, "options": options, "answer": correct_answer})


# ✅ Speech-to-Text using Google Speech Recognition
 # Ensure user is logged in
def speech_to_text(request):
    recognizer = sr.Recognizer()
    with sr.AudioFile("audio.wav") as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
    return JsonResponse({"text": text})


# ✅ Text-to-Speech using pyttsx3
 # Ensure user is logged in
def text_to_speech(request):
    text = request.GET.get("text", "Hello, this is an AI Teacher.")
    engine = pyttsx3.init()
    engine.save_to_file(text, "output.mp3")
    engine.runAndWait()
    return JsonResponse({"status": "Audio generated"})
