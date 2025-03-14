from .views import *
from django.urls import path

urlpatterns = [
    path("", home_view, name="home"),  # Home page
    path('chatbot/', ai_chatbot, name="ai_chatbot"),
    path('lesson/', generate_lesson, name="generate_lesson"),
    path('quiz/', generate_quiz, name="generate_quiz"),
]
