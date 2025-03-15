from django.urls import path
from .views import chat_page, ai_response

urlpatterns = [
    path('', chat_page, name='chat_page'),
    path('ai-response/', ai_response, name='ai_response'),
]
