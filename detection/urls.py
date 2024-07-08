from django.urls import path
from . import views

urlpatterns = [
    path('', views.detect_plate, name='detect_plate'),
]
