from django.urls import path

from . import views

urlpatterns = [
  path('<str:namespace>', views.index, name='index'),
]
