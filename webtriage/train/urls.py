from django.urls import path

from . import views

urlpatterns = [
  path('', views.index, name='index'),
  path('<str:namespace>', views.train_model, name='train'),
]
