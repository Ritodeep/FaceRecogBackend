from . import views
from django.urls import path

urlpatterns = [
    path('faces/', views.FaceBoxView.as_view(), name= 'facebox_list'),
]