from django.shortcuts import render

from .serializers import StudentSerializer
from .models import Student
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
import json
from django.http import JsonResponse

# Create your views here.

class StudentView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        students = Student.objects.all()
        serializer = StudentSerializer(students, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        students_serializer = StudentSerializer(data=request.data)
        if students_serializer.is_valid():
            students_serializer.save()
            return Response('done', status=status.HTTP_201_CREATED)
        else:
            print('error', students_serializer.errors)
            return Response(students_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
