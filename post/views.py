from django.shortcuts import render

from .serializers import PostSerializer
from .models import Post

from student.models import Student
from student.serializers import StudentSerializer

from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from . import utils
from . import hog_det
from . import mt_embed1
import json

from django.http import JsonResponse
from django.http import FileResponse

# Create your views here.

class PostView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        posts = Post.objects.all()
        serializer = PostSerializer(posts, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        posts_serializer = PostSerializer(data=request.data)
        students = Student.objects.all()
        student_serializer = StudentSerializer(students, many=True)
        if posts_serializer.is_valid():
            posts_serializer.save()
            hog_det.hog('/home/ritodeep/Desktop/form-data/backend'+posts_serializer.data['image'])
            names = utils.predict('/home/ritodeep/Desktop/form-data/backend/media/HOG_Test/20-05-2020')
            # names = mt_embed1.test_im_folder('/home/ritodeep/Desktop/form-data/backend/media/HOG_Test/19-05-2020') 
            
            for i in range(len(student_serializer.data)):
                if student_serializer.data[i]['name'] in  names:
                    present = student_serializer.data[i]['days_present'] + 1
                    Student.objects.filter(name=student_serializer.data[i]['name']).update(days_present=present) 

            return JsonResponse(names, status=status.HTTP_201_CREATED,safe=False)
            
        else:
            print('error', posts_serializer.errors)
            return Response(posts_serializer.errors, status=status.HTTP_400_BAD_REQUEST)