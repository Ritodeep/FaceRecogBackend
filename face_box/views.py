from django.shortcuts import render

# Create your views here.
from post.serializers import PostSerializer
from post.models import Post

from student.models import Student
from student.serializers import StudentSerializer

from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from post import utils
from post import hog_det
from post import mt_embed1
import json

from . import utils

from django.http import JsonResponse
from django.http import FileResponse
from django.http import HttpResponse

class FaceBoxView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        posts = Post.objects.all()
        serializer = PostSerializer(posts, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        posts_serializer = PostSerializer(data=request.data)
        if posts_serializer.is_valid():
            posts_serializer.save()
            hog_det.hog('/home/ritodeep/Desktop/form-data/backend'+posts_serializer.data['image'])

            # img = open('/home/ritodeep/Desktop/form-data/backend/media/HOG_Test/Faces.jpg','rb')
            responseImg = utils.convert('/home/ritodeep/Desktop/form-data/backend/media/HOG_Test/Faces.jpg')

            return Response(responseImg)   


            try:
                with open('/home/ritodeep/Desktop/form-data/backend/media/HOG_Test/Faces.jpg', "rb") as f:
                    return JsonResponse(f.read(), content_type="image/jpeg",safe=False)
            except IOError:
                red = Image.new('RGBA', (1, 1), (255,0,0,0))
                response = HttpResponse(content_type="image/jpeg")
                red.save(response, "JPEG")
                return response
            
        else:
            print('error', posts_serializer.errors)
            return Response(posts_serializer.errors, status=status.HTTP_400_BAD_REQUEST)