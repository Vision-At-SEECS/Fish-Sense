import cv2
from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json
import base64
from django.utils.encoding import smart_str
from django.http import HttpResponse
from django.views.static import serve
from django.http import FileResponse
#from sendfile import sendfile
from . import hist_equalization
import os
# Create your views here.

def index(request):
    return render(request, 'index.html')


@api_view(["POST"])
def histEqaulization(request):
    try:
        data=json.loads(request.body)
        source_img = data['source_img']
        base64_img_bytes = source_img.encode('utf-8')
        print("before open")
        with open('source_img.png', 'wb') as file_to_save:
            decoded_image_data = base64.decodebytes(base64_img_bytes)
            file_to_save.write(decoded_image_data)
        print("after open")
        data['source_img'] = 'source_img.png'


        resp = hist_equalization.result(data)
        # resp = {
        #     1 : [10, 20],
        #     2 : [30, 40],
        #     3 : [50, 60]
        # }
        print("resp in view", resp)
        return Response(resp)
        
        """print("responseeeee",resp)
            # return Response(resp)
        temp = {
            "length" : resp[1][0],
            "weight" : resp[1][1]
        }
        print("Tempeeeeee",temp)
    
        context = {"length": resp[1][0], "weight": resp[1][1]}
        return render(request, 'index.html',context)
        # return sendfile(request, resp, attachment=True)"""

    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)
@api_view(['Get'])
def downloadImage(request):
    image_data = open("\\static\\matched.png", "rb").read()
    response =  HttpResponse(image_data, content_type="image/png")
    response['Content-Disposition'] = f'attachment; filename=matched_img.png'
    return response

@api_view(['Get'])
def downloadImage2(request):
    image_data = open("\\static\\after_thresholding.jpg", "rb").read()
    response =  HttpResponse(image_data, content_type="image/jpg")
    response['Content-Disposition'] = f'attachment; filename=after_thresholding.jpg'
    return response    