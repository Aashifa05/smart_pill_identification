"""
URL configuration for Detection_and_Analysis_of_Pill project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views as mv
from Admin import views as av
from Users import views as uv

urlpatterns = [
    path('admin/', admin.site.urls),

    #mainView urls
    path('',mv.index,name='index'),
    path('adminLoginForm',mv.adminLoginForm,name='adminLoginForm'),
    path('userLoginForm',mv.userLoginForm,name='userLoginForm'),
    path('userRegisterForm',mv.userRegisterForm,name='userRegisterForm'),

    #adminUrls
    path('adminLoginCheck',av.adminLoginCheck,name='adminLoginCheck'),
    path('adminHome',av.adminHome,name='adminHome'),
    path('userDetails',av.userDetails,name='userDetails'),
    path('activateUser',av.activateUser,name='activateUser'),
    path('adminclassificationView',av.adminclassificationView,name='adminclassificationView'),

    #userurls
    path('userRegister',uv.userRegister,name='userRegister'),
    path('userLoginCheck',uv.userLoginCheck,name='userLoginCheck'),
    path('prediction',uv.prediction,name='prediction'),
    path('classificationView',uv.classificationView,name='classificationView'),
    path('api/ask_advisor/',uv.ask_medical_advisor,name='ask_advisor'),
    
    # Voice input endpoints
    path('api/check_voice_input/',uv.check_voice_input_availability,name='check_voice_input'),
    path('api/record_voice_query/',uv.record_voice_query,name='record_voice_query'),
    path('api/voice_medication_query/',uv.voice_based_medication_query,name='voice_medication_query'),


    
    

    
    
]
