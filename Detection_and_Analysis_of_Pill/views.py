
from django.shortcuts import render


def index(request):
    return render(request,'index.html')

def adminLoginForm(request):
    return render(request,'adminLogin.html')

def userLoginForm(request):
    return render(request,'userLogin.html')

def userRegisterForm(request):
    return render(request,'register.html')