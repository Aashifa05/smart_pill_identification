from django.shortcuts import render
from django.contrib import messages

from Users.models import UserRegisteredTable

# Create your views here.

def adminLoginCheck(request):
    if request.method=="POST":
        login_id=request.POST['loginid']
        password=request.POST['password']

        if login_id=='admin' and password=='admin':
            return render(request,'admin/adminHome.html')
        else:
            messages.error(request,'Invalid details')
            return render(request,'adminLogin.html')
        
def adminHome(request):
    return render(request,'admin/adminHome.html')

def userDetails(request):
    user=UserRegisteredTable.objects.all()
    return render(request,'admin/userDetails.html',{'user':user})

def activateUser(request):
    loginid=request.GET['loginid']
    user=UserRegisteredTable.objects.get(loginid=loginid)
    user.status='activated'
    user.save()
    userr=UserRegisteredTable.objects.all()
    return render(request,'admin/userDetails.html',{'user':userr})

#import of main function from utility/users
from Users.utility.requirement import main

def adminclassificationView(request):
    accuracy,precision,recall=main()
    return render(request,'users/classificationView.html',context={'accurecy':accuracy,'precision':precision,'recall':recall})


