from django.db import models

# Create your models here.
from django.db import models
from django.core.validators import EmailValidator, MinLengthValidator, RegexValidator

class UserRegisteredTable(models.Model):
    
    
    name = models.CharField(max_length=100, validators=[
        RegexValidator(regex=r'^[a-zA-Z\s]+$', message="Name must contain only letters and spaces.")
    ])
    
    # Login ID: alphanumeric and between 5-20 characters
    loginid = models.CharField(max_length=20, unique=True, validators=[
        RegexValidator(regex=r'^[a-zA-Z0-9]{5,20}$', message="Login ID must be alphanumeric and between 5-20 characters.")
    ])
    
    # Email
    email = models.EmailField(max_length=254, unique=True, validators=[
        EmailValidator(message="Enter a valid email address.")
    ])
    
    # Mobile: 10 digits
    mobile = models.CharField(max_length=10, validators=[
        RegexValidator(regex=r'^\d{10}$', message="Mobile number must be 10 digits.")
    ])
    
    # Password: Minimum 8 characters, at least one letter and one number
    password = models.CharField(max_length=50, validators=[
        RegexValidator(regex=r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@,$,%,&]{8,}$', message="Password must contain at least 8 characters, including letters and numbers.")
    ])
    
    # Locality
    locality = models.CharField(max_length=100, validators=[
        RegexValidator(regex=r'^[\w\s]+$', message="Locality must contain only letters, numbers, and spaces.")
    ])

    # State
    state = models.CharField(max_length=50, validators=[
        RegexValidator(regex=r'^[\w\s]+$', message="State must contain only letters, numbers, and spaces.")
    ])
    status = models.CharField(max_length=10,  default='Waiting')

    def __str__(self):
        return f'{self.name} - {self.loginid}'

class PillInfo(models.Model):
    name = models.CharField(max_length=100, unique=True)
    usage = models.TextField()
    dosage = models.TextField()
    consumption_time = models.CharField(max_length=100, blank=True, help_text="e.g., Morning, Night, As needed")
    food_relation = models.CharField(max_length=100, blank=True, help_text="e.g., Before food, After food, With food")
    side_effects = models.TextField(blank=True)
    precautions = models.TextField(blank=True)

    def __str__(self):
        return self.name