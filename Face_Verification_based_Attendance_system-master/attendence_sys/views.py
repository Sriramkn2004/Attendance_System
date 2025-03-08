from venv import logger
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse
import cv2
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators import gzip
from datetime import date

from .forms import CreateStudentForm, FacultyForm
from .models import Student, Attendence
from .filters import AttendenceFilter
from .recognizer import Recognizer

@login_required(login_url='login')
def home(request):
    studentForm = CreateStudentForm()

    if request.method == 'POST':
        studentForm = CreateStudentForm(data=request.POST, files=request.FILES)
        if studentForm.is_valid():
            reg_id = request.POST.get('registration_id')
            if not Student.objects.filter(registration_id=reg_id).exists():
                studentForm.save()
                name = f"{studentForm.cleaned_data.get('firstname')} {studentForm.cleaned_data.get('lastname')}"
                messages.success(request, f'Student {name} was successfully added.')
            else:
                messages.error(request, f"Student with Registration ID {reg_id} already exists.")
        return redirect('home')

    return render(request, 'attendence_sys/home.html', {'studentForm': studentForm})

def loginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)

        if user:
            login(request, user)
            return redirect('home')
        else:
            messages.info(request, 'Username or Password is incorrect')

    return render(request, 'attendence_sys/login.html')

@login_required(login_url='login')
def logoutUser(request):
    logout(request)
    return redirect('login')

@login_required(login_url='login')
def updateStudentRedirect(request):
    if request.method == 'POST':
        try:
            student = Student.objects.get(registration_id=request.POST['reg_id'], branch=request.POST['branch'])
            updateStudentForm = CreateStudentForm(instance=student)
            return render(request, 'attendence_sys/student_update.html', {
                'form': updateStudentForm, 
                'prev_reg_id': request.POST['reg_id'], 
                'student': student
            })
        except Student.DoesNotExist:
            messages.error(request, 'Student Not Found')
    return redirect('home')

@login_required(login_url='login')
def updateStudent(request):
    if request.method == 'POST':
        try:
            student = Student.objects.get(registration_id=request.POST['prev_reg_id'])
            updateStudentForm = CreateStudentForm(data=request.POST, files=request.FILES, instance=student)
            if updateStudentForm.is_valid():
                updateStudentForm.save()
                messages.success(request, 'Update successful.')
        except Student.DoesNotExist:
            messages.error(request, 'Update unsuccessful.')
    return redirect('home')

@login_required(login_url='login')
def takeAttendence(request):
    if request.method == 'POST':
        details = {
            'branch': request.POST['branch'],
            'year': request.POST['year'],
            'section': request.POST['section'],
            'period': request.POST['period'],
        }
        if Attendence.objects.filter(date=str(date.today()), **details).exists():
            messages.error(request, "Attendance already recorded.")
        else:
            students = Student.objects.filter(
                branch=details['branch'], 
                year=details['year'], 
                section=details['section']
            )
            recognized_names = Recognizer(details)
            for student in students:
                Attendence.objects.create(
                    Faculty_Name=request.user.faculty,
                    Student_ID=str(student.registration_id),
                    branch=details['branch'],
                    year=details['year'],
                    section=details['section'],
                    period=details['period'],
                    status='Present' if str(student.registration_id) in recognized_names else 'Absent',
                )
            messages.success(request, "Attendance taken successfully.")
        return redirect('home')

    return redirect('home')

def searchAttendence(request):
    attendences = Attendence.objects.all()
    myFilter = AttendenceFilter(request.GET, queryset=attendences)
    return render(request, 'attendence_sys/attendence.html', {
        'myFilter': myFilter, 
        'attendences': myFilter.qs, 
        'ta': False
    })

def facultyProfile(request):
    form = FacultyForm(instance=request.user.faculty)
    return render(request, 'attendence_sys/facultyForm.html', {'form': form})

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, image = self.video.read()
        if ret:
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
        return None

    @staticmethod
    def gen(camera):
        while True:
            frame = camera.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@gzip.gzip_page
def videoFeed(request):
    try:
        return StreamingHttpResponse(VideoCamera.gen(VideoCamera()),
                                     content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:
        logger.error(f"Streaming aborted: {e}")

def getVideo(request):
    return render(request, 'attendence_sys/videoFeed.html')
