import picamera     # Importing the library for camera module
from time import sleep
camera = picamera.PiCamera()    # Setting up the camera
camera.start_preview()
camera.annotate_text = 'Picture Taken with Raspberry camera'
sleep(5)
camera.capture('/home/pi/Desktop/camera/imag.jpg') # Capturing the image
camera.stop_preview()
print('Done')