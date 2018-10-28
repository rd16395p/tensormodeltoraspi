# tensormodeltoraspi
This is code to evaulate if a face is really a face on a raspberry pi interface or not using a pretrained model researched by Becca. The model is included in the picameraeval.py file. 

Items needed:
Raspberry Pi (3b was used here) <br />
Camera that works with raspberry pi <br />

Set-up:
Make sure to allow your raspberry pi to be used - here is a good guide; https://thepihut.com/blogs/raspberry-pi-tutorials/16021420-how-to-install-use-the-raspberry-pi-camera

Install:
pip install tensorflow <br />
pip install numpy <br />
pip install pandas <br />

To run:
Open up any pre-installed python IDEs on raspberry pi such as Thorny or Geanie, and run!

The results will be in the first directory called "results_withpredcnn2.csv" (maybe can be improved in the future)

"classes:" 1 - face

"classes:" 0 - not a face
