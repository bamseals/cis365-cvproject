# cis365-cvproject
CIS 365 - Final Project - OpenCV Image Utilities

Sam Beals, Brett Gritters, Travis Johnson

https://github.com/bamseals/cis365-cvproject

 ---------- HOW TO USE ----------

** opencv-python must be installed locally (pip install opencv-python)

From root project folder (/cis365-cvproject/) run any of the following commands:

"python main.py -func d" displays mustache detection overlay on video stream from webcam

"python main.py -func d -img imagename.jpg" displays a mustache detection overlay on an individual image

"python main.py -func m" display virtual mustache on detected faces/noses on video stream from webcam

"python main.py -func m -img imagename.jpg" display virtual mustache on detected faces/noses of individual image

"python main.py -func g" display virtual google eyes over detected faces/eyes on video stream from webcam

"python main.py -func g -img imagename.jpg" display virtual google eyes over detected faces/eyes on an individual image

"python main.py -func c" display virtual cat face over detected faces on video stream from webcam

"python main.py -func c -img imagename.jpg" display virtual cat face over detected faces on an individual image

"python main.py -func f" display swapped faces for two detected faces on video stream from webcam

"python main.py -func f -img imagename.jpg" display swapped faces for two detected faces on an individual image


---------- Requirements ----------

opencv-python
https://pypi.org/project/opencv-python/

Training Cascade Classifiers:
OpenCV - version 3.4.3 - Other versions may also work
https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.4.3/


--- Helpful documentation, articles, and tutorials --- 

https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html

https://docs.opencv.org/4.x/dc/d88/tutorial_traincascade.html

https://medium.com/@mumbaiyachori/train-a-custom-dataset-for-object-detection-using-haar-cascade-in-windows-f1504e9641c0

https://sublimerobots.com/2015/02/dancing-mustaches/
