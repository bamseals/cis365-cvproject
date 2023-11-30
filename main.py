import cv2
import os
import argparse
import numpy as np

def getLocalFile(filePath):
    cwd = os.getcwd()
    path = cwd+filePath
    return path

# for testing cascades on images
def cascadeTest():
    cascadeFile = getLocalFile('\cascades\mustache8.xml')
    cascade = cv2.CascadeClassifier(cascadeFile)
    imageFile = getLocalFile("\\testimg\\19.jpg")
    #face_classifier = cv2.CascadeClassifier('/haarcascade_frontalface_default.xml')
    img = cv2.imread(imageFile)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected = cascade.detectMultiScale(img_gray)
    print("objects detected: "+str(len(detected)))
    for cords in detected:
        print(type(cords[0]))
        image = cv2.rectangle(img,(cords[0],cords[1]),(cords[0]+cords[2],cords[1]+cords[3]), (255, 0, 0), 1)
        cv2.imwrite("testimg/19_8.jpg",image)

# for testing cascades on video
def videoTest():
    cascadeFile = getLocalFile('\cascades\mustache8.xml')
    cascade = cv2.CascadeClassifier(cascadeFile)
    stream = cv2.VideoCapture(0)
    while True:
        ret, img = stream.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected = cascade.detectMultiScale(img_gray)
        for (x,y,w,h) in detected:
            cv2.rectangle(img, (x,y), (x+w , y+h), (255,0,0), 1)
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    stream.release()
    cv2.destroyAllWindows()

def mustacheDetectImg(file):
    mustacheCascadeFile = getLocalFile('\cascades\mustache8.xml')
    faceCascadeFile = getLocalFile('\cascades\pretrained\haarcascade_frontalface_default.xml')
    noseCascadeFile = getLocalFile('\cascades\pretrained\haarcascade_mcs_nose.xml')
    mustacheCascade = cv2.CascadeClassifier(mustacheCascadeFile)
    faceCascade = cv2.CascadeClassifier(faceCascadeFile)
    noseCascade = cv2.CascadeClassifier(noseCascadeFile)
    img = cv2.imread(file)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(img_grey)
    for (x, y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        face_grey = img_grey[y:y+h, x:x+w]
        face_color = img[y:y+h, x:x+w]
        nose_y = 0
        nose = noseCascade.detectMultiScale(face_grey)
        mustache = mustacheCascade.detectMultiScale(face_color)
        for (nx, ny, nw, nh) in nose:
            if ny > nose_y:
                nose_y = ny
            cv2.rectangle(face_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),1)
        for (mx, my, mw, mh) in mustache:
            if my > nose_y:
                cv2.rectangle(face_color,(mx,my),(mx+mw,my+mh),(0,0,255),1)
    while True:
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

def mustacheDetectVideo():
    mustacheCascadeFile = getLocalFile('\cascades\mustache8.xml')
    faceCascadeFile = getLocalFile('\cascades\pretrained\haarcascade_frontalface_default.xml')
    noseCascadeFile = getLocalFile('\cascades\pretrained\haarcascade_mcs_nose.xml')
    mustacheCascade = cv2.CascadeClassifier(mustacheCascadeFile)
    faceCascade = cv2.CascadeClassifier(faceCascadeFile)
    noseCascade = cv2.CascadeClassifier(noseCascadeFile)
    stream = cv2.VideoCapture(0)
    while True:
        _, img = stream.read()
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(img_grey)
        for (x, y, w, h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
            face_grey = img_grey[y:y+h, x:x+w]
            face_color = img[y:y+h, x:x+w]
            nose_y = 0
            nose = noseCascade.detectMultiScale(face_grey)
            mustache = mustacheCascade.detectMultiScale(face_color)
            for (nx, ny, nw, nh) in nose:
                if ny > nose_y:
                    nose_y = ny
                cv2.rectangle(face_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),1)
            for (mx, my, mw, mh) in mustache:
                if my > nose_y:
                    cv2.rectangle(face_color,(mx,my),(mx+mw,my+mh),(0,0,255),1)
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    stream.release()
    cv2.destroyAllWindows()
    

# handle adding a mustache to an image or frame
def processMustacheify(img):
    faceCascadeFile = getLocalFile('\cascades\pretrained\haarcascade_frontalface_default.xml')
    noseCascadeFile = getLocalFile('\cascades\pretrained\haarcascade_mcs_nose.xml')
    faceCascade = cv2.CascadeClassifier(faceCascadeFile)
    noseCascade = cv2.CascadeClassifier(noseCascadeFile)
    mustacheFile = cv2.imread("img\mustache.png", -1)
    orig = mustacheFile[:,:,3]
    orig_inverted = cv2.bitwise_not(orig)
    mustache = mustacheFile[:,:,0:3]
    origHeight, origWidth = mustache.shape[:2]
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(img_grey)
    for (x, y, w, h) in faces:
        face_grey = img_grey[y:y+h, x:x+w]
        face_color = img[y:y+h, x:x+w]
        nose = noseCascade.detectMultiScale(face_grey)
        for (nx,ny,nw,nh) in nose:
            mustacheWidth =  3 * nw
            mustacheHeight = mustacheWidth * origHeight / origWidth
            x1 = int(nx - (mustacheWidth/4))
            x2 = int(nx + nw + (mustacheWidth/4))
            y1 = int(ny + nh - (mustacheHeight/2))
            y2 = int(ny + nh + (mustacheHeight/2))
            # prevent from exceeding the bounds of the face, which causes errors
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h
            finalWidth = x2-x1
            finalHeight = y2-y1
            mustacheBackground = face_color[y1:y2, x1:x2]
            resizeMustache = cv2.resize(mustache, (finalWidth,finalHeight), interpolation = cv2.INTER_AREA)
            resizeOrig = cv2.resize(orig, (finalWidth,finalHeight), interpolation = cv2.INTER_AREA)
            resizeInvert = cv2.resize(orig_inverted, (finalWidth,finalHeight), interpolation = cv2.INTER_AREA)
            background = cv2.bitwise_and(mustacheBackground,mustacheBackground,mask=resizeInvert)
            foreground = cv2.bitwise_and(resizeMustache,resizeMustache,mask=resizeOrig)
            overlay = cv2.add(background,foreground)
            face_color[y1:y2, x1:x2] = overlay
    return img

# add mustaches to faces on video
def mustacheifyVideo():
    stream = cv2.VideoCapture(0)
    while True:
        _, img = stream.read()
        img = processMustacheify(img)
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    stream.release()
    cv2.destroyAllWindows()

def mustacheifyImg(file):
    img = cv2.imread(file)
    output = processMustacheify(img)
    while True:
        cv2.imshow('img', output)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        
# process an image or frame, add google eyes
def processGoogleEye(img):
    faceCascadeFile = getLocalFile('\cascades\pretrained\haarcascade_frontalface_default.xml')
    eyeCascadeFile = getLocalFile('\cascades\pretrained\haarcascade_eye.xml')
    faceCascade = cv2.CascadeClassifier(faceCascadeFile)
    eyeCascade = cv2.CascadeClassifier(eyeCascadeFile)
    google1File = cv2.imread("img\googleeye1.png", -1)
    google2File = cv2.imread("img\googleeye2.png", -1)
    # left eye
    orig1 = google1File[:,:,3]
    orig1_inverted = cv2.bitwise_not(orig1)
    eye1 = google1File[:,:,0:3]
    # right eye
    orig2 = google2File[:,:,3]
    orig2_inverted = cv2.bitwise_not(orig2)
    eye2 = google2File[:,:,0:3]

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(img_grey)
    for (x, y, w, h) in faces:
        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        face_grey = img_grey[y:y+h, x:x+w]
        face_color = img[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(face_grey)
        eyeindex = 0
        for (ex,ey,ew,eh) in eyes:
            # cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
            eyeindex += 1
            if eyeindex %2 == 0:
                eye = eye1
                mask = orig1
                mask_inv = orig1_inverted
            else:
                eye = eye2
                mask = orig2
                mask_inv = orig2_inverted
            eyeBackground = face_color[ey:ey+eh, ex:ex+ew]
            resizeMustache = cv2.resize(eye, (ew,eh), interpolation = cv2.INTER_AREA)
            resizeOrig = cv2.resize(mask, (ew,eh), interpolation = cv2.INTER_AREA)
            resizeInvert = cv2.resize(mask_inv, (ew,eh), interpolation = cv2.INTER_AREA)
            background = cv2.bitwise_and(eyeBackground,eyeBackground,mask=resizeInvert)
            foreground = cv2.bitwise_and(resizeMustache,resizeMustache,mask=resizeOrig)
            overlay = cv2.add(background,foreground)
            face_color[ey:ey+eh, ex:ex+ew] = overlay
    return img

# run video stream through google eye process
def googleEyeVideo():
    stream = cv2.VideoCapture(0)
    while True:
        _, img = stream.read()
        img = processGoogleEye(img)
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    stream.release()
    cv2.destroyAllWindows()

def googleEyeImg(file):
    img = cv2.imread(file)
    output = processGoogleEye(img)
    while True:
        cv2.imshow('img', output)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

def faceSwap():
    faceCascadeFile = getLocalFile('\cascades\pretrained\haarcascade_frontalface_default.xml')
    faceCascade = cv2.CascadeClassifier(faceCascadeFile)
    # Start up the camera
    stream = cv2.VideoCapture(0)

    face1 = None
    face2 = None
    while True:
        # Read the camera in values rather than actual 'pixels'
        _, img = stream.read()

        # Read the Image(s) for faces in grayscale
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(img_grey)

        # (x,y) and (x+w, y+h) are the bounds of the rectangle that outlines each face
        #   (w and h are distances (positive) from the x and y values)
        #   (or better known, width and height)
        #
        # Usually finding features on faces, like eyes, work better within this 'for' loop, \
        #   because the classifier doesn't have to 'search' as far.
        #   For example, if you wanted to look for eyes, you generate a frame (sub-image) from the face coords \
        #       faceROI = frame_gray[y:y+h, x:x+w]
        #       eyes = eyes_cascade.detectMultiScale(faceROI)
        #       for (x2,y2,w2,h2):
        #           ...

        # Store the face images
        if len(faces) == 2:
            a1, b1, c1, d1 = faces[0]
            a2, b2, c2, d2 = faces[1]
            d = max(d1,d2)
            c = max(c1,c2)

            f1a2 = a1+c
            f2a2 = a2+c
            f1b2 = b1+d
            f2b2 = b2+d

            if a1 < 0:
                a1 = 0
            if a2 < 0:
                a2 = 0
            if b1 < 0:
                b1 = 0
            if b2 < 0:
                b2 = 0

            if f1a2 > 1:
                f1a2 = a1
            if f2a2 > 1:
                f2a2 = a2
            if f1b2 > 1:
                f1b2 = b1
            if f2b2 > 1:
                f1b1 = b2

            face1loc = faces[0]
            #face1 = img[b1:f1b2, a1:f1a2]
            face1 = img[20:28, 20:28]
            print(face1.shape)

            face2loc = faces[1]
            # todo: img sizes differ, causing errors
            #face2 = img[b2:b2+d, a2:a2+c]
            face2 = img[28:36, 28:36]
            print(face2.shape)


            # Overlay opposing faces onto each other
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            #maskface1 = cv2.bitwise_and(face1, face1, mask=mask)
            #maskface2 = cv2.bitwise_and(face2, face2, mask=mask)
            overlay1 = cv2.add(face2, face1)

        for (x, y, w, h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)

        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    stream.release()
    cv2.destroyAllWindows()

### handle args ###
parser = argparse.ArgumentParser()
parser.add_argument("-func", help="which function to run (mustacheify, googleeye, etc..)")
parser.add_argument("-img", help="image to run function, if none with run video")
args = parser.parse_args()
func = str(args.func)
img = str(args.img)
if (func == 'm' or func == 'mustache' or func == 'mustacheify'):
    if img != 'None':
        mustacheifyImg(img)
    else:
        mustacheifyVideo()
elif (func == 'g' or func == 'google' or func == 'googleeye' or func == 'googleeyes'):
    if img != 'None':
        googleEyeImg(img)
    else:
        googleEyeVideo()
elif (func == 't' or func == 'test'):
    if img != 'None':
        cascadeTest()
    else:
        videoTest()
elif (func == 'd' or func == 'detect'):
    if img != 'None':
        mustacheDetectImg(img)
    else:
        mustacheDetectVideo()
# NOTE: Currently non-functional - issue with face sizes or something
elif (func == 'f' or func == 'faceswap'):
    faceSwap()