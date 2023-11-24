import cv2
import os

def cascadeTest():
    cwd = os.getcwd()
    cascadeFile = cwd+'\cascades\mustache7.xml'
    cascade = cv2.CascadeClassifier(cascadeFile)
    imageFile = cwd+"\\testimg\\5.jpg"
    #face_classifier = cv2.CascadeClassifier('/haarcascade_frontalface_default.xml')
    img = cv2.imread(imageFile)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected = cascade.detectMultiScale(img_gray)
    print("objects detected: "+str(len(detected)))
    for cords in detected:
        print(type(cords[0]))
        image = cv2.rectangle(img,(cords[0],cords[1]),(cords[0]+cords[2],cords[1]+cords[3]), (255, 0, 0), 1)
        cv2.imwrite("testimg/5_7.jpg",image)

# cascadeTest()

def videoTest():
    cwd = os.getcwd()
    cascadeFile = cwd+'\cascades\mustache7.xml'
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

# videoTest()

def mustacheifyImg(img):
    cwd = os.getcwd()
    faceCascadeFile = cwd+'\cascades\pretrained\haarcascade_frontalface_default.xml'
    noseCascadeFile = cwd+'\cascades\pretrained\haarcascade_mcs_nose.xml'
    faceCascade = cv2.CascadeClassifier(faceCascadeFile)
    noseCascade = cv2.CascadeClassifier(noseCascadeFile)

def mustacheifyVideo():
    cwd = os.getcwd()
    faceCascadeFile = cwd+'\cascades\pretrained\haarcascade_frontalface_default.xml'
    noseCascadeFile = cwd+'\cascades\pretrained\haarcascade_mcs_nose.xml'
    faceCascade = cv2.CascadeClassifier(faceCascadeFile)
    noseCascade = cv2.CascadeClassifier(noseCascadeFile)
    mustacheFile = cv2.imread("img\mustache.png", -1)
    orig = mustacheFile[:,:,3]
    orig_inverted = cv2.bitwise_not(orig)
    mustache = mustacheFile[:,:,0:3]
    origHeight, origWidth = mustache.shape[:2]

    # video input stream
    stream = cv2.VideoCapture(0)
    while True:
        ret, img = stream.read()
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(img_grey)
        for (x, y, w, h) in faces:
            # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
            face_grey = img_grey[y:y+h, x:x+w]
            face_color = img[y:y+h, x:x+w]
            nose = noseCascade.detectMultiScale(face_grey)
            for (nx,ny,nw,nh) in nose:
                # cv2.rectangle(face_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),1)
                mustacheWidth =  3 * nw
                mustacheHeight = mustacheWidth * origHeight / origWidth
                x1 = int(nx - (mustacheWidth/4))
                x2 = int(nx + nw + (mustacheWidth/4))
                y1 = int(ny + nh - (mustacheHeight/2))
                y2 = int(ny + nh + (mustacheHeight/2))
                x1 = 0 if x1 < 0 else x1
                x2 = 0 if x2 < 0 else x2
                y1 = 0 if y1 < 0 else y1
                y2 = 0 if y2 < 0 else y2
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
                break
        
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    stream.release()
    cv2.destroyAllWindows()
        
# mustacheifyVideo()

def googleEyeVideo():
    cwd = os.getcwd()
    faceCascadeFile = cwd+'\cascades\pretrained\haarcascade_frontalface_default.xml'
    eyeCascadeFile = cwd+'\cascades\pretrained\haarcascade_eye.xml'
    faceCascade = cv2.CascadeClassifier(faceCascadeFile)
    eyeCascade = cv2.CascadeClassifier(eyeCascadeFile)
    google1File = cv2.imread("img\googleeye1.png", -1)
    google2File = cv2.imread("img\googleeye2.png", -1)
    # left eye
    orig1 = google1File[:,:,3]
    orig1_inverted = cv2.bitwise_not(orig1)
    eye1 = google1File[:,:,0:3]
    orig1Height, orig1Width = eye1.shape[:2]
    # right eye
    orig2 = google2File[:,:,3]
    orig2_inverted = cv2.bitwise_not(orig2)
    eye2 = google2File[:,:,0:3]
    orig2Height, orig2Width = eye2.shape[:2]

    # video input stream
    stream = cv2.VideoCapture(0)
    while True:
        ret, img = stream.read()
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
                
        
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    stream.release()
    cv2.destroyAllWindows()

googleEyeVideo()


