import cv2
import os
import argparse

def getLocalFile(filePath):
    cwd = os.getcwd()
    path = cwd + filePath
    return path


# for testing cascades on images
def cascadeTest():
    cascadeFile = getLocalFile('\cascades\mustache8.xml')
    cascade = cv2.CascadeClassifier(cascadeFile)
    imageFile = getLocalFile("\\testimg\\19.jpg")
    # face_classifier = cv2.CascadeClassifier('/haarcascade_frontalface_default.xml')
    img = cv2.imread(imageFile)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected = cascade.detectMultiScale(img_gray)
    print("objects detected: " + str(len(detected)))
    for cords in detected:
        print(type(cords[0]))
        image = cv2.rectangle(img, (cords[0], cords[1]), (cords[0] + cords[2], cords[1] + cords[3]), (255, 0, 0), 1)
        cv2.imwrite("testimg/19_8.jpg", image)


# for testing cascades on video
def videoTest():
    cascadeFile = getLocalFile('\cascades\mustache8.xml')
    cascade = cv2.CascadeClassifier(cascadeFile)
    stream = cv2.VideoCapture(0)
    while True:
        ret, img = stream.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected = cascade.detectMultiScale(img_gray)
        for (x, y, w, h) in detected:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    stream.release()
    cv2.destroyAllWindows()

# process an image to display mustache detection overlay
def processMustacheDetect(img):
    mustacheCascadeFile = getLocalFile('\cascades\mustache8.xml')
    faceCascadeFile = getLocalFile('\cascades\pretrained\haarcascade_frontalface_default.xml')
    noseCascadeFile = getLocalFile('\cascades\pretrained\haarcascade_mcs_nose.xml')
    mustacheCascade = cv2.CascadeClassifier(mustacheCascadeFile)
    faceCascade = cv2.CascadeClassifier(faceCascadeFile)
    noseCascade = cv2.CascadeClassifier(noseCascadeFile)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(img_grey)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        face_grey = img_grey[y:y + h, x:x + w]
        face_color = img[y:y + h, x:x + w]
        nose_y = 0
        nose = noseCascade.detectMultiScale(face_grey)
        mustache = mustacheCascade.detectMultiScale(face_color)
        for (nx, ny, nw, nh) in nose:
            if ny > nose_y:
                nose_y = ny
            cv2.rectangle(face_color, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 1)
        for (mx, my, mw, mh) in mustache:
            if my > nose_y:
                cv2.rectangle(face_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 1)
    return img

# apply mustache detection overlay to specified image file
def mustacheDetectImg(file):
    img = cv2.imread(file)
    output = processMustacheDetect(img)
    while True:
        cv2.imshow('img', output)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

# display mustache detection overlay over video stream
def mustacheDetectVideo():
    stream = cv2.VideoCapture(0)
    while True:
        _, img = stream.read()
        img = processMustacheDetect(img)
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    stream.release()
    cv2.destroyAllWindows()


# handle adding a mustache to an image or frame
# based on code in tutorial by Noah Dietrich: https://sublimerobots.com/2015/02/dancing-mustaches/
def processMustacheify(img):
    faceCascadeFile = getLocalFile('\cascades\pretrained\haarcascade_frontalface_default.xml')
    noseCascadeFile = getLocalFile('\cascades\pretrained\haarcascade_mcs_nose.xml')
    faceCascade = cv2.CascadeClassifier(faceCascadeFile)
    noseCascade = cv2.CascadeClassifier(noseCascadeFile)
    mustacheFile = cv2.imread("img\mustache.png", -1)
    orig = mustacheFile[:, :, 3]
    orig_inverted = cv2.bitwise_not(orig)
    mustache = mustacheFile[:, :, 0:3]
    origHeight, origWidth = mustache.shape[:2]
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(img_grey)
    for (x, y, w, h) in faces:
        face_grey = img_grey[y:y + h, x:x + w]
        face_color = img[y:y + h, x:x + w]
        nose = noseCascade.detectMultiScale(face_grey)
        for (nx, ny, nw, nh) in nose:
            mustacheWidth = 3 * nw
            mustacheHeight = mustacheWidth * origHeight / origWidth
            x1 = int(nx - (mustacheWidth / 4))
            x2 = int(nx + nw + (mustacheWidth / 4))
            y1 = int(ny + nh - (mustacheHeight / 2))
            y2 = int(ny + nh + (mustacheHeight / 2))
            # prevent from exceeding the bounds of the face, which would cause errors
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h
            finalWidth = x2 - x1
            finalHeight = y2 - y1
            mustacheBackground = face_color[y1:y2, x1:x2]
            resizeMustache = cv2.resize(mustache, (finalWidth, finalHeight), interpolation=cv2.INTER_AREA)
            resizeOrig = cv2.resize(orig, (finalWidth, finalHeight), interpolation=cv2.INTER_AREA)
            resizeInvert = cv2.resize(orig_inverted, (finalWidth, finalHeight), interpolation=cv2.INTER_AREA)
            background = cv2.bitwise_and(mustacheBackground, mustacheBackground, mask=resizeInvert)
            foreground = cv2.bitwise_and(resizeMustache, resizeMustache, mask=resizeOrig)
            overlay = cv2.add(background, foreground)
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

# display mustache on faces in specified image file
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
    orig1 = google1File[:, :, 3]
    orig1_inverted = cv2.bitwise_not(orig1)
    eye1 = google1File[:, :, 0:3]
    # right eye
    orig2 = google2File[:, :, 3]
    orig2_inverted = cv2.bitwise_not(orig2)
    eye2 = google2File[:, :, 0:3]

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(img_grey)
    for (x, y, w, h) in faces:
        face_grey = img_grey[y:y + h, x:x + w]
        face_color = img[y:y + h, x:x + w]
        eyes = eyeCascade.detectMultiScale(face_grey)
        eyeindex = 0
        for (ex, ey, ew, eh) in eyes:
            eyeindex += 1
            if eyeindex % 2 == 0:
                eye = eye1
                mask = orig1
                mask_inv = orig1_inverted
            else:
                eye = eye2
                mask = orig2
                mask_inv = orig2_inverted
            eyeBackground = face_color[ey:ey + eh, ex:ex + ew]
            resizeMustache = cv2.resize(eye, (ew, eh), interpolation=cv2.INTER_AREA)
            resizeOrig = cv2.resize(mask, (ew, eh), interpolation=cv2.INTER_AREA)
            resizeInvert = cv2.resize(mask_inv, (ew, eh), interpolation=cv2.INTER_AREA)
            background = cv2.bitwise_and(eyeBackground, eyeBackground, mask=resizeInvert)
            foreground = cv2.bitwise_and(resizeMustache, resizeMustache, mask=resizeOrig)
            overlay = cv2.add(background, foreground)
            face_color[ey:ey + eh, ex:ex + ew] = overlay
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

# apply google eyes to faces on specified image file
def googleEyeImg(file):
    img = cv2.imread(file)
    output = processGoogleEye(img)
    while True:
        cv2.imshow('img', output)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        
# perform a face swap on faces detected in an image
def processFaceSwap(img):
    # NOTE FOR THIS FUNCTION: If you want to use squares/areas of equal size, uncomment out the '''...''' sections
    #   and comment out their counterparts instead (if applicable).

    # This function is currently oriented to switch faces onto each other with opposing sizes.

    # Load the Cascade
    faceCascadeFile = getLocalFile(r'/cascades/pretrained/haarcascade_frontalface_default.xml')
    faceCascade = cv2.CascadeClassifier(faceCascadeFile)

    # Read the image for faces, in grayscale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(img_grey)

    # (x,y) and (x+w, y+h) are the bounds of the rectangle that outlines each face
    #   (w and h are distances (positive) from the x and y values)
    #   (or better known, width and height)
    #
    # Usually, finding features on faces like eyes, work better within this 'for' loop, \
    #   because the classifier doesn't have to 'search' as far.
    #   For example, if you wanted to look for eyes, you generate a frame (sub-image) from the face coords:
    #       faceROI = frame_gray[y:y+h, x:x+w]
    #       eyes = eyes_cascade.detectMultiScale(faceROI)
    #       for (x2,y2,w2,h2):
    #           ...

    if len(faces) >= 2:
        # Store data on the faces, such as width (f1c1,f2c1) and height (f1d1,f2d1)
        f1a1, f1b1, f1c1, f1d1 = faces[0]
        f2a1, f2b1, f2c1, f2d1 = faces[1]

        # Ensure that image values are within acceptable parameters
        #   Also updates image width/height if shortened.
        if f1a1 < 0:
            f1c1 -= 0 - f1a1
            f1a1 = 0
        if f2a1 < 0:
            f2c1 -= 0 - f2a1
            f2a1 = 0
        if f1b1 < 0:
            f1d1 -= 0 - f1b1
            f1b1 = 0
        if f2b1 < 0:
            f2d1 -= 0 - f2b1
            f2b1 = 0

        '''
        # Max face size is the larger width/heights of the two
        d = max(f1d1, f2d1)
        c = max(f1c1, f2c1)

        # Normalize the sizes of the images to be extracted to be the same
        f1a2 = f1a1 + c
        f2a2 = f2a1 + c
        f1b2 = f1b1 + d
        f2b2 = f2b1 + d
        '''

        # The following is adjusted due to the failure of the face1 = img[] lines
        #   to properly store the data from the image. While the indices are correct, the
        #   actual size of the images tends to differ. There are likely null pixels or
        #   pixels that do not store information that are within these bounds, which
        #   artificially shorten the image sizes.
        # Essentially, rather than resizing to c and d, the workaround is that they resize to each other's sizes
        #   Then, a similar issue (to the prev paragraph) occurred with the img[] = face... lines
        #   The solution was, rather than base the indices there on the size of the resized imaged, we base it on the
        #   original image sizes, which somehow worked.

        # Store the faces to be switched
        '''
        face1 = img[f1b1:f1b2, f1a1:f1a2]
        face2 = img[f2b1:f2b2, f2a1:f2a2]
        '''
        face1 = img[f1b1:f1b1 + f1d1, f1a1:f1a1 + f1c1]
        face2 = img[f2b1:f2b1 + f2d1, f2a1:f2a1 + f1c1]

        # Resize the face images to be the size of the faces they will be applied to
        face1Resized = cv2.resize(face1, (face2.shape[1], face2.shape[0]), interpolation=cv2.INTER_AREA)
        face2Resized = cv2.resize(face2, (face1.shape[1], face1.shape[0]), interpolation=cv2.INTER_AREA)

        # Replace that area in the display to be the opposing face
        '''
        img[f1b1:f1b2, f1a1:f1a2] = face2Resized
        img[f2b1:f2b2, f2a1:f2a2] = face1Resized
        '''
        img[f1b1:f1b1 + face1.shape[0], f1a1:f1a1 + face1.shape[1]] = face2Resized
        img[f2b1:f2b1 + face2.shape[0], f2a1:f2a1 + face2.shape[1]] = face1Resized
    else:
        # Outline areas being switched for assistance in understanding function
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    return img

# swap faces found on video stream
def faceSwap():
    faceCascadeFile = getLocalFile(r'/cascades/pretrained/haarcascade_frontalface_default.xml')
    faceCascade = cv2.CascadeClassifier(faceCascadeFile)
    # Start up the camera
    stream = cv2.VideoCapture(0)

    while True:
        # Read the camera in values rather than actual 'pixels'
        _, img = stream.read()
        img = processFaceSwap(img)

        # Display the image
        cv2.imshow('img', img)

        # Quit condition: press Esc to quit (ASCII = 27)
        k = cv2.waitKey(50) & 0xff
        if k == 27:
            break

    stream.release()
    cv2.destroyAllWindows()

# swap faces found in specified image file
def faceSwapImg(file):
    img = cv2.imread(file)
    output = processFaceSwap(img)
    while True:
        cv2.imshow('img', output)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

# apply a cat face filter over detected faces
def catifyFaces(img):
    catFaceFile = cv2.imread("img/cat.png", -1)  # Replace with your cat face overlay image
    catFace = catFaceFile[:, :, 0:3]
    catFaceAlpha = catFaceFile[:, :, 3]

    faceCascadeFile =getLocalFile(r'/cascades/pretrained/haarcascade_frontalface_default.xml')
    faceCascade = cv2.CascadeClassifier(faceCascadeFile)

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(img_grey)

    for (x, y, w, h) in faces:
        face_color = img[y:y + h, x:x + w]

        # Resize cat face to match face dimensions
        catFaceResized = cv2.resize(catFace, (w, h), interpolation=cv2.INTER_AREA)
        catFaceAlphaResized = cv2.resize(catFaceAlpha, (w, h), interpolation=cv2.INTER_AREA)

        # Overlay cat face on the person's face
        background = cv2.bitwise_and(face_color, face_color, mask=cv2.bitwise_not(catFaceAlphaResized))
        foreground = cv2.bitwise_and(catFaceResized, catFaceResized, mask=catFaceAlphaResized)
        overlay = cv2.add(background, foreground)
        img[y:y + h, x:x + w] = overlay

    return img

# apply cat filter over video stream
def catifyVideo():
    stream = cv2.VideoCapture(0)
    while True:
        _, img = stream.read()
        img = catifyFaces(img)
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    stream.release()
    cv2.destroyAllWindows()


### handle args ###
parser = argparse.ArgumentParser()
parser.add_argument("-func", help="which function to run (mustacheify, googleeye, etc..)")
parser.add_argument("-img", help="image to run function, if none will run video")
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
elif (func == 'c' or func == 'catify'):
    if img != 'None':
        img = cv2.imread(img)
        output = catifyFaces(img)
        while True:
            cv2.imshow('img', output)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
    else:
        catifyVideo()
elif (func == 'f' or func == 'faceswap'):
    if img != 'None':
        faceSwapImg(img)
    else:
        faceSwap()
