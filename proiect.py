# Imports
import cv2
import numpy as np
import pyvirtualcam
import math
import cupy as cp


# Structura:
# Functii:
# Eliminare gaussiana
# 

# Functie Eliminare Gaussiana folosind Placa Video
def gaussian_elimination_cupy(a, b):
    n = len(b)
    for k in range(0, n - 1):
        for i in range(k + 1, n):
            if a[i, k] != 0.0:
                lam = a[i, k] / a[k, k]
                a[i, k + 1:n] = a[i, k + 1:n] - lam * a[k, k + 1:n]
                b[i] = b[i] - lam * b[k]
    for k in range(n - 1, -1, -1):
        b[k] = (b[k] - cp.dot(a[k, k + 1:n], b[k + 1:n])) / a[k, k]
    return b



# Efect - Ajustare Luminozitate
def adjust_brightness_cupy(frame, brightness=1.0, contrast=1.0):
    # Conversie de la RGB la BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Converseie de la BGR la YUV
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # Imparte frame-ul in canalele Y, U si V
    y, u, v = cv2.split(frame)
    # Conversie de la 2D la 1D
    y = y.flatten()
    # Conversie de la 1d la 2d
    y = y.reshape(frame.shape[0], frame.shape[1])
    # Ajusteaza luminozitate
    y = (contrast * y + brightness).clip(0, 255).astype(np.uint8)
    # Combina canalele
    frame = cv2.merge((y, u, v))
    # Conversie de la YUV la BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
    # Conversie de la BGR la RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


# Efect - detectie muchii (returneaza un frame alb-negru cu muchiile evidentiate)
def edge_detection(frame, kernel_size):
    # Conversie de la RGB la BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Converseie de la BGR la YUV
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # Imparte frame-ul in canalele Y, U si V
    y, u, v = cv2.split(frame)
    # Creaza kernel-ul
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # Aplica filtrul
    y = cv2.filter2D(y, -1, kernel)
    # Combina canalele
    frame = cv2.merge((y, u, v))
    # Conversie de la YUV la BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
    # Conversie de la BGR la RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


# Functie kernel blur
def blur_kernel(kernel_size, sigma):
    # Creaza kernel-ul
    kernel = np.zeros((kernel_size, kernel_size))
    # Calculeaza mijlocul kernel-ului
    center = kernel_size // 2
    # Calculeaza suma
    sum = 0
    # Umple kernel-ul
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            sum += kernel[i, j]
    # Normalizare kernel
    kernel /= sum
    return kernel


# Efect - Gaussian Blur folosind CuPy
def gaussian_blur_cupy(frame, kernel_size, sigma):
    # Conversie de la RGB la BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Converseie de la BGR la YUV
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # Imparte frame-ul in canalele Y, U si V
    y, u, v = cv2.split(frame)
    # Creeaza kernel-ul
    kernel = blur_kernel(kernel_size, sigma)
    # Aplica kernel-ul
    y = cv2.filter2D(y, -1, kernel)
    # Combina canalele
    frame = cv2.merge((y, u, v))
    # Conversie de la YUV la BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
    # Conversie de la BGR la RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


# Efect de rotire a imaginii folosind o matrice de rotatie (frame: 640 x 480 x 3)
def rotate_image_cupy(frame, angle):
    # Conversie de la RGB la BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Converseie de la BGR la YUV
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # Imparte frame-ul in canalele Y, U si V
    y, u, v = cv2.split(frame)
    # Calculeaza matricea de rotatie
    rot_mat = cv2.getRotationMatrix2D((y.shape[1] / 2, y.shape[0] / 2), angle, 1.0)
    # Aplica rotatia
    y = cv2.warpAffine(y, rot_mat, (y.shape[1], y.shape[0]), flags=cv2.INTER_LINEAR)
    # Combina canalele
    frame = cv2.merge((y, u, v))
    # Conversie de la YUV la BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
    # Conversie de la BGR la RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame



# functie de compresie a imginii (frame de la opencv: 640 x 480 x 3)
def compress_image_cupy(frame, quality):
    # Conversie de la RGB la BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Converseie de la BGR la YUV
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # Imparte frame-ul in canalele Y, U si V
    y, u, v = cv2.split(frame)
    # Aplica compresia
    y = cv2.imencode('.jpg', y, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    y = cv2.imdecode(y, cv2.IMREAD_GRAYSCALE)
    # Combina canalele
    frame = cv2.merge((y, u, v))
    # Conversie de la YUV la BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
    # Conversie de la BGR la RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame



# initialize the camera
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    

# initialize the virtual camera
with pyvirtualcam.Camera(width=640, height=480, fps=20) as cam:
    print(f'Using virtual camera: {cam.device}')

    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        ret, frame = cap.read()
        if not ret:
            break
        



        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            color = (255, 0, 0)  # BGR 0-255
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x,
                          end_cord_y), color, stroke)
            

        # Take x and y from faces
        print("\n\n\n FACES SHAPE: ", len(faces), "\n\n\n")

        # check if faces has at least one value, then compute the center of the face
        if len(faces) > 0:
            for i in range (0, len(faces)):
                # compute face center
                x_center = faces[i][0] + (faces[0][2]) / 2
                # debug print
                print("x_center: ", x_center)
                y_center = faces[i][1] + (faces[0][3]) / 2
                # debug print
                print("y_center: ", y_center)
                print(x_center, y_center)

                frame = frame[int(y_center - 120):int(y_center + 120), int(x_center - 160):int(x_center + 160)]
                # debug print
                print("crop successful: ", frame.shape)

                # Check if the frame is not empty
                if frame.size != 0:
                    # Resize the frame
                    frame = cv2.resize(frame, (640, 480))
                else:
                    continue
                
                
                

                # debug print
                print("resize successful: ", frame.shape)
            else:
                print("No faces detected")


        # if frame's empty, skip it
        if frame.size == 0:
            continue


        ##### Applying effects
        
        # Brightness adjustment - fine
        frame = adjust_brightness_cupy(frame, brightness=0.5, contrast=1.8)
        
        # APLICARE BLUR
        # frame = gaussian_blur_cupy(frame, kernel_size=35, sigma=1.6)
        
        # Aplicare Rotatie
        # frame = rotate_image_cupy(frame, angle=45)
        
        # Aplicare compresie
        # frame = compress_image_cupy(frame, quality=10)
        
        # Aplicare edge detection
        # frame = edge_detection(frame, kernel_size=5)
        
        # Convert frame from BGR to RGB
        
        

        
        
        cv2.imshow("Frame", frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # feed the input to pyvirtualcam to output the image to a virtual webcam
        cam.send(frame)
        cam.sleep_until_next_frame()

        # show the output frame

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
