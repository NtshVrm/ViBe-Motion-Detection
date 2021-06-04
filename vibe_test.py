
import numpy as np
import os
import cv2

def initial_background(I_gray, N):

    I_pad = np.pad(I_gray, 1, 'symmetric')
    
    #Current image dimensions
    height = I_pad.shape[0]
    width = I_pad.shape[1]
   
   #initial background model
    samples = np.zeros((height,width,N))

    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            for n in range(N):
                x, y = 0, 0
                while(x == 0 and y == 0):
                    x = np.random.randint(-1, 1)
                    y = np.random.randint(-1, 1)

                    
                ri = i + x
                rj = j + y
                samples[i, j, n] = I_pad[ri, rj]
   

  
    samples = samples[1:height-1, 1:width-1]
  
    return samples
    
def vibe_detection(I_gray, samples, _min, N, R):
    
    height = I_gray.shape[0]
    width = I_gray.shape[1]

    #background/foreground segmentation map
    segMap = np.zeros((height, width)).astype(np.uint8)
    
    
    for i in range(height): 
        for j in range(width):

            #compare pixel to background model
            count, index, dist = 0, 0, 0
            while count < _min and index < N:

                #Euclidean Distance computation
                dist = np.abs(I_gray[i,j] - samples[i,j,index])
                if dist < R:
                    count += 1
                index += 1
            #Classify pixel and update model
            if count >= _min:

                r = np.random.randint(0, N-1)
                if r == 0:
                    r = np.random.randint(0, N-1)
                    samples[i,j,r] = I_gray[i,j]
                #Update neighbouring pixel model
                r = np.random.randint(0, N-1)
                if r == 0:
                    x, y = 0, 0
                    while(x == 0 and y == 0):
                        x = np.random.randint(-1, 1)
                        y = np.random.randint(-1, 1)
                    r = np.random.randint(0, N-1)
                    ri = i + x                          
                    rj = j + y
                    try:
                        samples[ri, rj, r] = I_gray[i, j]
                    except:
                        pass
            else:
                segMap[i, j] = 255
    return segMap, samples
    
rootDir = r'office/input'
image_file = os.path.join(rootDir, os.listdir(rootDir)[0])
image = cv2.imread(image_file, 0)

#Number of samples per pixel
N = 20

#Radius of the sphere
R = 20

'''Number of close samples for being 
part of the background'''
_min = 2

#amount of random subsampling
phai = 16



samples = initial_background(image, N)

for lists in os.listdir(rootDir): 
    path = os.path.join(rootDir, lists)
    frame = cv2.imread(path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    segMap, samples = vibe_detection(gray, samples, _min, N, R)
    cv2.imshow('segMap', segMap)
    if cv2.waitKey(1) and 0xff == ord('q'):
        break
cv2.destroyAllWindows()
