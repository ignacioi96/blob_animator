#%matplotlib inline
import numpy as np
import cv2
import matplotlib.cm as mcm
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom
from skimage.feature import blob_log, blob_doh
from skimage.color import rgb2gray
from imageio import mimsave, get_writer
from pathlib import Path
import math, time, multiprocessing as mp
from tqdm import tqdm

# input: - set of images to be displayed
#        - the amount of images per row and column
# returns: nothing
# effect: displays the images ordered with rows and columns given
def plotNImages(imgs, row, col, titles=[], figsize=(3,15), cscheme='viridis'):
    index = 0
    if row == 1 and col == 1:
        fig, ax = plt.subplots(row, col, figsize=(2,7))
        fig.dpi = 200
        ax.imshow(imgs[index],cmap=plt.get_cmap(cscheme))
        if(len(titles) > 0):
            ax.set_title(titles[index], fontsize=5)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        fig, ax = plt.subplots(row, col, figsize=figsize)
        fig.dpi = 700
        for j in range(row):
            for k in range(col):
                if(index < len(imgs)):
                    if(row > 1 and col > 1):
                        ax[j,k].imshow(imgs[index],cmap=plt.get_cmap(cscheme));
                        if(len(titles) > 0):
                            ax[j,k].set_title(titles[index], fontsize=5)
                        ax[j,k].set_xticks([])
                        ax[j,k].set_yticks([])
                    else:
                        ax[max(j, k)].imshow(imgs[index],cmap=plt.get_cmap(cscheme));
                        if(len(titles) > 0):
                            ax[max(j,k)].set_title(titles[index], fontsize=5)
                        ax[max(j,k)].set_xticks([])
                        ax[max(j,k)].set_yticks([])
                index += 1
    fig.tight_layout()
    plt.draw()

def comic(img, canny_threshs=[100,110],edge_size=(3,3), inv=False):
    # edge detection on grayscale image for faster results
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    edges = cv2.blur(gray, (3, 3)) # get rid of noise
    edges = cv2.Canny(edges, canny_threshs[0], canny_threshs[1], apertureSize=3) # detect edges

    # make edges thicker
    kernel = np.ones(edge_size, dtype=np.float) / 15.0
    edges = cv2.filter2D(edges, 0, kernel)
    edges = cv2.threshold(edges, 50, 255, 0)[1]

    # convert back to BGR image (that's cv2's order not RGB)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # use a faster, more comprehnesible version of
    # a bilateral filter mean shift
    shifted = cv2.pyrMeanShiftFiltering(img, 5, 20)

    # now compose with the edges, the edges are white so take them away
    # to leave black

    if inv:
        return cv2.add(edges,shifted)
    return cv2.subtract(shifted, edges)

def getBlobsFromImg(orig_img):
    origImg = cv2.cvtColor(orig_img,cv2.COLOR_BGR2RGB)
    comicImg = cv2.cvtColor(comic(orig_img,[100,200],edge_size=(7,7)),cv2.COLOR_RGB2GRAY)
    ret, threshImg = cv2.threshold(comicImg,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.floodFill(threshImg,None,(0,0),255)
    threshImg = 255-threshImg
    colorImg = threshImg.copy()
    yxr = blob_log(threshImg,min_sigma=5,max_sigma=20, num_sigma=1,overlap=0.9,threshold=.37)
    #yxr = blob_doh(threshImg,min_sigma=5, max_sigma=8, num_sigma=1,overlap=0.9,threshold=.016)
    blobs = []
    curr = colorImg.copy()
    curr[:]=0
    prev = curr.copy()
    min_val = 10
    for y,x,r in yxr:
        blob = threshImg.copy()
        blob[blob==255]=min_val-1
        pos=(int(x),int(y))
        color=np.random.randint(min_val+1,high=250)
        cv2.floodFill(blob,None,pos,color)
        blob[blob<=min_val]=0
        curr[blob>=min_val] = 255
        if sum(sum(prev))!=sum(sum(curr)):
            colorImg[blob>min_val-1]=color
            prev = curr.copy()
            blobs.append(blob)
        else:
            curr = prev.copy()
    cv2.floodFill(colorImg,None,(0,1),0)
    return colorImg, np.array(blobs)

def getBlobsFromPath(path_to_img):
    origImg = cv2.cvtColor(path_to_img,cv2.COLOR_BGR2RGB)
    comicImg = cv2.cvtColor(comic(origImg,[100,200],edge_size=(7,7)),cv2.COLOR_RGB2GRAY)
    ret, threshImg = cv2.threshold(comicImg,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.floodFill(threshImg,None,(0,0),255)
    threshImg = 255-threshImg
    colorImg = threshImg.copy()
    #yxr = blob_log(threshImg,min_sigma=5,max_sigma=20, num_sigma=1,overlap=0.9,threshold=.37)
    yxr = blob_doh(threshImg,min_sigma=5, max_sigma=8, num_sigma=1,overlap=0.9,threshold=.016)
    blobs = []
    curr = colorImg.copy()
    curr[:]=0
    prev = curr.copy()
    min_val = 10
    for y,x,r in yxr:
        blob = threshImg.copy()
        blob[blob==255]=min_val-1
        pos=(int(x),int(y))
        color=np.random.randint(min_val+1,high=250)
        cv2.floodFill(blob,None,pos,color)
        blob[blob<=min_val]=0
        curr[blob>=min_val] = 255
        if sum(sum(prev))!=sum(sum(curr)):
            colorImg[blob>min_val-1]=color
            prev = curr.copy()
            blobs.append(blob)
        else:
            curr = prev.copy()
    cv2.floodFill(colorImg,None,(0,1),0)
    return colorImg, np.array(blobs)

def resizeAndRotateBlob(blob,coloredImg,ratio=1.0,angle=0):
    missingBlob = coloredImg.copy()
    missingBlob[blob>1]=0
    funkyImg = missingBlob.copy()
    new_blob = blob.copy()
    poly = np.nonzero(new_blob)
    x0=poly[0].min()
    x1=poly[0].max()
    y0=poly[1].min()
    y1=poly[1].max()
    center=(int((x0+x1)/2),int((y0+y1)/2))
    new_blob=blob[x0:x1,y0:y1]
    new_blob=zoom(new_blob,ratio)
    new_blob=rotate(new_blob,angle=angle)
    if len(new_blob) == 0:
        return coloredImg
    (width,height)=new_blob.shape
    (new_x,new_y)=(int(center[0]-width/2),int(center[1]-height/2))

    big_x = slice(max(0,new_x),max(min(new_x+width,funkyImg.shape[0]),0))
    big_y = slice(max(0,new_y),max(min(new_y+height,funkyImg.shape[1]),0))

    new_blob_x = slice(max(0,-new_x),min(-new_x+funkyImg.shape[0], width))
    new_blob_y = slice(max(0,-new_y),min(-new_y+funkyImg.shape[1], height))
    mask = funkyImg.copy()
    mask[:]=0
    kernel = np.ones((7,7),np.uint8)
    dilated = cv2.dilate(new_blob,kernel)[new_blob_x,new_blob_y]
    mask[big_x,big_y] += dilated
    funkyImg[mask>1]=0
    funkyImg[big_x,big_y] += new_blob[new_blob_x,new_blob_y]
    return funkyImg

def resizeAndRotateMany(blobs,coloredImg,ratios,angles,function,idx):
    funkyImg = coloredImg.copy()
    for i in range(len(blobs)):
        funkyImg = resizeAndRotateBlob(blobs[i],funkyImg,
                                       ratio=func(function[i][0],ratios[i],idx),
                                       angle=350*func(function[i][1],angles[i],idx))
    return funkyImg

def findObjectInImage(path_to_img,path_to_object='/Users/iki/Desktop/UBC/semestre_9/GRSJ_224F/images_exploration/esta.JPG'):
    img1 = cv2.imread(path_to_object)
    img2 = cv2.cvtColor(cv2.imread(path_to_img),cv2.COLOR_BGR2RGB)
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)
    sorted_matches = sorted(matches, key= lambda x:math.sqrt(math.pow(x[0].distance,2)+
                                                             math.pow(x[1].distance,2)))

    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in sorted_matches]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in sorted_matches ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    #sorted_matches = sorted_matches[:15]
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.15*n.distance:
            matchesMask[i]=[1,0]

    h,w = img1.shape[:2]
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    dst_extra = dst.copy()
    dst_extra += (w,0)
    draw_params = dict(matchColor = (0,0,255),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,sorted_matches,None,**draw_params)
    img3 = cv2.polylines(img3, [np.int32(dst_extra)],True, (0,255,0),5,cv2.LINE_AA)
    min_x = int(min(dst[:,0,1]))
    max_x = int(max(dst[:,0,1]))
    min_y = int(min(dst[:,0,0]))
    max_y = int(max(dst[:,0,0]))
    img_alpha = np.ones((img2.shape[0],img2.shape[1]),dtype=img2.dtype)
    for x in range(min_x,max_x):
        for y in range(min_y,max_y):
            if cv2.pointPolygonTest(dst,(y,x),False) >= 0:
                img_alpha[x,y] = 0
    cropped = img2[min_x:max_x, min_y:max_y]
    obj_alpha = 1-img_alpha[min_x:max_x, min_y:max_y]
    return cropped, (min_x, max_x, min_y, max_y), img2, img_alpha, obj_alpha

def reinsertObjectToImg(obj,img,limits,obj_alpha,mapping='viridis'):
    min_x = limits[0]
    max_x = limits[1]
    min_y = limits[2]
    max_y = limits[3]
    cmap = mcm.get_cmap(mapping)
    color_obj = cmap(obj)*255
    color_obj[:,:,3] = obj_alpha
    img_alpha = 1.0-obj_alpha
    #img[min_x:max_x,min_y:max_y] = color_obj
    for c in range(3):
        img[min_x:max_x,min_y:max_y, c] = (obj_alpha*color_obj[:,:,c]+
                                           img_alpha*img[min_x:max_x,min_y:max_y, c])
    return img, color_obj

def func(choice, var, idx):
    result = 1
    if choice == 0:
            result = np.sin(var*idx)+1.01
    elif choice == 1:
            result = np.cos(var*idx)+1.01
    elif choice ==  2:
            result = np.sin(np.sin(var*idx)*idx)+1.01
    elif choice ==  3:
            result = np.cos(np.cos(idx)*var*idx)+1.01
    if result > 3 or result < 0.1:
        result = np.random.randint(1,3)/2
    return round(result,2)

def makeSequence(image_path,path_to_object='/Users/iki/Desktop/UBC/semestre_9/GRSJ_224F/images_exploration/esta.JPG',
	dark_ratio=0.0,length=24,mapping='viridis', num_blobs=None):
    obj,limits,img,img_alpha,obj_alpha = findObjectInImage(image_path,path_to_object=path_to_object)
    colorImg, blobs = getBlobsFromImg(obj)
    sequence = []
    if (len(blobs) != 0):
        if num_blobs==None:
            num_blobs = np.random.randint(2,max(4,int(len(blobs)/2)))
        blob_ids = np.random.randint(0,len(blobs)-1,num_blobs)
        print('Number of blobs chosen: ', num_blobs)
        size_vars = np.random.randint(1,4,num_blobs)/np.random.randint(1,5,num_blobs)
        angle_vars = np.random.randint(1,4,num_blobs)/np.random.randint(1,5,num_blobs)
        func_vars = [[np.random.randint(0,4),np.random.randint(0,4)] for i in range(num_blobs)]
        for idx in range(length):
            if idx <= dark_ratio*length:
                thresholds = [1,2*idx]
                result = 90-cv2.cvtColor(comic(obj,thresholds,(7,7),inv=True), cv2.COLOR_RGB2GRAY)
                result[result==0]=255
            else:
                result = resizeAndRotateMany(blobs[blob_ids],colorImg,
                                             size_vars, angle_vars, func_vars, idx)
            sequence.append(result)
    final_sequence = []
    for scene in sequence:
        r,g,b = cv2.split(img)
        im = cv2.merge((r,g,b,img_alpha))
        newImg, colorObj = reinsertObjectToImg(scene,im,limits,obj_alpha,mapping=mapping)
        final_sequence.append(newImg)
    return final_sequence

def helper(count, im):
    seq = []
    print(count)
    path_to_obj = '/Users/iki/Desktop/UBC/semestre_9/GRSJ_224F/images_exploration/esta{}.JPG'.format(count%4)
    try:
        if count > 60:
            seq += makeSequence(str(im),path_to_object=path_to_obj,
            	dark_ratio=0.0,length=40,mapping='inferno')
        else:
            seq += makeSequence(str(im),path_to_object=path_to_obj,
            	dark_ratio=1.0,length=40,mapping='inferno')
        print('seq is: ', len(seq))
    except Exception as e:
        print(e)
    print('sigue')
    if len(seq) > 2:
        print('video start')
        start_vid = time.time()
        name = '/Users/iki/Desktop/UBC/semestre_9/GRSJ_224F/images_exploration/videos/video_{}.avi'.format(count)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(name,fourcc,20.0,(1024,768))
        for im in seq:
            im = cv2.resize(im,(1024,768))
            out.write(im)
        out.release()
        print('Time for video: ',time.time()-start_vid)
    return seq

start = time.time()
fotos_dir = Path.home().joinpath('Desktop/UBC/semestre_9/GRSJ_224F/images_exploration/fotos')
jpgs = fotos_dir.glob('*.JPG')
ordered = []
for count, jpg in enumerate(jpgs):
    ordered.append(jpg)
ordered = sorted(ordered,reverse=True)
seq = []

for count, im in enumerate(tqdm(ordered)):
    path_to_obj = '/Users/iki/Desktop/UBC/semestre_9/GRSJ_224F/images_exploration/esta{}.JPG'.format(count%4)
    try:
        if count > 0:
            seq = makeSequence(str(im),path_to_object=path_to_obj,
                                dark_ratio=0.0,length=20,mapping='inferno')
        else:
            seq = makeSequence(str(im),path_to_object=path_to_obj,
                                dark_ratio=1.0,length=20,mapping='inferno')
    except Exception as e:
        print(e)
    for i, img in enumerate(seq):
        img = cv2.resize(img,(1024,768))
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        cv2.imwrite('/Users/iki/Desktop/UBC/semestre_9/GRSJ_224F/'+
                    'images_exploration/fotos_listas/{}.jpg'.format(i+20*count),
                    img)
    seq = []
#pool = mp.Pool(processes=mp.cpu_count())
#process = pool.starmap(helper,[(count, im) for count, im in enumerate(ordered)])
#seq += [p for p in tqdm(process.get())]
print('Time for completion: ', time.time()-start)
