
import os
import re
import numpy as np
import uuid
from scipy import misc
import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

 
def read(file_in):
    if file_in.endswith('.float3'):

        return readFloat(file_in)
    elif file_in.endswith('.flo'): 
        return readFlow(file_in)
    elif file_in.endswith('.ppm'):
       return readImage(file_in)
    elif file_in.endswith('.pgm'): 
        return readImage(file_in)
    elif file_in.endswith('.png'):
        return readImage(file_in)
    elif file_in.endswith('.jpg'):
        return readImage(file_in)
    elif file_in.endswith('.pfm'):
        return readPFM(file_in)[0]
    else: raise Exception('don\'t know how to read %s' % file_in)
 
def write(file_in, data):
    if file_in.endswith('.float3'): return writeFloat(file_in, data)
    elif file_in.endswith('.flo'): return writeFlow(file_in, data)
    elif file_in.endswith('.ppm'): return writeImage(file_in, data)
    elif file_in.endswith('.pgm'): return writeImage(file_in, data)
    elif file_in.endswith('.png'): return writeImage(file_in, data)
    elif file_in.endswith('.jpg'): return writeImage(file_in, data)
    elif file_in.endswith('.pfm'): return writePFM(file_in, data)
    else: raise Exception('don\'t know how to write %s' % file_in)
 
def readPFM(file):
    file = open(file, 'rb')
 
    color = None
    width = None
    height = None
    scale = None
    endian = None
 
    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
 
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')
 
    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian
 
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
 
    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
 
def writePFM(file, image, scale=1):
    file = open(file, 'wb')
 
    color = None
 
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
 
    image = np.flipud(image)
 
    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
 
    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))
 
    endian = image.dtype.byteorder
 
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale
 
    file.write('%f\n'.encode() % scale)
 
    image.tofile(file)
 
def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]
 
    f = open(name, 'rb')
 
    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')
 
    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()
 
    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
 
    return flow.astype(np.float32)
 
def readImage(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        data = readPFM(name)[0]
        if len(data.shape)==3:
            return data[:,:,0:3]
        else:
            return data
 
    return misc.imread(name)
 
def writeImage(name, data):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return writePFM(name, data, 1)
 
    return misc.imsave(name, data)
 
def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)
 
def readFloat(name):
    f = open(name, 'rb')
 
    if(f.readline().decode("utf-8"))  != 'float\n':
        raise Exception('float file %s did not contain <float> keyword' % name)
 
    dim = int(f.readline())
 
    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d
 
    dims = list(reversed(dims))
 
    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim > 2:
        data = np.transpose(data, (2, 1, 0))
        data = np.transpose(data, (1, 0, 2))
 
    return data
 
def writeFloat(name, data):
    f = open(name, 'wb')
 
    dim=len(data.shape)
    if dim>3:
        raise Exception('bad float file dimension: %d' % dim)
 
    f.write(('float\n').encode('ascii'))
    f.write(('%d\n' % dim).encode('ascii'))
 
    if dim == 1:
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
    else:
        f.write(('%d\n' % data.shape[1]).encode('ascii'))
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
        for i in range(2, dim):
            f.write(('%d\n' % data.shape[i]).encode('ascii'))
    
    data = data.astype(np.float32)
    if dim==2:
        data.tofile(f)
 
    else:
        np.transpose(data, (2, 0, 1)).tofile(f)

def Pfm2Numpy(path):
    img=read(path)
    if img is None:
        raise Exception("Can't display an empty image.")
    else:
        temp_img=np.reshape(img, (len(img), len(img[0])))
        #plt.imshow(temp_img)  
        #plt.show()
        file_type=os.path.splitext(path)[0]
        #print(temp_img)
        temp_img=np.where(temp_img==np.inf,0,temp_img)
        temp_img=np.where(temp_img==np.nan,0,temp_img)
        Temp_img=Image.fromarray(cv2.applyColorMap(cv2.convertScaleAbs((temp_img-np.min(temp_img))/(np.max(temp_img)-np.min(temp_img))*255),cv2.COLORMAP_JET))
        
        #Temp_img=Temp_img.convert("I")

        #print()
        #Temp_img.save(file_type+".png")
        #print(file_type.split("/")[-2])
        GT=temp_img
        GT = cv2.resize(GT, (int(GT.shape[1]), int(GT.shape[0])))
        GT=GT
        GT=np.where(GT==np.inf,0,GT)
        GT=np.where(GT==np.nan,0,GT)
        #plt.imshow(Temp_img)
        #plt.show()
        #np.save(file_type+".npy",GT)
        return GT
def find_pfm_files(root_dir):
    pfm_files = []  # 用于保存找到的.pfm文件路径
    left_files=[]
    right_files=[]
    for root, dirs, files in os.walk(root_dir):
        if 'TEST' not in root.split(os.sep):  # 如果路径中不包含'TEST'
            if os.path.basename(root) == 'left':  # 如果当前文件夹是left
                for file in files:
                    if file.endswith('.pfm'):  # 如果文件以.pfm结尾
                        pfm_files.append(os.path.join(root, file))  # 添加文件路径到列表
                        left_files.append(os.path.join(root, file).replace("disparity","images").replace(".pfm",".png"))
                        right_files.append(os.path.join(root, file).replace("disparity","images").replace("left","right").replace(".pfm",".png"))
    return pfm_files,left_files,right_files


if __name__ == '__main__':
    root_dir = 'E:/Rongcheng/SceneFlow/'  # 请替换为你的目录
    pfm_files,left_files,right_files = find_pfm_files(root_dir)
    print(len(pfm_files))
    max_dis=-1
    for i, item in enumerate(tqdm(pfm_files)):
        
        GT=Pfm2Numpy(item)
        max_dis=max(np.max(GT),max_dis)
        print(max_dis)