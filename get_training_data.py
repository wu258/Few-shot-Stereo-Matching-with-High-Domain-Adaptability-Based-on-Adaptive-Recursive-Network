import numpy as np
import os
import ctypes as c
from PIL import Image
from scipy.signal import convolve2d
from multiprocessing import Process,Manager
import multiprocessing
import itertools
import matplotlib.pyplot as plt
import numpy as np
import math
from numba import cuda, float32
import numba
from numba import jit
from numba import prange
from collections import OrderedDict
import cv2 as cv
import cv2 as cv2
from scipy.ndimage.interpolation import shift
from multiprocessing import Process,Manager
import torch
import time
from pfm2numpy import Pfm2Numpy 
import random
import gc
import argparse

def find_pfm_files(root_dir):
    pfm_files = []  # 用于保存找到的.pfm文件路径
    left_files=[]
    right_files=[]
    for root, dirs, files in os.walk(root_dir):
        if 'TEST' not in root.split(os.sep):  # 如果路径中不包含'TEST'
            if os.path.basename(root) == 'left':  # 如果当前文件夹是left
                for file in files:
                    if file.endswith('.pfm'):  # 如果文件以.pfm结尾
                        pfm_files.append(os.path.join(root, file).replace("\\","/"))  # 添加文件路径到列表
                        left_files.append(os.path.join(root, file).replace("disparity","images").replace(".pfm",".png").replace("\\","/"))
                        right_files.append(os.path.join(root, file).replace("disparity","images").replace("left","right").replace(".pfm",".png").replace("\\","/"))
    return pfm_files,left_files,right_files


@cuda.jit
def matching_Census(max_dis,img1_paded,img2_paded,padding_size,cost_volumn,windows_size):
    '''
    x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    y = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
    '''
    x, y = cuda.grid(2)
    if x<padding_size or (x>=(img1_paded.shape[1]-padding_size)) or (y<padding_size) or (y>=(img1_paded.shape[0]-padding_size)):
        return
    #print(x)
    windows_size=int(windows_size)
    if windows_size%2==0:
        windows_size=windows_size+1
    same=int(windows_size/2)

    left_img_patch=img1_paded[y-same:y+same+1,x-same:x+same+1]
    #left_census_list=np.arange(windows_size*windows_size)
    left_census_list=cuda.local.array(1226,numba.int32) 
    #for i in range(625):
        #print(C[i])
    left_count=0
    #left_census_list[2]=1
    
    for i in range(left_img_patch.shape[0]):
        for j in range(left_img_patch.shape[1]):

            if left_img_patch[i][j]<left_img_patch[same][same]:
                left_census_list[left_count]=1
            else:
                left_census_list[left_count]=0
            left_count=left_count+1
    for d in range(0,max_dis+1):
        cost=0
        #same=dist_map[y][x]//2
        if x-d-same<0:
            break
        right_img_patch=img2_paded[y-same:y+same+1,x-d-same:x-d+same+1]

        right_census_list=cuda.local.array(1226,numba.int32) 

        right_count=0

        for i in range(right_img_patch.shape[0]):
            for j in range(right_img_patch.shape[1]):

                if right_img_patch[i][j]<right_img_patch[same][same]:
                    right_census_list[right_count]=1
                else:
                    right_census_list[right_count]=0
                right_count=right_count+1
        if right_count!=left_count:
            print("haha")
        lenght_census=left_count
        cost=0
        for k in range(right_count):
            if(right_census_list[k]!=left_census_list[k]):
                cost=cost+1
        cost=cost/lenght_census

        cost_volumn[d][y-padding_size][x-padding_size]=cost
    
def get_init_disaprity(max_dis,img1,img2,windows_size,methode):
    print(img1.shape)
    #max_dis=512
    


    cost_volumn=np.ones((max_dis+1,img1.shape[0],img1.shape[1]))

    np.multiply(cost_volumn, 99, out=cost_volumn)
    #print(cost_volumn)
    #windows_size=15
    padding_size=windows_size
    img1_paded=np.pad(img1, ((padding_size, padding_size), (padding_size, padding_size)))
    img2_paded=np.pad(img2, ((padding_size, padding_size), (padding_size, padding_size)))

    BLOCK_SIZE=16
    threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
    blocks_per_grid_x = int(math.ceil(img1_paded.shape[1] / BLOCK_SIZE))
    blocks_per_grid_y = int(math.ceil(img1_paded.shape[0] / BLOCK_SIZE))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    img1_paded = cuda.to_device(img1_paded)
    img2_paded = cuda.to_device(img2_paded)
    cost_volumn = cuda.to_device(cost_volumn) 
    
    if methode=="Census":
        matching_Census[blocks_per_grid, threads_per_block](max_dis,img1_paded,img2_paded,padding_size,cost_volumn,windows_size)
    
    cuda.synchronize()
    
    cost_volumn=cost_volumn.copy_to_host()
    cost_volunm=torch.from_numpy(cost_volumn).float()
    init_dis=(cost_volunm.argmin(dim=0))
    cost_volunm=cost_volunm.numpy()
    return init_dis,cost_volunm

def hand_raw_cost_data(cost_volunm):
    # Get indices where cost_volunm equals 99.0 once and store them
    cutting_index = np.where(cost_volunm == 99.0)
    
    # Set those values to 0.0
    cost_volunm[cutting_index] = 0.0
    
    # Subtract the values in cost_volunm from 1.0
    cost_volunm = 1.0 - cost_volunm
    
    # Reset the values at cutting_index back to 0.0
    cost_volunm[cutting_index] = 0.0

    return cost_volunm

def get_cost_GT(cost_volunm,depth_map):
    GT_cost=np.zeros((cost_volunm.shape[1],cost_volunm.shape[2]))
    for y in range(depth_map.shape[0]):
            for x in range(depth_map.shape[1]):
                d=int(depth_map[y][x])
                #print(d)
                GT_cost[y][x]=cost_volunm[d][y][x]
                if d==0:
                    GT_cost[y][x]=0
    print("min_cost:"+str(np.min(GT_cost)))
    return GT_cost
def crop(cost_volumn_4,costmap,sparse_disaprity,GT,init_dim,img_name):
    for h_dim  in range(0,GT.shape[0],64):
        for w_dim in range(0,GT.shape[1],64):
            max_lenght=max(GT.shape[0],GT.shape[1])
            
            if h_dim-init_dim<0 or w_dim-init_dim<0 or h_dim>=GT.shape[0]-1 or w_dim>=GT.shape[1]-1:
                continue
            new_cost_volumn_4=cost_volumn_4[:,h_dim-init_dim:h_dim,w_dim-init_dim:w_dim]
            if new_cost_volumn_4.shape[0]==0 or new_cost_volumn_4.shape[1]!=init_dim or new_cost_volumn_4.shape[2]!=init_dim:
                print("wrong")
                continue
            
            new_GT=GT[h_dim-init_dim:h_dim,w_dim-init_dim:w_dim]
            gt_num=np.sum(new_GT>0)
            if gt_num==0:
                print("continue!!!!!!!")
                continue
            new_costmap=costmap[h_dim-init_dim:h_dim,w_dim-init_dim:w_dim]
            new_sparse_disaprity=sparse_disaprity[h_dim-init_dim:h_dim,w_dim-init_dim:w_dim]

            saving_path = "./training_patchs/"
            if not os.path.exists(saving_path):
                os.makedirs(saving_path)

            print(new_cost_volumn_4.shape)

            np.save(saving_path+img_name+"_cost_volumn_for_training_4"+"_h_"+str(h_dim-init_dim)+"_w_"+str(w_dim-init_dim),new_cost_volumn_4)
            
            np.save(saving_path+img_name+"_GT"+"_h_"+str(h_dim-init_dim)+"_w_"+str(w_dim-init_dim),new_GT)

            np.save(saving_path+img_name+"_costmap"+"_h_"+str(h_dim-init_dim)+"_w_"+str(w_dim-init_dim),new_costmap)
            np.save(saving_path+img_name+"_sparse"+"_h_"+str(h_dim-init_dim)+"_w_"+str(w_dim-init_dim),new_sparse_disaprity)
            
            #count=count+1
def handel(left_img_path,right_img_path,Pfm_img_path,file_name):
    
    if os.path.exists(right_img_path) and os.path.exists(left_img_path) and os.path.exists(Pfm_img_path):
        
        GT=Pfm2Numpy(Pfm_img_path)
        GT[GT>512]=0
        print(file_name)

        img1 = cv2.imread(left_img_path)
        print(img1.shape)
        img1=cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
        img2 = cv2.imread(right_img_path)
        img2=cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)

        img1=np.array(img1)
        img2=np.array(img2)  

        start_time = time.time()
        init_disaprity1,cost_volunm1=get_init_disaprity(511,img1,img2,35,"Census")
        init_disaprity2,cost_volunm2=get_init_disaprity(511,img1,img2,25,"Census")
        init_disaprity3,cost_volunm3=get_init_disaprity(511,img1,img2,9,"Census")
        

        init_disaprity1=init_disaprity1.numpy()
        init_disaprity2=init_disaprity2.numpy()
        init_disaprity3=init_disaprity3.numpy()
        

        init_disaprity_backup=np.copy(init_disaprity1)
        gap=np.absolute(init_disaprity1-init_disaprity2)
        init_disaprity_backup[gap!=0]=0
        gap=np.absolute(init_disaprity1-init_disaprity3)
        init_disaprity_backup[gap!=0]=0
        gap=np.absolute(init_disaprity2-init_disaprity3)
        init_disaprity_backup[gap!=0]=0


        cost_volunm1=hand_raw_cost_data(cost_volunm1)
        cost_volunm2=hand_raw_cost_data(cost_volunm2)
        cost_volunm3=hand_raw_cost_data(cost_volunm3)
        cost_volunm=(cost_volunm1+cost_volunm2+cost_volunm3)/3
        del cost_volunm1
        del cost_volunm2
        del cost_volunm3
        gc.collect()
        cost_map=get_cost_GT(cost_volunm,GT)

        crop(cost_volunm,cost_map,init_disaprity_backup,GT,128,file_name)

def worker_process(worker_id, tasks):
    # 自动获取GPU数量
    num_gpus = torch.cuda.device_count()
    
    # 使用worker_id选择GPU
    if num_gpus > 0:
        cuda.select_device(worker_id % num_gpus)  # 根据GPU数量分配
    else:
        raise RuntimeError("No GPUs available.")

    for i, left_file, right_file, pfm_file in tasks:
        print(left_file)
        print(right_file)
        handel(left_file, right_file, pfm_file, str(i))
def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process input arguments for the script.")
    parser.add_argument('--root_dir', type=str, default='./SceneFlow/', help='Root directory for input files')
    parser.add_argument('--num_processes', type=int, default=4, help='Number of processes to use')
    parser.add_argument('--images_num', type=int, default=500, help='Number of images to sample')
    args = parser.parse_args()

    root_dir = args.root_dir
    num_processes = args.num_processes
    images_num = args.images_num

    pfm_files, left_files, right_files = find_pfm_files(root_dir)
    print(len(pfm_files))
    assert len(left_files) == len(right_files) == len(pfm_files)

    selected_indices = random.sample(range(len(left_files)), images_num)

    tasks = [(i, left_files[i], right_files[i], pfm_files[i]) for i in selected_indices]
    tasks_per_worker = [tasks[i::num_processes] for i in range(num_processes)]  # 根据进程数量分配任务

    # 创建一个进程池
    with multiprocessing.Pool(num_processes) as pool:
        pool.starmap(worker_process, list(zip(range(num_processes), tasks_per_worker)))


if __name__ == '__main__':
    main()