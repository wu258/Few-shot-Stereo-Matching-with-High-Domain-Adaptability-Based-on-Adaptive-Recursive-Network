from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from PIL import Image

import os.path
import glob
import torch
import cv2
from numba import cuda, float32

from numba import jit
from numba import prange
import numba
# Class for c
@jit(parallel=True)
def get_index(max_dis):
    index_m=np.zeros((max_dis,128,128))  
    for y in prange(128):
        for x in range(128):
            for i in range(max_dis):
                index_m[i][y][x]=i
    return index_m
class cost_aggregation_dataloader(Dataset):
    """Face Landmarks dataset."""
    
    def __init__(self,root_dir,type):
        self.type=type
        self.cost_volumn_4_list=[]
        self.cost_volumn_8_list=[]
        self.cost_volumn_16_list=[]
        self.GT_list=[]
        self.file_name_list=[]
        self.left_list=[]
        self.index_list=[]
        self.costmap_list=[]
        self.shift_list=[]
        self.cosRight_list=[]
        self.sparse_list=[]
        self.index_512=get_index(512)
        self.index_512=self.index_512/512
        #self.index_512=np.load("index_512.npy")


        self.index_256=get_index(256)
        self.index_256=self.index_256/256
        #np.save("index_256", self.index_256)
        #self.index_256=np.load("index_256.npy")

        self.index_128=get_index(128)
        self.index_128=self.index_128/128
        #np.save("index_128", self.index_128)
        #self.index_128=np.load("index_128.npy")

        self.index_64=get_index(64)
        self.index_64=self.index_64/64
        #np.save("index_64", self.index_64)
        #self.index_64=np.load("index_64.npy")
        self.index_32=get_index(32)
        self.index_32=self.index_32/32
        #np.save("index_32", self.index_32)
        #self.index_32=np.load("index_32.npy")
        for filename in os.listdir(root_dir):
            file_path=root_dir+filename
            
            file_type=os.path.splitext(file_path)[-1]
                
            if ".npy"==file_type:
                #png_path=os.path.splitext(file_path)[0]+".npy"
                #print(file_path.split('_')[-5])
                if file_path.split('_')[-5]=="GT":
                    #print('hahaha')
                    GT_path=file_path
                    self.file_name_list.append(filename)
                    left_img_path=file_path.replace('GT','left')
                    #print(GT_path)
                        
                    self.left_list.append(left_img_path)
                       
                    self.GT_list.append(GT_path)
                    cost_volumn_4=file_path.replace('GT','cost_volumn_for_training_4')
                    #index_volumn_path=file_path.replace('GT','index')

                    costmap_path=file_path.replace('GT','costmap')

                    self.costmap_list.append(costmap_path)
                    #self.index_list.append(index_volumn_path)
                        
                    self.cost_volumn_4_list.append(cost_volumn_4)
                        
                    shift_volumn_path=file_path.replace('GT','shift')
                    self.shift_list.append(shift_volumn_path)

                    sparse_disaprity_path=file_path.replace('GT','sparse')
                    self.sparse_list.append(sparse_disaprity_path)

                        
    def __len__(self):
        return len(self.GT_list)

    def __getitem__(self,idx):

        threshold=0.96
        transform1 = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])
        cost_volumn_4=np.load(self.cost_volumn_4_list[idx])

        index_m=None
        #index_m=np.rot90(index_m,k,(1,2))
        init_dis_patch=np.load(self.sparse_list[idx])
        real_dim=-1
        non_zero_count=np.sum(init_dis_patch>0)
        zero_count=np.sum(init_dis_patch==0)
        within_32_percentage=0
        within_64_percentage=0
        within_128_percentage=0
        within_256_percentage=0
        within_512_percentage=0
        if non_zero_count>0:
            within_32_percentage=(np.sum(init_dis_patch<32)-zero_count)/non_zero_count
            within_64_percentage=(np.sum(init_dis_patch<64)-zero_count)/non_zero_count
            within_128_percentage=(np.sum(init_dis_patch<128)-zero_count)/non_zero_count
            within_256_percentage=(np.sum(init_dis_patch<256)-zero_count)/non_zero_count
            within_512_percentage=(np.sum(init_dis_patch<512)-zero_count)/non_zero_count
        #count=count+1
        
        if within_32_percentage>threshold:
            real_dim=32
            index_m=self.index_32
        elif within_64_percentage>threshold:
            real_dim=64
            index_m=self.index_64
        
        elif within_128_percentage>threshold:
            real_dim=128
            index_m=self.index_128

        elif within_256_percentage>threshold:
            real_dim=256
            index_m=self.index_256

        elif within_512_percentage>threshold:
            real_dim=512
            index_m=self.index_512
        if real_dim==-1:
            real_dim=512
            index_m=self.index_512
        GT=np.load(self.GT_list[idx])
        max_dis=np.max(GT)
        '''
        rot_seed=0
        if self.type=="train":
            #rot_seed=torch.randint(0,4,(1,))
            rot_seed=0
        GT=np.rot90(GT,rot_seed,(0,1))

        cost_volumn_4=np.rot90(cost_volumn_4,rot_seed,(1,2))
        '''
        k=0

        result=np.zeros((512,cost_volumn_4.shape[1],cost_volumn_4.shape[2]))
        cost_volumn_4=cost_volumn_4[0:result.shape[0]]
        result[0+k:cost_volumn_4.shape[0]+k]=cost_volumn_4[0:cost_volumn_4.shape[0]]

        index_result=np.zeros((512,cost_volumn_4.shape[1],cost_volumn_4.shape[2]))
        index_result[0+k:index_m.shape[0]+k]=index_m[0:index_m.shape[0]]
        index_m=index_result
        cost_volumn_4=result

        GT[GT>real_dim]=0
        GT[GT>real_dim]=0
        GT=GT+k
        #GT=np.rot90(GT,k,(0,1))
        GT_cost=np.load(self.costmap_list[idx])
        GT_cost[GT>real_dim]=0
        GT_cost=np.reshape(GT_cost,(1,GT_cost.shape[0],GT_cost.shape[1]))
        index_m=np.reshape(index_m,(1,index_m.shape[0],index_m.shape[1],index_m.shape[2]))
        GT=np.reshape(GT,(1,GT.shape[0],GT.shape[1]))
        
        
        cost_volumn_4=np.reshape(cost_volumn_4,(1,cost_volumn_4.shape[0],cost_volumn_4.shape[1],cost_volumn_4.shape[2]))

        
        #left_img=np.rot90(left_img,rot_seed,(0,1))





        
       
        cost_volumn_4=torch.from_numpy(cost_volumn_4.copy()).float()
        
        index_m=torch.from_numpy(index_m.copy()).float()
        GT_cost=torch.from_numpy(GT_cost.copy()).float()

        GT=torch.from_numpy(GT.copy()).float()
        GT=GT/real_dim

        #print(cost_volumn_4.shape)
        #print(index_m.shape)
        cost_volumn_4=torch.cat((cost_volumn_4,index_m), dim=0)
        GT=torch.cat((GT_cost,GT), dim=0)

        sample = {'cost_volumn_4': cost_volumn_4,'GT':GT,"real_dim":real_dim}
        #print("finish!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return sample



def main():
    # Use a GPU if available, as it should be faster.
    Dataset=cost_aggregation_dataloader("D:/MiddEval3-H_new/cense_data_tranining_patch/")
    

if __name__ == '__main__':
    main()



