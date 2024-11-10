import os
import numpy as np
import torch
import torch.nn as tnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as topti
from cost_aggregation_dataloader import cost_aggregation_dataloader
import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import cv2
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from PIL import Image
from tqdm import tqdm
from math import exp
import math
from net import Cost_aggregation_layer, gradient
#torch.cuda.set_device(1)
# Class for creating the neural network.
torch.backends.cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patch_size=128
max_dis=512



def lossFunc(input,labels):
    loss_fn = torch.nn.L1Loss()
    loss_fn2 = torch.nn.MSELoss()
    
    input = input.clone().cuda()
    labels = labels.clone().cuda()

    temp_input1=input[:,0,:,:,].clone()
    temp_input2=input[:,1,:,:,].clone()
    temp_label1=labels[:,0,:,:,].clone()
    temp_label2=labels[:,1,:,:,].clone()

    index=np.where(temp_label2.cpu() == 0)  # Add .cpu() before converting to numpy
    temp_input1[index]=0
    temp_input2[index]=0
    temp_label1[index]=0
    temp_label2[index]=0
    
    index=np.where(temp_label2.cpu() == 0.0)  # Add .cpu() before converting to numpy
    temp_input1[index]=0
    temp_input2[index]=0
    temp_label1[index]=0
    temp_label2[index]=0
    
    input[:,0,:,:,]=temp_input1
    input[:,1,:,:,]=temp_input2
    labels[:,0,:,:,]=temp_label1
    labels[:,1,:,:,]=temp_label2
    


    output_gradient_input=torch.unsqueeze(temp_input2, 1)
    output_gradient=gradient(output_gradient_input)

    return loss_fn2(input[:,0,:,:,], labels[:,0,:,:,])+loss_fn(input[:,1,:,:,], labels[:,1,:,:,])




def get_cost_volumn_path(path):
    cost_volumn_list=[]
    GT_list=[]
    for filename in os.listdir(path):
        file_path=path+filename

        file_type=os.path.splitext(file_path)[-1]
        if ".npy"==file_type:
            #png_path=os.path.splitext(file_path)[0]+".npy"
            if file_path.split('_')[-1]=="training.npy":
                GT_path=file_path.replace('cost_volumn_for_training','GT')
                print(GT_path)
                cost_volumn=np.load(file_path)
                GT=np.load(GT_path)
                cost_volumn_list.append(cost_volumn)
                GT_list.append(GT)


    return cost_volumn_list,GT_list

def test(net,Dataset):
    #test_size = len(Dataset) - train_size
    #train_dataset, test_dataset = torch.utils.data.random_split(Dataset, [train_size, test_size])
    dataloader = DataLoader(Dataset, batch_size=4,
                        shuffle=False, num_workers=0)

    net=net.eval()
    with torch.no_grad():
        for epoch in range(1):
            net=net.eval()
            running_loss = 0
            sum_loss=0
            acc=0
            count=0
            #net=net.train()
            for i, batch in enumerate(tqdm(dataloader)):
           
              
                label=batch['GT']
                #print(label)
                with torch.cuda.amp.autocast():
                    output,real_dim = net(batch["cost_volumn_4"],batch["real_dim"])
                    loss = lossFunc(output, label)
            
                temp_labal=label.cpu().numpy()
               
                temp_output1=output.cpu().detach().numpy()

                temp_output1=temp_output1[:,1,:,:,]
                temp_labal=temp_labal[:,1,:,:,]

                real_dim=real_dim.cpu().numpy()
                for batch_dim in range(temp_output1.shape[0]):
                    temp_output1[batch_dim]=temp_output1[batch_dim]*real_dim[batch_dim]
                    temp_labal[batch_dim]=temp_labal[batch_dim]*real_dim[batch_dim]

                temp_output1[temp_labal<=0]=0
                acc1=np.absolute(temp_output1-temp_labal)
                gt_num=np.sum(temp_labal>=0)
                zero_num=np.sum(temp_labal==0)
                correct_num=np.sum(acc1 <=1)
                acc1=0
                if gt_num>0:

                    acc1=((correct_num-zero_num)/gt_num)
                else:
                    acc1=0
                acc=acc+acc1
                count=count+1

                running_loss += loss.item()
                sum_loss+=loss.item()

            print("eval_loss:"+str(sum_loss)+"eval_acc:"+str(acc/count))
            return sum_loss,acc

def main():
    torch.cuda.empty_cache()
    # Use a GPU if available, as it should be faster.
    print("running!!!!!!!!!")

    Dataset=cost_aggregation_dataloader("./training_patchs/","train")
    testing_Dataset=cost_aggregation_dataloader("./testing_patch/","test")
    dataloader = DataLoader(Dataset, batch_size=4,
                        shuffle=True, num_workers=0)
    print(torch.__version__)  #注意是双下划线

    print("Using device: " + str(device))
    model = Cost_aggregation_layer()

    scaler = torch.cuda.amp.GradScaler()

    model = nn.DataParallel(model)
    net = model.to(device)

    gobal_loss=999999999999
    gobal_test_loss =99999999999999
    gobal_test_acc =0
    optimiser = topti.RAdam(net.parameters(), lr=0.0001)  # Minimise the loss using the Adam algorithm.

    gobal_test_loss,gobal_test_acc=test(net,testing_Dataset)
    for epoch in range(10000000):
        running_loss = 0
        sum_loss=0
        net=net.train()
        acc=0 
        count=0
        
        for i, batch in enumerate(tqdm(dataloader)):

            optimiser.zero_grad()
            label=batch['GT']
            loss=0
            with torch.cuda.amp.autocast():
                output,real_dim = net(batch["cost_volumn_4"],batch["real_dim"])
                loss = lossFunc(output, label)
           
            scaler.scale(loss).backward()
            
            scaler.step(optimiser)
            scaler.update()

            temp_labal=label.cpu().numpy()

            temp_output1=output.cpu().detach().numpy()
            temp_output1=temp_output1[:,1,:,:,]
            temp_labal=temp_labal[:,1,:,:,]
            real_dim=real_dim.cpu().numpy()
            for batch_dim in range(temp_output1.shape[0]):
                temp_output1[batch_dim]=temp_output1[batch_dim]*real_dim[batch_dim]
                temp_labal[batch_dim]=temp_labal[batch_dim]*real_dim[batch_dim]

            acc1=np.absolute(temp_output1-temp_labal)
            gt_num=np.sum(temp_labal>0)
            zero_num=np.sum(temp_labal==0)
            correct_num=np.sum(acc1 <=1)
            acc1=0
            if gt_num>0:

                acc1=((correct_num-zero_num)/gt_num)
            else:
                acc1=0
            acc=acc+acc1
            count=count+1

            running_loss += loss.item()
            sum_loss+=loss.item()

        test_loss,test_acc=test(net,testing_Dataset)

        gobal_loss=sum_loss
        torch.save(net.state_dict(), "./good_model/model.pth")
        print("sum_epoch_loss:"+str(sum_loss)+"train_acc:"+str(acc/count))
       
    num_correct = 0

    # Save mode
    #torch.save(net.state_dict(), "C:/Users/Arthur/source/repos/small_psf_layer/model-weel_trained/cost_aggregation.pth")
    print("Saved model")
if __name__ == '__main__':
    main()

