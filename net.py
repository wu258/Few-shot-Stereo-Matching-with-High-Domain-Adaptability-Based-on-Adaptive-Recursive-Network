import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Gradient_Net(nn.Module):
  def __init__(self):
    super(Gradient_Net, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

    self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

  def forward(self, x):
    grad_x = F.conv2d(x, self.weight_x)
    grad_y = F.conv2d(x, self.weight_y)
    gradient = torch.abs(grad_x) + torch.abs(grad_y)
    return gradient

def gradient(x):
    gradient_model = Gradient_Net().to(device)
    g = gradient_model(x)
    return g


def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_dim),
        activation,)

def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, in_dim, kernel_size=(3,3,3), stride=(1,2,2), padding=1, output_padding=(0,1,1)),
        nn.InstanceNorm3d(out_dim),
        activation,)


def max_pooling_3d(in_dim,activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, in_dim, (3 , 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)),
        #nn.InstanceNorm3d(in_dim),
        activation,)
    #return nn.Conv3d(in_dim, in_dim, (3 , 2, 2), stride=(1, 2, 2), padding=(1, 0, 0))


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        #nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        )
def conv_block_2_3d_out(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, (2 , 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),

        nn.Sigmoid(),
        )
class Unet_3d(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(Unet_3d, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.GELU()
        #activation2 = nn.Softmax(dim=4)
        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d(self.num_filters,activation)
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d(self.num_filters * 2,activation)
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d(self.num_filters* 4,activation)
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d(self.num_filters* 8,activation)
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.pool_5 = max_pooling_3d(self.num_filters* 16,activation)
        
        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)
        
        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)
        
        # Output
        self.out = conv_block_2_3d_out(self.num_filters, out_dim)
    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x) # -> [1, 4, 128, 128, 128]
        #print(down_1.shape)
        pool_1 = self.pool_1(down_1) # -> [1, 4, 64, 64, 64]
        #print(pool_1.shape)
        down_2 = self.down_2(pool_1) # -> [1, 8, 64, 64, 64]
        pool_2 = self.pool_2(down_2) # -> [1, 8, 32, 32, 32]
        
        down_3 = self.down_3(pool_2) # -> [1, 16, 32, 32, 32]
        pool_3 = self.pool_3(down_3) # -> [1, 16, 16, 16, 16]
        
        down_4 = self.down_4(pool_3) # -> [1, 32, 16, 16, 16]
        pool_4 = self.pool_4(down_4) # -> [1, 32, 8, 8, 8]
        
        down_5 = self.down_5(pool_4) # -> [1, 64, 8, 8, 8]
        pool_5 = self.pool_5(down_5) # -> [1, 64, 4, 4, 4]
        
        # Bridge
        bridge = self.bridge(pool_5) # -> [1, 128, 4, 4, 4]
        
        # Up sampling

        trans_1 = self.trans_1(bridge) # -> [1, 128, 8, 8, 8]
        #print(trans_1.shape)
        concat_1 = torch.cat([trans_1, down_5], dim=1) # -> [1, 192, 8, 8, 8]
        up_1 = self.up_1(concat_1) # -> [1, 64, 8, 8, 8]
        
        trans_2 = self.trans_2(up_1) # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, down_4], dim=1) # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2) # -> [1, 32, 16, 16, 16]
        
        trans_3 = self.trans_3(up_2) # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_3], dim=1) # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3) # -> [1, 16, 32, 32, 32]
        
        trans_4 = self.trans_4(up_3) # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, down_2], dim=1) # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4) # -> [1, 8, 64, 64, 64]
        
        trans_5 = self.trans_5(up_4) # -> [1, 8, 128, 128, 128]
        concat_5 = torch.cat([trans_5, down_1], dim=1) # -> [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5) # -> [1, 4, 128, 128, 128]
        
        # Output
        out = self.out(up_5) # -> [1, 3, 128, 128, 128]
        #print(out.shape)
        return out
class Cost_aggregation_layer(nn.Module):

    def __init__(self, in_channels=2, out_channels=2, init_features=16):
        super(Cost_aggregation_layer, self).__init__()
        

        features = init_features

        self.cnn3d1=Unet_3d(in_dim=in_channels, out_dim=out_channels, num_filters=16)
        

        
    def forward(self, cost_volumn_4,real_dim):
        #print(cost_volumn_4.shape)
        #print(real_dim)
        for batch_i in range(cost_volumn_4.shape[0]):
            first_channel=cost_volumn_4[batch_i][0][0:real_dim]
            second_channel=cost_volumn_4[batch_i][1][0:real_dim]

            first_channel=torch.unsqueeze(first_channel, 0)
            second_channel=torch.unsqueeze(second_channel, 0)

            cost_volumn_4=torch.cat((first_channel,second_channel),dim=0)
            #cost_volumn_4[batch_i][1]=cost_volumn_4[batch_i][1][0:real_dim]
        
        cost_volumn=torch.unsqueeze(cost_volumn_4, 0)
        del cost_volumn_4
        #print(cost_volumn_4.shape)
        for i in range(int(math.log2(real_dim))):
            cost_volumn = self.cnn3d1(cost_volumn)
        
        #cost_volumn_4 = self.cnn3d1(cost_volumn_4)
        #print(cost_volumn_4.shape)
        #left_img=self.conv2d(lef_img)
        #print(x.shape)
        cost_volumn=torch.squeeze(cost_volumn,2)
        #print(cost_volumn_4.shape)

        #x=torch.cat((cost_volumn_4,left_img), dim=1)
        #x=self.final_conv(x)
        #print(cost_volumn_4)
        return cost_volumn,real_dim
