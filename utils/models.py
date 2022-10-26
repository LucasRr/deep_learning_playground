import torch
from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28*28, 10)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear(x)
        return logits



class MLP_2layers(nn.Module):
    def __init__(self):
        super(MLP_2layers, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        out = self.linear_relu_stack(x)
        return out 


class MLP_4layers(nn.Module):
    def __init__(self):
        super(MLP_4layers, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        out = self.linear_relu_stack(x)
        return out



class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5,5), stride=1, padding=2)   # 6x32x32
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)                            # 6x16x16
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=1, padding=0)  # 16x12x12
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)                            # 16x6x6
        self.dropout2 = nn.Dropout(0.2)
                
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(16*6*6, 120)
        self.dropout3 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(120, 84)
        self.dropout4 = nn.Dropout(0.2)
        self.dense3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(self.maxpool1(x))
        
        x = F.relu(self.conv2(x))
        x = self.dropout2(self.maxpool2(x))
                
        x = self.flatten(x)
        x = self.dropout3(F.relu(self.dense1(x)))
        x = self.dropout4(F.relu(self.dense2(x)))
        x = self.dense3(x)
        return x



class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(7,7), stride=1, padding=3)   # 6x32x32
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)                            # 6x16x16
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=1, padding=2)  # 16x16x16
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)                            # 16x8x8
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv3a = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=1, padding=1)  # 32x8x8
        self.conv3b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=1, padding=1)  # 32x8x8
        self.conv3c = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=1, padding=1)  # 32x8x8

        self.maxpool3 = nn.MaxPool2d((2, 2), stride=2, padding=0)                                         # 32x4x4                                              # 32x4x4
        
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(32*4*4, 120)
        self.dropout3 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(120, 84)
        self.dropout4 = nn.Dropout(0.2)
        self.dense3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(self.maxpool1(x))
        
        x = F.relu(self.conv2(x))
        x = self.dropout2(self.maxpool2(x))
        
        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = F.relu(self.conv3c(x))
        x = self.maxpool3(x)
                        
        x = self.flatten(x)
        x = self.dropout3(F.relu(self.dense1(x)))
        x = self.dropout4(F.relu(self.dense2(x)))
        x = self.dense3(x)
        return x

class VGG1(nn.Module):
    """ small VGG network with 1 convolutional block """
    def __init__(self):
        super(VGG1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (3, 3), stride=1, padding=1)           # 16x32x32
        self.conv2 = nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)          # 16x32x32
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0) # 16x16x16
        self.dropout1 = nn.Dropout(0.5)
        
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(16*16*16, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = self.dense2(x)
        return x



class VGG2(nn.Module):
    """ VGG network with 2 convolutional blocks """
    def __init__(self):
        super(VGG2, self).__init__()
        # Block 1:
        self.conv1 = nn.Conv2d(3, 16, (3, 3), stride=1, padding=1)           # 16x32x32
        self.conv2 = nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)          # 16x32x32
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0) # 16x16x16
        self.dropout1 = nn.Dropout(0.5)
    
        # Block 2:
        self.conv3 = nn.Conv2d(16, 32, (3, 3), stride=1, padding=1)          # 32x16x16
        self.conv4 = nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)          # 32x16x16
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0) # 32x8x8
        self.dropout2 = nn.Dropout(0.5)
        
        # Classifier:
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(32*8*8, 512)
        self.dropout3 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = self.dense2(x)
        return x
    


class VGG3(nn.Module):
    """ VGG with 3 convolutional blocks """
    def __init__(self):
        super(VGG3, self).__init__()
        # Block 1:
        self.conv11 = nn.Conv2d(3, 16, (3, 3), stride=1, padding=1)           # 16x32x32
        self.conv12 = nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)          # 16x32x32
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)  # 16x16x16
        self.dropout1 = nn.Dropout(0.5)
    
        # Block 2:
        self.conv21 = nn.Conv2d(16, 32, (3, 3), stride=1, padding=1)          # 32x16x16
        self.conv22 = nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)          # 32x16x16
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)  # 32x8x8
        self.dropout2 = nn.Dropout(0.5)
        
        # Block 3:
        self.conv31 = nn.Conv2d(32, 64, (3, 3), stride=1, padding=1)          # 64x8x8
        self.conv32 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)          # 64x8x8
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)  # 64x4x4
        self.dropout3 = nn.Dropout(0.5)
        
        # Classifier:
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(64*4*4, 512)
        self.dropoutd = nn.Dropout(0.5)
        self.dense2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.maxpool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = self.maxpool3(x)
        x = self.dropout3(x)
        
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dropoutd(x)
        x = self.dense2(x)
        return x
    


class ResNet_plain(nn.Module):
    def __init__(self):
        super(ResNet_plain, self).__init__()
                                                                              # input: 3x32x32
        self.conv0 = nn.Conv2d(3, 16, (3, 3), stride=1, padding=1)
        
        # Block 1:
        self.conv11 = nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)          # 16x32x32
        self.conv12 = nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)          # 16x32x32
        self.conv13 = nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)          # 16x32x32
        self.conv14 = nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)          # 16x32x32
        self.conv15 = nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)          # 16x32x32
        self.conv16 = nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)          # 16x32x32
        
        # Block 2:
        self.conv21 = nn.Conv2d(16, 32, (3, 3), stride=2, padding=1)          # 32x16x16
        self.conv22 = nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)          # 32x16x16
        self.conv23 = nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)          # 32x16x16
        self.conv24 = nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)          # 32x16x16
        self.conv25 = nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)          # 32x16x16
        self.conv26 = nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)          # 32x16x16
        
        # Block 3:
        self.conv31 = nn.Conv2d(32, 64, (3, 3), stride=2, padding=1)          # 64x8x8
        self.conv32 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)          # 64x8x8
        self.conv33 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)          # 64x8x8
        self.conv34 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)          # 64x8x8
        self.conv35 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)          # 64x8x8
        self.conv36 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)          # 64x8x8
        
        self.avgpool = nn.AvgPool2d((8, 8))                                   # 64x1x1
        
        # Classifier:
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(64, 10)
        
    def forward(self, x):
        
        x0 = self.conv0(x)
        
        x11 = F.relu(self.conv11(x0))
        x12 = F.relu(self.conv12(x11))
        x13 = F.relu(self.conv13(x12))
        x14 = F.relu(self.conv14(x13))
        x15 = F.relu(self.conv15(x14))
        x16 = F.relu(self.conv16(x15))
        
        x21 = F.relu(self.conv21(x16))
        x22 = F.relu(self.conv22(x21))
        x23 = F.relu(self.conv23(x22))
        x24 = F.relu(self.conv24(x23))
        x25 = F.relu(self.conv25(x24))
        x26 = F.relu(self.conv26(x25))
        
        x31 = F.relu(self.conv31(x26))
        x32 = F.relu(self.conv32(x31))
        x33 = F.relu(self.conv33(x32))
        x34 = F.relu(self.conv34(x33))
        x35 = F.relu(self.conv35(x34))
        x36 = F.relu(self.conv36(x35))
        
        x = self.avgpool(x36)
        x = self.flatten(x)
        x = self.dense(x)
        return x
    


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
                                                                              # input: 3x32x32
        self.conv0 = nn.Conv2d(3, 16, (3, 3), stride=1, padding=1)
                
        # Block 1:
        self.conv11 = nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)          # 16x32x32
        self.conv12 = nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)          # 16x32x32
        self.conv13 = nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)          # 16x32x32
        self.conv14 = nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)          # 16x32x32
        self.conv15 = nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)          # 16x32x32
        self.conv16 = nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)          # 16x32x32
        
        self.res1 = nn.Conv2d(16, 32, (1, 1), stride=2, padding=0, bias=False)
        
        # Block 2:
        self.conv21 = nn.Conv2d(16, 32, (3, 3), stride=2, padding=1)          # 32x16x16
        self.conv22 = nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)          # 32x16x16
        self.conv23 = nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)          # 32x16x16
        self.conv24 = nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)          # 32x16x16
        self.conv25 = nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)          # 32x16x16
        self.conv26 = nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)          # 32x16x16
        
        self.res2 = nn.Conv2d(32, 64, (1, 1), stride=2, padding=0, bias=False)

        # Block 3:
        self.conv31 = nn.Conv2d(32, 64, (3, 3), stride=2, padding=1)          # 64x8x8
        self.conv32 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)          # 64x8x8
        self.conv33 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)          # 64x8x8
        self.conv34 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)          # 64x8x8
        self.conv35 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)          # 64x8x8
        self.conv36 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)          # 64x8x8
        
        self.maxpool = nn.MaxPool2d((8, 8))                                   # 64x1x1
        
        # Classifier:
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(64, 10)
        
    def forward(self, x):
        
        x0 = self.conv0(x)
        
        x11 = F.relu(self.conv11(x0))
        x12 = F.relu(self.conv12(x11)+x0)   # residual connection
        x13 = F.relu(self.conv13(x12))
        x14 = F.relu(self.conv14(x13)+x12)  # residual connection
        x15 = F.relu(self.conv15(x14))
        x16 = F.relu(self.conv16(x15)+x14)  # residual connection
        
        x21 = F.relu(self.conv21(x16))
        x22 = F.relu(self.conv22(x21)+self.res1(x16))  # res. connection with downscaling and increased depth
        x23 = F.relu(self.conv23(x22))
        x24 = F.relu(self.conv24(x23)+x22)  # residual connection
        x25 = F.relu(self.conv25(x24))
        x26 = F.relu(self.conv26(x25)+x24)  # residual connection
        
        x31 = F.relu(self.conv31(x26))
        x32 = F.relu(self.conv32(x31)+self.res2(x26))  # res. connection with downscaling and increased depth
        x33 = F.relu(self.conv33(x32))
        x34 = F.relu(self.conv34(x33)+x32)  # residual connection
        x35 = F.relu(self.conv35(x34))
        x36 = F.relu(self.conv36(x35)+x34)  # residual connection
        
        x = self.maxpool(x36)
        x = self.flatten(x)
        x = self.dense(x)
        return x
    
    

class Inception(nn.Module):
    
    def __init__(self):
        super(Inception, self).__init__()
        
        # stem:
        self.conv0 = nn.Conv2d(3, 16, (3, 3), padding=1, stride=1)           # 16x32x32
        self.maxpool0 = nn.MaxPool2d((3, 3), padding=1, stride=2)            # 16x16x16
        
        # Inception block 1:
        self.conv1a = nn.Conv2d(16, 32, (1, 1), padding=0, stride=1)         # 32x16x16
        self.conv1b = nn.Conv2d(16, 32, (3, 3), padding=1, stride=1)         # 32x16x16
        self.conv1c = nn.Conv2d(16, 32, (5, 5), padding=2, stride=1)         # 32x16x16
        self.maxpool1d = nn.MaxPool2d((3, 3), padding=1, stride=1)           # 16x16x16
                                                                        # --> concat: 112x16x16
            
        self.maxpool1 = nn.MaxPool2d((3, 3), padding=1, stride=2)            # 112x8x8
        
        # Inception block 2:
        self.conv2a = nn.Conv2d(112, 64, (1, 1), padding=0, stride=1)        # 64x8x8
        self.conv2b = nn.Conv2d(112, 64, (3, 3), padding=1, stride=1)        # 64x8x8
        self.conv2c = nn.Conv2d(112, 64, (5, 5), padding=2, stride=1)        # 64x8x8
        self.maxpool2d = nn.MaxPool2d((3, 3), padding=1, stride=1)           # 112x8x8
                                                                        # --> concat: 304x8x8  
        
        self.maxpool2 = nn.MaxPool2d((3, 3), padding=1, stride=2)            # 304x4x4

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(304*4*4, 256)
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 10)
        
    def forward(self, x):
        
        x = F.relu(self.conv0(x))
        x = self.maxpool0(x)
        
        x1a = F.relu(self.conv1a(x))
        x1b = F.relu(self.conv1b(x))
        x1c = F.relu(self.conv1c(x))
        x1d = self.maxpool1d(x)
        x = torch.cat([x1a, x1b, x1c, x1d], dim=1)
        x = self.maxpool1(x)
        
        x2a = F.relu(self.conv2a(x))
        x2b = F.relu(self.conv2b(x))
        x2c = F.relu(self.conv2c(x))
        x2d = self.maxpool2d(x)
        x = torch.cat([x2a, x2b, x2c, x2d], dim=1)
        x = self.maxpool2(x)
        
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        
        return x
            


class Inception2(nn.Module):
    ''' Inception with dimension reduction layers '''
    def __init__(self):
        super(Inception2, self).__init__()
        
        # stem:
        self.conv0 = nn.Conv2d(3, 16, (3, 3), padding=1, stride=1)           # 16x32x32
        self.maxpool0 = nn.MaxPool2d((3, 3), padding=1, stride=2)            # 16x16x16
        
        # Inception block 1:
        self.conv1a = nn.Conv2d(16, 32, (1, 1), padding=0, stride=1)         # 32x16x16
        self.conv1b = nn.Conv2d(16, 32, (3, 3), padding=1, stride=1)         # 32x16x16
        self.conv1c = nn.Conv2d(16, 16, (5, 5), padding=2, stride=1)         # 16x16x16
        self.maxpool1d = nn.MaxPool2d((3, 3), padding=1, stride=1)           # 16x16x16
                                                                        # --> concat: 96x16x16
            
        self.maxpool1 = nn.MaxPool2d((3, 3), padding=1, stride=2)            # 96x8x8
        
        
        # Inception block 2 w/ reduced dimensions:
        self.conv2a        = nn.Conv2d(96, 64, (1, 1), padding=0, stride=1)   # 64x8x8
        self.conv2b_red    = nn.Conv2d(96, 64, (1, 1), padding=0, stride=1)
        self.conv2b        = nn.Conv2d(64, 128, (3, 3), padding=1, stride=1)  # 128x8x8
        self.conv2c_red    = nn.Conv2d(96, 16, (1, 1), padding=0, stride=1)
        self.conv2c        = nn.Conv2d(16, 32, (5, 5), padding=2, stride=1)   # 32x8x8
        self.maxpool2d     = nn.MaxPool2d((3, 3), padding=1, stride=1)           
        self.maxpool2d_red = nn.Conv2d(96, 64, (1, 1), padding=0, stride=1)   # 64x8x8
                                                                        # --> concat: 288x8x8  
        
        self.maxpool2 = nn.MaxPool2d((3, 3), padding=1, stride=2)             # 288x4x4
        
        
        # Inception block 3 w/ reduced dimensions:
        self.conv3a        = nn.Conv2d(288, 128, (1, 1), padding=0, stride=1)  # 128x4x4
        self.conv3b_red    = nn.Conv2d(288, 128, (1, 1), padding=0, stride=1)
        self.conv3b        = nn.Conv2d(128, 256, (3, 3), padding=1, stride=1)  # 256x4x4
        self.conv3c_red    = nn.Conv2d(288, 32, (1, 1), padding=0, stride=1)
        self.conv3c        = nn.Conv2d(32, 32, (5, 5), padding=2, stride=1)    # 32x4x4
        self.maxpool3d     = nn.MaxPool2d((3, 3), padding=1, stride=1)           
        self.maxpool3d_red = nn.Conv2d(288, 128, (1, 1), padding=0, stride=1)  # 128x4x4
                                                                        # --> concat: 544x4x4  
        
        self.maxpool3 = nn.MaxPool2d((3, 3), padding=1, stride=2)             # 544x2x2    
        
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(544*2*2, 256)
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 10)
        
    def forward(self, x):
        
        x = F.relu(self.conv0(x))
        x = self.maxpool0(x)
        
        # Inception block 1:
        x1a = F.relu(self.conv1a(x))
        x1b = F.relu(self.conv1b(x))
        x1c = F.relu(self.conv1c(x))
        x1d = self.maxpool1d(x)
        x = torch.cat([x1a, x1b, x1c, x1d], dim=1)
        x = self.maxpool1(x)
        
        # Inception block 2 w/ reduced dimensions:
        x2a = F.relu(self.conv2a(x))
        x2b_red = F.relu(self.conv2b_red(x))
        x2b = F.relu(self.conv2b(x2b_red))
        x2c_red = F.relu(self.conv2c_red(x))
        x2c = F.relu(self.conv2c(x2c_red))
        x2d = self.maxpool2d(x)
        x2d_red = F.relu(self.maxpool2d_red(x2d))
        x = torch.cat([x2a, x2b, x2c, x2d_red], dim=1)
        x = self.maxpool2(x)
        
        # Inception block 3 w/ reduced dimensions:
        x3a = F.relu(self.conv3a(x))
        x3b_red = F.relu(self.conv3b_red(x))
        x3b = F.relu(self.conv3b(x3b_red))
        x3c_red = F.relu(self.conv3c_red(x))
        x3c = F.relu(self.conv3c(x3c_red))
        x3d = self.maxpool2d(x)
        x3d_red = F.relu(self.maxpool3d_red(x3d))
        x = torch.cat([x3a, x3b, x3c, x3d_red], dim=1)
        x = self.maxpool3(x)
        
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        
        return x
            

def number_of_parameters(model):
    return sum(torch.numel(p) for p in model.parameters())