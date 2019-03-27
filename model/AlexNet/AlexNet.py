import torch.nn as nn
import torch.nn.functional as f

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 0),       # input shape 227x227x3, output shape 55x55x96
            nn.ReLU(),                        #
            nn.MaxPool2d(3, 2)                # input shape 55x55x96, out shape 27x27x96
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),      # input shape 27x27x96, output shape 27x27x256
            nn.ReLU(),                        #
            nn.MaxPool2d(3, 2)                # input shape 27x27x256, out shape 13x13x256
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),     # input shape 13x13x256, out shape 13x13x384
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),     # input shape 13x13x384, out shape 13x13x384
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 265, 3, 1, 1),     # input shape 13x13x384, out shape 13x13x256
            nn.ReLU(),
            nn.MaxPool2d(3, 2)                # input shape 13x13x256, out shape 6x6x256
        )
        self.dense = nn.Sequential(
            nn.Linear(9540, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 2)
        )

    def forward(self, input_data):
        conv1_out = self.conv1(input_data)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        res = conv5_out.view(conv5_out.size(0), -1)
        out = self.dense(res)
        return out

