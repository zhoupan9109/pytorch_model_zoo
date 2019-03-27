import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),        # input data 224x224x3, output data 224x224x64
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),       # input data 224x224x64, output data 224x224x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # input data 224x224x64, output data 112x112x64

            nn.Conv2d(64, 128, 3, 1, 1),      # input data 112x112x64, output data 112x112x128
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),     # input data 112x112x128, output data 112x112x128
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # input data 112x112x128, output data 56x56x128

            nn.Conv2d(128, 256, 3, 1, 1),     # input data 56x56x128, output data 56x56x256
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),     # input data 56x56x256, output data 56x56x256
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),     # input data 56x56x256, output data 56x56x256
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # input data 56x56x256, output data 28x28x256

            nn.Conv2d(256, 512, 3, 1, 1),     # input data 28x28x256, output data 28x28x512
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),     # input data 28x28x512, output data 28x28x512
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),     # input data 28x28x512, output data 28x28x512
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # input data 28x28x512, output data 14x14x512

            nn.Conv2d(512, 512, 3, 1, 1),     # input data 14x14x512, output data 14x14x512
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),     # input data 14x14x512, output data 14x14x512
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),     # input data 14x14x512, output data 14x14x512
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                # input data 14x14x512, output data 7x7x512
        )
        self.dense = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2)
        )

    def forward(self, input_data):
        conv_out = self.conv(input_data)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)

        return out
