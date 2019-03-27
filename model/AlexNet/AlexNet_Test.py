import torch
import AlexNet
import numpy as np
import cv2
from torch.autograd import Variable

resizeH = 227
resizeW = 227

pth_file = "model.pth"  # model_file
image_path = "313.jpg"  #test_image

classes = ["dog", "cat"]


def AlexNet_Test(pth_file, image_path):
    model = AlexNet.AlexNet()
    model.load_state_dict(torch.load(pth_file))
    model.eval()

    image = cv2.imread(image_path)
    image = cv2.resize(image, (resizeH, resizeW), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 1, 0))
    image = torch.from_numpy(image).unsqueeze(0)
    print(image.size())

    if torch.cuda.is_available():
        model = model.cuda()
        image = image.cuda()

    # out = model(Variable(image))
    out = model(image)
    pre = torch.max(out, 1)[1].cpu()
    pre = pre.numpy()
    pre_class = int(pre[0])
    print(classes[pre_class])
    # print("prdict is {:.s}".format(classes[pre[0]]))


if __name__ == "__main__":
    AlexNet_Test(pth_file, image_path)
