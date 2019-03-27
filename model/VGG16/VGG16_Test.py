import torch
import cv2
import VGG16
import numpy as np
import os

model_path = ".//model//_iter_60.pth"
image_path = ""
test_reuslt = ""

classes = ["dog", "cat"]


def VGG16_Test(model, image_path, image_name, test_result):
    image_data = os.path.join(image_path, image_name)
    print(image_data)
    image_origin = cv2.imread(image_data)
    image = cv2.resize(image_origin, (224, 244))
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 1, 0))
    image = torch.from_numpy(image).unsqueeze(0)

    if torch.cuda.is_available():
        image = image.cuda()

    out = model(image)
    pre = torch.max(out, 1)[1].cpu()
    pre = pre.numpy()
    print(classes[int(pre[0])])
    cv2.imwrite(test_result + str(classes[int(pre[0])]) + "_" + image_name, image_origin)


if __name__ == "__main__":
    model = VGG16.VGG16()
    model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        model = model.cuda()

    data_list = os.listdir(image_path)
    for data in data_list:
        VGG16_Test(model, image_path, data, test_reuslt)
