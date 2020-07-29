import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from model import create_model

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

Color = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128]]

def decoder(pred):
    contain1 = pred[:,:,4].unsqueeze(2)
    contain2 = pred[:,:,9].unsqueeze(2)
    contain = torch.cat((contain1,contain2),2)
    # print(contain.shape)

    mask1 = contain > 0.1 #大于阈值
    mask2 = (contain == contain.max()) #可能存在最大值小于0.1的情况
    mask = (mask1 + mask2).gt(0)

    cell_size = 1. / 14
    boxes = []
    class_indexs = []
    probs = []
    for i in range(14):
        for j in range(14):
            for b in range(2):
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i, j, b] == 1:
                    box = pred[i, j, b*5:b*5+4]
                    contain_prob = pred[i, j, b*5+4]
                    xy = torch.FloatTensor([j, i]) * cell_size
                    '''
                    这里特征图的像素数为1，格子的像素数为1/14，把1和1/14当做像素数，就很容易理解了
                    预测的中心点是在某个格子内，相对于格子左上角的偏差，是0到1范围的百分比
                    这个百分比乘以格子的尺寸，就变成相对于格子左上角的像素偏差
                    这里格子的像素宽度是小于1的，所以可能不好理解，假设格子的尺寸是10像素，预测出来是(0.3, 0,6)，则相对于格子左上角，偏差是(3, 6)像素
                    然后再加上格子左上角相对于整个特征图左上角的偏差，就得到了预测中心点相对于左上角的像素偏差
                    这个像素偏差除以特征图的像素数1，就是中心点相对于原图的偏差百分比
                    '''
                    box[0:2] = box[0:2] * cell_size + xy
                    box_xy = torch.FloatTensor(box.size())
                    box_xy[0:2] = box[0:2] - 0.5 * box[2:]
                    box_xy[2:] = box[0:2] + 0.5 * box[2:]
                    max_prob, class_index = torch.max(pred[i, j, 10:], 0)
                    if contain_prob * max_prob > 0.05:
                        boxes.append(box_xy.cpu().detach().numpy())
                        class_indexs.append(int(class_index.cpu().detach().numpy()))
                        probs.append(float((contain_prob * max_prob).cpu().detach().numpy()))
    # print(boxes)
    return boxes, class_indexs, probs

def predict(model, img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (448, 448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = (123, 117, 104) #RGB
    img = img - np.array(mean, dtype=np.float32)
    transform = transforms.Compose([transforms.ToTensor(),]) #需要看下都做了什么处理
    img = transform(img)
    img = img.unsqueeze(0).cuda()
    # print(img.shape)

    pred = model(img).cpu().squeeze()
    # print(pred.shape)
    return decoder(pred)

if __name__ == '__main__':
    model = create_model('resnet50')()
    model = model.cuda()
    model.eval()

    state_dict_ok = torch.load('./yolo_v1_epoch40.pth')
    # state_dict_ok = torch.load('./yolo.pth')
    state_dict = model.state_dict()
    for key, val in state_dict_ok.items():
        key = key[7:]
        # print(key, key2)
        state_dict[key] = val
    model.load_state_dict(state_dict)

    img_name = './images/dog.jpg'
    img_path = os.path.join(os.path.abspath('.'), img_name)
    boxes, class_indexs, probs = predict(model, img_path)
    # print(boxes)
    # print(class_indexs)
    # print(probs)

    img = cv2.imread(img_name)
    h, w, _ = img.shape
    for box, class_index, prob in zip(boxes, class_indexs, probs):
        print(box, class_index, prob)
        name = VOC_CLASSES[class_index]
        color = Color[class_index]
        left_up = (int(box[0]*w), int(box[1]*h))
        right_bottom = (int(box[2]*w), int(box[3]*h))
        cv2.rectangle(img, left_up, right_bottom, color, 2)
        # label = class_name+str(round(prob,2))
        # text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        # p1 = (left_up[0], left_up[1]- text_size[1])
        # cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        # cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

    cv2.imwrite('result.jpg', img)
