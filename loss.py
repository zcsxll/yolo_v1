import torch
import numpy as np

class YoloLoss(torch.nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()

    def compute_iou(self, boxes1, boxes2):
        '''
        boxes1中含有N个box，boxes2中有M个box，都是x1, y1, x2, y2的格式
        输出的iou是N*M个，即boxes1中的每个box都和boxes2中的每个box做iou
        '''
        N = boxes1.shape[0]
        M = boxes2.shape[0]
        
        lt = torch.max(
            # [N,2] -> [N,1,2] -> [N,M,2]
            boxes1[:, :2].unsqueeze(1).expand(N, M, 2),
            # [M,2] -> [1,M,2] -> [N,M,2]
            boxes2[:, :2].unsqueeze(0).expand(N, M, 2),
        )
        
        rb = torch.min(
            # [N,2] -> [N,1,2] -> [N,M,2]
            boxes1[:, 2:].unsqueeze(1).expand(N, M, 2),
            # [M,2] -> [1,M,2] -> [N,M,2]
            boxes2[:, 2:].unsqueeze(0).expand(N, M, 2),
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N,]
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred, target):
        '''
        shape: (N,S,S,Bx5+20=30); content of B is [x,y,w,h,c]
        '''
        coo_mask = target[:,:,:,4] > 0
        noo_mask = target[:,:,:,4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

if __name__ == '__main__':
    yloss = YoloLoss()

    box1 = np.array([10, 10, 20, 20], dtype=np.float32)
    box2 = np.array([15, 15, 25, 25], dtype=np.float32)
    box3 = np.array([10, 15, 20, 25], dtype=np.float32)
    box4 = np.array([5, 10, 20, 20], dtype=np.float32)
    box5 = np.array([10, 5, 20, 25], dtype=np.float32)

    boxes1 = np.vstack((box1, box2, box3))
    boxes2 = np.vstack((box4, box5))
    iou = yloss.compute_iou(torch.from_numpy(boxes1), torch.from_numpy(boxes2))
    print(iou.shape)
    print(iou)