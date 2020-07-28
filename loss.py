import torch
import numpy as np

class YoloLoss(torch.nn.Module):
    def __init__(self, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj

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
        coo_mask = target[:, :, :, 4] > 0
        noo_mask = target[:, :, :, 4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target)

        #计算target中不含目标的位置对应的两个bbox的C和pred中对应的C的MSE
        noo_pred = pred[noo_mask].view(-1, 30) #noo_pred对应的位置是不该有target bbox的即对应物体类别的，这里只会用到bbox的概率值
        noo_target = target[noo_mask].view(-1, 30)
        noo_pred_mask = torch.BoolTensor(noo_pred.size())
        noo_pred_mask.zero_()
        noo_pred_mask[:, 4] = 1
        noo_pred_mask[:, 9] = 1
        noo_pred_c = noo_pred[noo_pred_mask] 
        noo_target_c = noo_target[noo_pred_mask] #如果是hard label，这里是一个全零的向量
        nooobj_loss = torch.nn.MSELoss(reduction='sum')(noo_pred_c, noo_target_c)
        # print(nooobj_loss)


        coo_pred = pred[coo_mask].view(-1, 30)
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5) #box[x1,y1,w1,h1,c1]
        class_pred = coo_pred[:, 10:]

        coo_target = target[coo_mask].view(-1, 30)
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        class_target = coo_target[:, 10:]

        coo_response_mask = torch.BoolTensor(box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = torch.BoolTensor(box_target.size())
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size())
        for i in range(0, box_target.size()[0], 2): #如果target有3个bbox，则预测出了6个pred boox，循环3次，每次循环从两个bbox中找出最大的
            box1 = box_pred[i: i+2] #每个格子预测了两个bbox，两个两个取出来，这里box1的shape是[2, 5]
            box1_xyxy = torch.FloatTensor(box1.size())
            box1_xyxy[:, :2] = box1[:, :2] / 14. - 0.5 * box1[:, 2:4] #前两个数字是中心坐标，后两个是宽高，这样算得到了左上角坐标，用于计算iou
            box1_xyxy[:, 2:4] = box1[:, :2] / 14. + 0.5 * box1[:, 2:4] #前两个数字是中心坐标，后两个是宽高，这样算得到了右下角坐标，用于计算iou
            box2 = box_target[i].view(-1, 5) #target bbox，两个是一样的，取出一个
            box2_xyxy = torch.FloatTensor(box2.size())
            box2_xyxy[:, :2] = box2[:, :2] / 14. - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / 14. + 0.5 * box2[:, 2:4]
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4]) # iou的shape是[2,1]，表示两个预测bbox和target bbox的iou
            max_iou, max_index = iou.max(0)
            
            coo_response_mask[i+max_index, :] = 1 #如果target有3个bbox，则预测出了6个pred boox，循环3次，本次循环从两个bbox中找出最大的
            coo_not_response_mask[i+1-max_index, :] = 1

            box_target_iou[i+max_index, 4] = max_iou.data

        box_pred_response = box_pred[coo_response_mask].view(-1, 5) #如果target是3个bbox，则这里是box_pred_response的shape是[3, 5]
        box_target_response = box_target[coo_response_mask].view(-1, 5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        contain_loss = torch.nn.MSELoss(reduction='sum')(box_pred_response[:, 4], box_target_response_iou[:, 4]) #预测框的C和IOU(预测框，目标框)做MSE，而不是与1做MSE，为啥呢
        loc_loss = torch.nn.MSELoss(reduction='sum')(box_pred_response[:, :2], box_target_response[:, :2]) + torch.nn.MSELoss(reduction='sum')(torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]))

        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0
        not_contain_loss = torch.nn.MSELoss(reduction='sum')(box_pred_not_response[:, 4], box_target_not_response[:, 4]) #非预测框的C和0做MSE

        class_loss = torch.nn.MSELoss(reduction='sum')(class_pred, class_target)

        N = pred.size()[0]
        return (self.l_coord * loc_loss + 2 * contain_loss + not_contain_loss + self.l_noobj * nooobj_loss + class_loss) / N

if __name__ == '__main__':
    yloss = YoloLoss(5, 0.5)

    # box1 = np.array([10, 10, 20, 20], dtype=np.float32)
    # box2 = np.array([15, 15, 25, 25], dtype=np.float32)
    # box3 = np.array([10, 15, 20, 25], dtype=np.float32)
    # box4 = np.array([5, 10, 20, 20], dtype=np.float32)
    # box5 = np.array([10, 5, 20, 25], dtype=np.float32)

    # boxes1 = np.vstack((box1, box2, box3))
    # boxes2 = np.vstack((box4, box5))
    # iou = yloss.compute_iou(torch.from_numpy(boxes1), torch.from_numpy(boxes2))
    # print(iou.shape)
    # print(iou)

    pred = torch.ones((1, 14, 14, 30), dtype=torch.float32)
    # print(pred.shape)
    data = np.load('./img_target.npz')
    target = torch.from_numpy(data['target'])
    # print(target.shape)
    # print(target.sum())
    loss = yloss(pred, target)
    print(loss)