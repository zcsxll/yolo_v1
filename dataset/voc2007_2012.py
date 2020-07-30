import os
import torch
import cv2
import random
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, train=True):
        # random.seed(1)
        self.transform = transform
        self.train = train
        self.mean = (123, 117, 104) #RGB
        self.image_size = 448

        self.root_2007 = '/home/zhaochengshuai/dataset/image/VOC2007/JPEGImages/'
        self.root_2012 = '/home/zhaochengshuai/dataset/image/VOC2012/JPEGImages/'
        list_file_2007 = os.path.join(os.path.abspath(''), 'dataset/voc2007.txt')
        list_file_2012 = os.path.join(os.path.abspath(''), 'dataset/voc2012.txt')
        with open(list_file_2007) as fp:
            lines_2007 = fp.read().splitlines()
        with open(list_file_2012) as fp:
            lines_2012 = fp.read().splitlines()

        self.images = []
        self.boxes = []
        self.labels = []
        for line in lines_2007:
            splited = line.split(' ')
            self.images.append(os.path.join(self.root_2007, splited[0]))
            assert (len(splited) - 1) % 5 == 0
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                x1 = float(splited[5*i+1])
                y1 = float(splited[5*i+2])
                x2 = float(splited[5*i+3])
                y2 = float(splited[5*i+4])
                c = splited[5*i+5]
                box.append([x1, y1, x2, y2])
                label.append(int(c)+1)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

        for line in lines_2012:
            splited = line.split(' ')
            self.images.append(os.path.join(self.root_2012, splited[0]))
            assert (len(splited) - 1) % 5 == 0
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                x1 = float(splited[5*i+1])
                y1 = float(splited[5*i+2])
                x2 = float(splited[5*i+3])
                y2 = float(splited[5*i+4])
                c = splited[5*i+5]
                box.append([x1, y1, x2, y2])
                label.append(int(c)+1)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

        self.num_samples = len(self.boxes)
        print('totally %d images' % self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # print(os.path.join(self.root, self.images[idx]))
        img = cv2.imread(self.images[idx])
        boxes = self.boxes[idx]
        labels = self.labels[idx]

        if self.train:
            #img = self.random_bright(img)
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.randomScale(img,boxes)
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)
            img, boxes, labels = self.randomShift(img, boxes, labels)
            img, boxes, labels = self.randomCrop(img, boxes, labels)

        h, w, _ = img.shape
        # print(boxes)
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        img = self.BGR2RGB(img) #because pytorch pretrained model use RGB
        img = self.subMean(img, self.mean) #减去均值
        img = cv2.resize(img, (self.image_size, self.image_size))
        target = self.encoder(boxes, labels)
        for t in self.transform:
            img = t(img)

        return img, target

    def encoder(self, boxes, labels):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]], 都是0到1范围
        labels (tensor) [...]
        return 14x14x30
        '''
        grid_num = 14
        target = torch.zeros((grid_num, grid_num, 30))
        cell_size = 1. / grid_num
        wh = boxes[:, 2:] - boxes[:, :2]
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
        # print(wh, '====', cxcy)

        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            # print(cxcy_sample)
            ij = (cxcy_sample / cell_size).ceil() - 1 #0到1范围的中心xy除以格子的尺寸，得到格子编号
            # print(ij)
            # print(cell_size)
            target[int(ij[1]), int(ij[0]), 4] = 1 #两个bbox的c都是1
            target[int(ij[1]), int(ij[0]), 9] = 1 #两个bbox的c都是1
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1 #是+9，而不是+10呢，因为分类是1到20，不是0到19，读取的时候加了1，这里相当于又减去1
            xy = ij * cell_size #格子左上角相对于14*14网格左上角的相对坐标
            # print(xy)
            '''
            cxcy_sample是目标中心相对于14*14格子左上角的坐标
            这个坐标位于第ij个格子内，
            xy是第ij个格子的左上角相对于14*14格子左上角的坐标
            cxcy_sample-xy就是目标中心相对于第ij个格子左上角的坐标，这个值的范围是0到格子的尺寸，即[0, cell_size]
            最后除以cell_size归一化到[0,1]，表示目标中心在第ij个格子内部，相对于第ij个格子的左上角偏差的百分比
            '''
            delta_xy = (cxcy_sample - xy) / cell_size
            # print(cxcy_sample - xy)
            # print(delta_xy)
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return target

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes

    def randomScale(self,bgr,boxes):
        #固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8,1.2)
            height,width,c = bgr.shape
            bgr = cv2.resize(bgr,(int(width*scale),height))
            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr,boxes
        return bgr, boxes

    def randomBlur(self, bgr):
        if random.random()<0.5:
            # print(bgr.shape)
            bgr = cv2.blur(bgr,(5,5))
        return bgr

    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomShift(self, bgr, boxes, labels):
        #平移变换
        center = (boxes[:,2:]+boxes[:,:2])/2
        if random.random() <0.5:
            height,width,c = bgr.shape
            after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)
            after_shfit_image[:,:,:] = (104,117,123) #bgr
            shift_x = random.uniform(-width*0.2,width*0.2)
            shift_y = random.uniform(-height*0.2,height*0.2)
            #print(bgr.shape,shift_x,shift_y)
            #原图像的平移
            if shift_x>=0 and shift_y>=0:
                after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]
            elif shift_x>=0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x <0 and shift_y >=0:
                after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
            elif shift_x<0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]

            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] >0) & (center[:,0] < width)
            mask2 = (center[:,1] >0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if len(boxes_in) == 0:
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image,boxes_in,labels_in
        return bgr, boxes, labels

    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:,2:]+boxes[:,:2])/2
            height,width,c = bgr.shape
            h = random.uniform(0.6*height,height)
            w = random.uniform(0.6*width,width)
            x = random.uniform(0,width-w)
            y = random.uniform(0,height-h)
            x,y,h,w = int(x),int(y),int(h),int(w)

            center = center - torch.FloatTensor([[x,y]]).expand_as(center)
            mask1 = (center[:,0]>0) & (center[:,0]<w)
            mask2 = (center[:,1]>0) & (center[:,1]<h)
            mask = (mask1 & mask2).view(-1,1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if(len(boxes_in)==0):
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:,0]=boxes_in[:,0].clamp_(min=0,max=w)
            boxes_in[:,2]=boxes_in[:,2].clamp_(min=0,max=w)
            boxes_in[:,1]=boxes_in[:,1].clamp_(min=0,max=h)
            boxes_in[:,3]=boxes_in[:,3].clamp_(min=0,max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y+h,x:x+w,:]
            return img_croped,boxes_in,labels_in
        return bgr, boxes, labels

    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

    def subMean(self,bgr,mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

if __name__ == '__main__':
    import torchvision.transforms as transforms

    dataset = Dataset(transform=[transforms.ToTensor()])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for idx, (img, target) in enumerate(dataloader):
        print(idx, img.shape, target.shape)
        # print(target[0, 6, 4, :])
        np.savez('img_target.npz', img=img.cpu().detach().numpy(), target=target.cpu().detach().numpy())
        break

    #acbd75e74e6c4b4d65e60b9ec3f3adc6
