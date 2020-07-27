import os
import numpy as np

def peek_label():
    list_file = os.path.join(os.path.abspath(''), 'dataset/voc2012.txt')
    with open(list_file) as fp:
        lines = fp.read().splitlines()

    labels = []
    for line in lines:
        splited = line.split(' ')
        # self.images.append(splited[0])
        assert (len(splited) - 1) % 5 == 0
        num_boxes = (len(splited) - 1) // 5
        label = []
        for i in range(num_boxes):
            c = splited[5*i+5]
            labels.append(int(c)+1)
        # labels.append(torch.LongTensor(label))
    print('totally %d labels' % len(labels))

    labels = np.array(labels)
    for i in range(22):
        cnt = (labels == i).sum()
        print(i, cnt)

if __name__ == '__main__':
    peek_label()