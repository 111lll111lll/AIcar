import os
import random

if __name__ == "__main__":
    trainval_percent = 0.1  # 可自行进行调节
    train_percent = 1
    xmlfilepath = 'VOC2007/Annotations/'
    txtsavepath = 'VOC2007/ImageSets/Main/'
    total_xml = os.listdir(xmlfilepath)

    num = len(total_xml)
    print("total num:", num)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    # ftrainval = open('ImageSets/Main/trainval.txt', 'w')
    ftest = open('VOC2007/ImageSets/Main/test.txt', 'w')
    ftrain = open('VOC2007/ImageSets/Main/train.txt', 'w')
    # fval = open('ImageSets/Main/val.txt', 'w')

    for i in list:
        if not total_xml[i].endswith(".xml"):
            continue
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            # ftrainval.write(name)
            if i in train:
                ftest.write(name)
            # else:
            # fval.write(name)
        else:
            ftrain.write(name)

    # ftrainval.close()
    ftrain.close()
    # fval.close()
    ftest.close()
    print("split success!")