import os
import lmdb
import cv2
from glob import glob
import numpy as np
from tqdm import tqdm


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtPath, outputFile, image_ext=".png", label_prefix="gt_", label_ext=".txt", checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputFile : LMDB output path
        gtPath     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputFile, exist_ok=True)
    env = lmdb.open(outputFile, map_size=1099511627776)
    cache = {}
    cnt = 0

    gtList = glob(f"{gtPath}/*.txt")
    nSamples = len(gtList)
    for i in tqdm(range(nSamples)):

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        labelPath = gtList[i]
        imagePath = labelPath.replace(gtPath, inputPath).replace(label_ext, image_ext).replace(label_prefix, "")
        #bn = label_prefix + os.path.basename(labelPath)
        #labelPath = labelPath.replace(os.path.basename(labelPath), bn)
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        elif not os.path.exists(labelPath):
            print('%s does not exist' % labelPath)
            continue

        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        with open(labelPath, 'rb') as f:
            labelBin = f.read()

        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                print('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        nameKey = 'name-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = labelBin
        cache[nameKey] = os.path.basename(imagePath).encode("utf-8")

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__=="__main__":
    createDataset("ch4_training_images", "ch4_training_localization_transcription_gt", "train_ma", image_ext=".png", label_prefix="gt_", label_ext=".txt", checkValid=True)
    createDataset("ch4_test_images", "ch4_test_localization_transcription_gt", "test_ma", image_ext=".png", label_prefix="gt_", label_ext=".txt", checkValid=True)
