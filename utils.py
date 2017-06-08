import numpy as np
import os
import struct
from array import array as pyarray
import pickle
import tarfile

def load_mnist(dataset="training", digits=np.arange(10), path=".", size = 60000):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = size #int(len(ind) * size/100.)
    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(N): #int(len(ind) * size/100.)):
        images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])\
            .reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    labels = [label[0] for label in labels]
    return images, labels

def convert_image_from_pack(raw_data):
    images = []
    labels = []
    
    def get_image_format(data):
        return np.array([[data[i], data[i + 1024], data[i + 2048]] for i in range(1024)])

    for i in range(len(raw_data[b'data'])):
        image_format = get_image_format(raw_data[b'data'][i])
        label = raw_data[b'labels'][i]
        #filepath = path + "/" + str(label) + "/" + raw_data['filenames'][i]
        #saveImage(image_format, "RGB", (32, 32), filepath)
        #logfile.write(filepath + "\t" + str(label) + "\n")
        images += [image_format]
        labels += [label]
        
    return np.array(images), labels

def load_cifar10(filename):
    batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
    
    tar = tarfile.open(filename, "r:gz")
    
    tar_contents = [(file.name.split('/')[-1], file) for file in tar.getmembers()]
    tar_contents = sorted(filter(lambda x: x[0] in batches, tar_contents), key=lambda x: x[0])
    all_images = {}
    all_labels = {}

    for member in tar_contents:
        f = tar.extractfile(member[1])
        if f is not None:
            content = pickle.load(f, encoding='bytes')
            images, labels = convert_image_from_pack(content)
            all_images[member[0]] = images
            all_labels[member[0]] = labels
    tar.close()
    return all_images, all_labels