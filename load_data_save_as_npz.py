# -*- coding: utf-8 -*-
# load the data and save as npz format.
import struct
import gzip
import numpy as np
import pandas as pd


def loadData():
    f = gzip.open("train-images-idx3-ubyte.gz", 'rb')
    a = struct.unpack(">iiii", f.read(16))
    print(a[0]) # 2051
    num_images, image_sizer, image_sizec = a[1:]
    buf = f.read(num_images * image_sizer * image_sizec)
    f.close()
    train_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    train_data = train_data.reshape(num_images, image_sizer, image_sizec)
    f = gzip.open("train-labels-idx1-ubyte.gz", 'rb')
    a = struct.unpack(">ii", f.read(8))
    print(a[0]) # 2049
    num_labels = a[1]
    buf = f.read(num_labels)
    f.close()
    train_lables = np.frombuffer(buf, dtype=np.uint8)
    train_lables = pd.get_dummies(train_lables).to_numpy()
    np.savez("train_data.npz", train_images=train_data, train_lables=train_lables)

    f = gzip.open("t10k-images-idx3-ubyte.gz", 'rb')
    a = struct.unpack(">iiii", f.read(16))
    print(a[0]) # 2051
    num_images, image_sizer, image_sizec = a[1:]
    buf = f.read(num_images * image_sizer * image_sizec)
    f.close()
    test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    test_data = test_data.reshape(num_images, image_sizer, image_sizec)
    f = gzip.open("t10k-labels-idx1-ubyte.gz", 'rb')
    a = struct.unpack(">ii", f.read(8))
    print(a[0]) # 2049
    num_labels = a[1]
    buf = f.read(num_labels)
    f.close()
    test_lables = np.frombuffer(buf, dtype=np.uint8)
    test_lables = pd.get_dummies(test_lables).to_numpy()
    np.savez("test_data.npz", test_images=test_data, test_lables=test_lables)
