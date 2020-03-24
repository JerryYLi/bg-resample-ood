'''
Dataloader for LMDB files
Code adapted from https://github.com/Fangyh09/Image2LMDB
'''

import os, sys
import os.path as osp
from PIL import Image
import six
import string

import lmdb
import pickle
import umsgpack
from tqdm import tqdm
import pyarrow as pa
from os.path import basename

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


def read_txt(fname):
    map = {}
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    for line in content:
        # img, idx = line.split(" ")
        # map[img] = idx
        img, idx, label = line.split(" ")
        map[idx] = label
    return map


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None, ret_im=True):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = loads_pyarrow(txn.get(b'__len__'))
            # self.keys = umsgpack.unpackb(txn.get(b'__keys__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform
        
        map_path = db_path[:-5] + "_images_idx_label.txt"
        # self.img2idx = read_txt(map_path)
        self.labels = read_txt(map_path)

        self.ret_im = ret_im

    def __getitem__(self, index):
        img, target = None, None

        if self.ret_im:
            env = self.env
            with env.begin(write=False) as txn:
                # print("key", self.keys[index].decode("ascii"))
                byteflow = txn.get(self.keys[index])
            unpacked = loads_pyarrow(byteflow)

            imgbuf = unpacked
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
        
            if self.transform is not None:
                img = self.transform(img)
        
        target = self.labels[self.keys[index].decode('utf-8')]
        target = int(target)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def folder2lmdb(dpath, outpath, name="train", write_frequency=5000):
    all_imgpath = []
    all_idxs = []
    all_labels = []
    directory = osp.expanduser(osp.join(dpath, name))
    print("Loading dataset from %s" % directory)
    dataset = ImageFolderWithPaths(directory, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)

    lmdb_path = osp.join(outpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    # print("Generate LMDB to %s" % lmdb_path)
    # db = lmdb.open(lmdb_path, subdir=isdir,
    #                map_size=1099511627776 * 2, readonly=False,
    #                meminit=False, map_async=True)

    # txn = db.begin(write=True)
    for idx, data in enumerate(tqdm(data_loader)):
        image, label, imgpath = data[0]
        imgpath = basename(imgpath)
        all_imgpath.append(imgpath)
        all_idxs.append(idx)
        all_labels.append(label)
        # txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow(image))
        # if idx % write_frequency == 0:
        #     print("[%d/%d]" % (idx, len(data_loader)))
        #     txn.commit()
        #     txn = db.begin(write=True)

    # finish iterating through dataset
    # txn.commit()
    # keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    # with db.begin(write=True) as txn:
    #     txn.put(b'__keys__', dumps_pyarrow(keys))
    #     txn.put(b'__len__', dumps_pyarrow(len(keys)))

    # print("Flushing database ...")
    # db.sync()
    # db.close()

    # txt_path = osp.join(outpath, "%s_images_idx.txt" % name)
    txt_path = osp.join(outpath, "%s_images_idx_label.txt" % name)
    print("Saving image paths to %s" % txt_path)
    fout = open(txt_path, "w")
    # for img, idx in zip(all_imgpath, all_idxs):
        # fout.write("{} {}\n".format(img, idx))
    for img, idx, label in zip(all_imgpath, all_idxs, all_labels):
        fout.write("{} {} {}\n".format(img, idx, label))
    fout.close()


import fire

if __name__ == '__main__':
    fire.Fire(folder2lmdb)
