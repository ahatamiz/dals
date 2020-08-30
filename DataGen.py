## Deep Active Lesion Segmention (DALS), Code by Ali Hatamizadeh ( http://web.cs.ucla.edu/~ahatamiz/ )

from __future__ import print_function, division, absolute_import, unicode_literals
import os
import numpy as np

class BaseDataProvider(object):

    channels = 1
    n_class = 2
    def _load_data_and_label(self):
        train_data, labels,shape = self._next_data()
        nx = train_data.shape[1]
        ny = train_data.shape[0]
        train_data = train_data.reshape(ny, nx, self.channels)
        labels = labels.reshape(ny, nx, self.n_class)

        return train_data, labels,shape
    def __call__(self, n):
        train_data, labels,shape = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[0]
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels,shape  = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels

        return X, Y,shape

class ImageGen(BaseDataProvider):

    def __init__(self, search_path,data_suffix, mask_suffix,
                 shuffle_data, n_class):
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class
        self.data_files = self._find_data_files(search_path)
        if self.shuffle_data:
            np.random.shuffle(self.data_files)
        assert len(self.data_files) > 0
        img = self._load_file(self.data_files[0])
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]

    def _find_data_files(self, search_path):
        all_files= [os.path.join(path, file) for (path, dirs, files) in os.walk(search_path)for file in files]

        return [name for name in all_files if self.data_suffix in name and not self.mask_suffix in name]

    def _load_file(self, path):
        image = np.load(path)
        image = image.astype('float32')
        image = (image - np.min(image)) * (255.0 / (np.max(image) - np.min(image)))

        return image

    def _load_label(self, path):
        label = np.load(path)
        label = label.astype('float32')
        label *= 1.0 / label.max()

        return label,label.shape

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0
            if self.shuffle_data:
                np.random.shuffle(self.data_files)
    def _next_data(self):

        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        label_name = image_name.replace(self.data_suffix, self.mask_suffix)
        img = self._load_file(image_name)
        label,shape = self._load_label(label_name)

        return img, label,shape
