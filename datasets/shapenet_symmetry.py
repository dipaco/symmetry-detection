'''
    ModelNet dataset. Support ModelNet40, XYZ channels. Up to 2048 points.
    Faster IO than ModelNetDataset in the first epoch.
'''

import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider


# Download dataset for point cloud classification
DATA_DIR = os.path.join(ROOT_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))


def shuffle_data(data, labels, cluster_labels):
    """ Shuffle data and labels.
        Input:
          data: B, K, N, ... numpy array
          label: B, K, ... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(labels.shape[0])
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx, ...], cluster_labels[idx, ...], idx

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    point_cloud = f['points'][:]

    point_cloud_10 = f['points_10'][:]
    point_cloud_20 = f['points_20'][:]
    point_cloud_30 = f['points_30'][:]
    point_cloud_50 = f['points_50'][:]

    data = np.concatenate(
        (point_cloud[:, None, ...],
         point_cloud_10[:, None, ...],
         point_cloud_20[:, None, ...],
         point_cloud_30[:, None, ...],
         point_cloud_50[:, None, ...]
         ), axis=1)

    symmetry_plane = f['symmetry_planes'][:]
    symmetry_plane = np.squeeze(symmetry_plane)

    #FIXME: the symmetry plane should have 4 dim as well
    if symmetry_plane.shape[1] < 4:
        symmetry_plane = np.concatenate(
            (
                symmetry_plane,
                np.zeros((symmetry_plane.shape[0], 1))
            ), axis=1
        )

    cut_plane_10 = f['cut_plane_10'][:]
    cut_plane_20 = f['cut_plane_20'][:]
    cut_plane_30 = f['cut_plane_30'][:]
    cut_plane_50 = f['cut_plane_50'][:]

    label = np.concatenate(
        (
            symmetry_plane[:, None, :],
            cut_plane_10[:, None, :],
            cut_plane_20[:, None, :],
            cut_plane_30[:, None, :],
            cut_plane_50[:, None, :]
        ), axis=1)

    cluster_labels_10 = f['cluster_labels_10'][:]
    cluster_labels_20 = f['cluster_labels_20'][:]
    cluster_labels_30 = f['cluster_labels_30'][:]
    cluster_labels_50 = f['cluster_labels_50'][:]

    cluster_labels = np.concatenate(
        (
            symmetry_plane[:, None, ...],
            cluster_labels_10[:, None, ...],
            cluster_labels_20[:, None, ...],
            cluster_labels_30[:, None, ...],
            cluster_labels_50[:, None, ...]
        ), axis=1)

    f.close()
    return (data, label, cluster_labels)

def loadDataFile(filename):
    return load_h5(filename)


class ShapenetSymmetryDataset(object):
    def __init__(self, list_filename, batch_size = 32, npoints = 1024, shuffle=True):
        self.list_filename = list_filename
        self.batch_size = batch_size
        self.npoints = npoints
        self.shuffle = shuffle
        self.h5_files = getDataFiles(self.list_filename)
        self.reset()

    def reset(self):
        ''' reset order of h5 files '''
        self.file_idxs = np.arange(0, len(self.h5_files))
        if self.shuffle: np.random.shuffle(self.file_idxs)
        self.current_data = None
        self.current_label = None
        self.current_file_idx = 0
        self.batch_idx = 0
   
    def _augment_batch_data(self, batch_data, labels):
        rotated_data, rotated_labels = provider.rotate_point_cloud_z(batch_data, labels)
        rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
        jittered_data = provider.random_scale_point_cloud(rotated_data)
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data = jittered_data
        return provider.shuffle_points(rotated_data), rotated_labels


    def _get_data_filename(self):
        return self.h5_files[self.file_idxs[self.current_file_idx]]

    def _load_data_file(self, filename):
        self.current_data, self.current_label, self.cluster_labels = load_h5(filename)
        #self.current_label = np.squeeze(self.current_label)
        self.batch_idx = 0
        if self.shuffle:
            self.current_data, self.current_label, self.cluster_labels, _ = shuffle_data(self.current_data, self.current_label, self.cluster_labels)
    
    def _has_next_batch_in_file(self):
        return self.batch_idx*self.batch_size < self.current_data.shape[0]

    def num_channel(self):
        return 3

    def has_next_batch(self):
        # TODO: add backend thread to load data
        if (self.current_data is None) or (not self._has_next_batch_in_file()):
            if self.current_file_idx >= len(self.h5_files):
                return False
            self._load_data_file(self._get_data_filename())
            self.batch_idx = 0
            self.current_file_idx += 1
        return self._has_next_batch_in_file()

    def next_batch(self, augment=False):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, self.current_data.shape[0])
        bsize = end_idx - start_idx
        data_batch = self.current_data[start_idx:end_idx, :, 0:self.npoints, :].copy()
        label_batch = self.current_label[start_idx:end_idx, ...].copy()
        cluster_labels_batch = self.cluster_labels[start_idx:end_idx, :, 0:self.npoints, :].copy()
        self.batch_idx += 1
        if augment: data_batch, label_batch = self._augment_batch_data(data_batch, label_batch)
        return data_batch, label_batch, cluster_labels_batch

if __name__=='__main__':
    d = ShapenetSymmetryDataset('data/modelnet40_ply_hdf5_2048/train_files.txt')
    print((d.shuffle))
    print((d.has_next_batch()))
    ps_batch, cls_batch = d.next_batch(True)
    print((ps_batch.shape))
    print((cls_batch.shape))
