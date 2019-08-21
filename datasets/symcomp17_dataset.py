import trimesh
import os

import numpy as np
import tensorflow as tf
import np_util

from collections import defaultdict
from dataset import Dataset
from glob import glob


class Symcomp17Dataset(Dataset):
    def __init__(self, basedir, num_samples):
        '''
        Input:
        basedir - train/test directories are assumed to be under this directory

        Output:
        label_map - dictionary that maps class names (annotations) to integer values
        dataset - dictionary with train/test data (keys are 'X_train, y_train, X_test, y_test'
        train_size - size of the training set
        test_size - size of the test set
        '''
        self.basedir = os.path.join(basedir, '')
        self.label_map = {}
        self.label_count = defaultdict(int)
        self.num_samples = num_samples

    def get_trainfiles(self, ext='ply'):
        self.trainfiles = glob(os.path.join(self.basedir, '3D-globalSym-synth-training', 'axis-aligned-models/*.' + ext), recursive=True)
        num_objects = len(self.trainfiles)
        self.trainfiles = self.trainfiles[:int(0.85 * num_objects)]

    def get_testfiles(self, ext='ply'):
        self.testfiles = glob(os.path.join(self.basedir, '3D-globalSym-synth-training', 'axis-aligned-models/*.' + ext), recursive=True)
        num_objects = len(self.testfiles)
        self.testfiles = self.testfiles[int(0.85 * num_objects):]

    @staticmethod
    def load_clean_mesh(filename, ext='ply', return_normal_pars=False):
        mesh = trimesh.load(filename, ext)

        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()

        r = np.linalg.norm(mesh.vertices, axis=1).max()
        centroid = mesh.centroid

        mesh.vertices -= centroid
        mesh.vertices /= r

        if return_normal_pars:
            return mesh, (centroid, r)
        else:
            return mesh

    @staticmethod
    def rotate_mesh_so3(mesh, return_rot_mat=True):
        matrix = np_util.get_rotation(2 * np.pi * np.random.rand(3))
        matrix = np_util.rot_to_trans(matrix)

        if return_rot_mat:
            return mesh.apply_transform(matrix), matrix
        else:
            return mesh.apply_transform(matrix)

    @staticmethod
    def rotate_mesh_azimuthal(mesh, return_rot_mat=True):
        matrix = np_util.get_rotation([2 * np.pi * np.random.rand(), 0, 0])
        matrix = np_util.rot_to_trans(matrix)

        if return_rot_mat:
            return mesh.apply_transform(matrix), matrix
        else:
            return mesh.apply_transform(matrix)


    def _sample_faces(self, mesh):

        p = mesh.area_faces / mesh.area
        face_idx = np.random.choice(mesh.faces.shape[0], size=self.num_samples, p=p)

        weights = np.random.random(size=(self.num_samples,3))
        sum = np.sum(weights, axis=1)[:, None]
        
        weights /= sum
        weights = weights[..., None]

        point_in_face = np.sum(mesh.triangles[face_idx,...] * weights, axis=1)

        return point_in_face


    def generate_tfRecord_train(self, tfrecord_path=None, shuffle_data=True, ext='off', rotate=False, offset=0, file_counter=0, rot_type='z'):
        self.trainfiles = self.shuffle() if shuffle_data else self.trainfiles
            
        if tfrecord_path is None:
            tfrecord_filename = self.basedir + '/train' + str(self.num_samples) + '_{}.tfrecord'.format(file_counter)
        else:
            tfrecord_filename = tfrecord_path + '/train' + str(self.num_samples) + '_{}.tfrecord'.format(file_counter)

        return self.build_tfrecord(tfrecord_filename, self.trainfiles[offset:], ext, rotate, rot_type=rot_type)


    def generate_tfRecord_test(self, tfrecord_path=None, ext='off', rotate=False, offset=0, file_counter=0, rot_type='z'):
        if tfrecord_path is None:
            tfrecord_filename = self.basedir + '/test' + str(self.num_samples) + '_{}.tfrecord'.format(file_counter)
        else:
            tfrecord_filename = tfrecord_path + '/test' + str(self.num_samples) + '_{}.tfrecord'.format(file_counter)

        return self.build_tfrecord(tfrecord_filename, self.testfiles[offset:], ext, rotate, rot_type=rot_type)


    def build_tfrecord(self, tfrecord_filename, files, ext='off', rand_rot=False, rot_type='z'):
        with tf.python_io.TFRecordWriter(tfrecord_filename) as writer:
        # for idx,f in enumerate(self.trainfiles):
            num_valid_meshes = 0
            for k, f in enumerate(files):
                print(k, f)
                # Open the txt ground truth file to read check the symmetry planes

                connector = '' if os.path.splitext(f)[0][-1] == '-' else '-'
                gt_filename = os.path.splitext(f)[0] + connector + 'plane.txt'

                symmetry_planes = np.zeros((3, 3, 3))
                w = np.zeros(3)

                with open(gt_filename, 'r') as gt:
                    num_planes = int(gt.readline())

                    for i in range(num_planes):
                        plane_coords = []
                        for _ in range(3):
                            plane_coords.append([float(v) for v in gt.readline().split(' ')])
                        symmetry_planes[i, ...] = np.array(plane_coords)
                        w[i] = float(gt.readline())

                try:
                    mesh, (mean, radius) = Symcomp17Dataset.load_clean_mesh(f, ext, return_normal_pars=True)
                    num_valid_meshes += 1
                except:
                    print(k, '(Error reading the mesh)', f)
                    continue

                # Normaliza the ground truth planes so everything is inside a unit sphere
                symmetry_planes -= mean[None, None, ...]
                symmetry_planes /= radius
                w /= radius

                if rand_rot:
                    mesh = Symcomp17Dataset.rotate_mesh_so3(mesh)

                face_normals = mesh.face_normals
                triangles = mesh.triangles

                points = self._sample_faces(mesh)

                filename, _ = os.path.splitext(os.path.basename(f))
                example = self.to_tfrecord(face_normals, triangles, points, symmetry_planes, w, num_planes, filename)
                writer.write(example.SerializeToString())
        print('Num. processed meshes: {}'.format(num_valid_meshes))

    def get_rot_matrix(self):
        pass

    @staticmethod
    def to_tfrecord(face_normals, triangles, points, symmetry_planes, w, num_planes, filename):
        '''
        Inputs:
        face_normals:    trimesh.cached.TrackedArray (p,3)
        triangles: trimesh.cached.TrackedArray (q,3,3)
        points:   ndarray (n, 3)
        label:    int (1,)
        '''
        face_normals_raw = np.float32(face_normals).tobytes()
        triangles_raw = np.float32(triangles).tobytes()
        points_raw = np.float32(points).tobytes()
        symmetry_planes_raw = np.float32(symmetry_planes).tobytes()
        w_raw = np.float32(w).tobytes()

        face_normals_shape = np.int32(face_normals.shape)
        triangles_shape = np.int32(triangles.shape)
        points_shape = np.int32(points.shape)
        symmetry_planes_shape = np.int32(symmetry_planes.shape)
        w_shape = np.int32(w.shape)

        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    'num_planes': Dataset._int64_feature(num_planes),

                    'face_normals_raw': Dataset._bytes_feature(face_normals_raw),
                    'triangles_raw': Dataset._bytes_feature(triangles_raw),
                    'points_raw': Dataset._bytes_feature(points_raw),
                    'symmetry_planes_raw': Dataset._bytes_feature(symmetry_planes_raw),
                    'w_raw': Dataset._bytes_feature(w_raw),

                    'face_normals_shape_raw': Dataset._bytes_feature(face_normals_shape.tobytes()),
                    'triangles_shape_raw': Dataset._bytes_feature(triangles_shape.tobytes()),
                    'points_shape_raw': Dataset._bytes_feature(points_shape.tobytes()),
                    'symmetry_planes_shape_raw': Dataset._bytes_feature(symmetry_planes_shape.tobytes()),
                    'w_shape_raw': Dataset._bytes_feature(w_shape.tobytes()),

                    'filename': Dataset._bytes_feature(filename.encode()),
                }))

    @staticmethod
    def from_tfrecord(serialized_example):
        '''
        Parse TFExample records
        '''
        example_fmt={
            'num_planes': tf.FixedLenFeature([], tf.int64),

            'face_normals_raw': tf.FixedLenFeature([], tf.string),
            'triangles_raw': tf.FixedLenFeature([], tf.string),
            'points_raw': tf.FixedLenFeature([], tf.string),
            'symmetry_planes_raw': tf.FixedLenFeature([], tf.string),
            'w_raw': tf.FixedLenFeature([], tf.string),

            'face_normals_shape_raw': tf.FixedLenFeature([], tf.string),
            'triangles_shape_raw': tf.FixedLenFeature([], tf.string),
            'points_shape_raw': tf.FixedLenFeature([], tf.string),
            'symmetry_planes_shape_raw': tf.FixedLenFeature([], tf.string),
            'w_shape_raw': tf.FixedLenFeature([], tf.string),

            'filename': tf.FixedLenFeature([], tf.string),
        }
        
        example_decode = tf.parse_single_example(serialized_example, example_fmt)

        num_planes = example_decode['num_planes']
        filename = example_decode['filename']

        face_normals_string = example_decode['face_normals_raw']
        triangles_string = example_decode['triangles_raw']
        points_string = example_decode['points_raw']
        symmetry_planes_string = example_decode['symmetry_planes_raw']
        w_string = example_decode['w_raw']

        face_normals_shape_string = example_decode['face_normals_shape_raw']
        triangles_shape_string = example_decode['triangles_shape_raw']
        points_shape_string = example_decode['points_shape_raw']
        symmetry_planes_shape_string = example_decode['symmetry_planes_shape_raw']
        w_shape_string = example_decode['w_shape_raw']
        
        face_normals = tf.decode_raw(face_normals_string, tf.float32)
        triangles = tf.decode_raw(triangles_string, tf.float32)
        points = tf.decode_raw(points_string, tf.float32)
        symmetry_planes = tf.decode_raw(symmetry_planes_string, tf.float32)
        w = tf.decode_raw(w_string, tf.float32)
        
        face_normals_shape = tf.decode_raw(face_normals_shape_string, tf.int32)
        triangles_shape = tf.decode_raw(triangles_shape_string, tf.int32)
        points_shape = tf.decode_raw(points_shape_string, tf.int32)
        symmetry_planes_shape = tf.decode_raw(symmetry_planes_shape_string, tf.int32)
        w_shape = tf.decode_raw(w_shape_string, tf.int32)

        face_normals = tf.reshape(face_normals, face_normals_shape)
        triangles = tf.reshape(triangles, triangles_shape)
        points = tf.reshape(points, points_shape)
        symmetry_planes = tf.reshape(symmetry_planes, symmetry_planes_shape)
        w = tf.reshape(w, w_shape)
        
        return face_normals, triangles, points, symmetry_planes, w, num_planes, filename


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Construct dataset')
    parser.add_argument('--datadir', type=str, help='dataset directory', default='/NAS/data/carlos/ModelNet40/')
    parser.add_argument('--save_path', type=str, help='Where to write tfrecord', default='/NAS/data/christine/tmp/')
    parser.add_argument('--num_samples', type=int, help='point cloud size', default=1024)
    parser.add_argument('--ext', type=str, help='Files extension', default='ply')
    parser.add_argument('--rotate', type=str, help='Randomly rotate the mesh.', default='True')

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    dataset = Symcomp17Dataset(args.datadir, args.num_samples)
    dataset.get_trainfiles(ext=args.ext)
    dataset.get_testfiles(ext=args.ext)

    dataset.generate_tfRecord_train(args.save_path, ext=args.ext, rotate=str2bool(args.rotate))
    dataset.generate_tfRecord_test(args.save_path, ext=args.ext, rotate=str2bool(args.rotate))

    # demonstrates use of the file
    tfdataset = tf.data.TFRecordDataset(args.save_path + '/train1024.tfrecord')
    tfdataset = tfdataset.map(map_func=Symcomp17Dataset.from_tfrecord)

    iterator = tfdataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        for i in range(5):
            value = sess.run(next_element)
            print(value[0].shape, value[1].shape, value[2].shape, value[3], value[4])
            print(type(value[0]), type(value[1]), type(value[2]), type(value[3]))
            print(type(value[0][0]), type(value[1][0]), type(value[2][0]), type(value[3]), value[3])

        


