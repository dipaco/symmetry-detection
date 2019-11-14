#import matplotlib
#matplotlib.use('TkAgg')
import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))

import tqdm
import pickle
import time
import h5py
import numpy as np
import tensorflow as tf
import symcomp17_dataset
import np_util
import vis_util
from glob import glob
from pathlib import Path
from tf_util import tf_get_cyl_rep
from misc import splitall
from shutil import copyfile
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy import cluster


class ShapeNetSymmetryDataset(symcomp17_dataset.Symcomp17Dataset):
    _data_filenames = None
    MAX_RUNNING_TIME = 3600

    def __init__(self, basedir, num_samples, max_num_triangles):
        super(ShapeNetSymmetryDataset, self).__init__(basedir, num_samples)
        self.max_num_triangles = max_num_triangles

    def _get_filenames(self, mode, ext, synset_id, use_all_cateogories=False):
        if self._data_filenames is None:
            if use_all_cateogories:
                self._data_filenames = self.shuffle(glob(os.path.join(self.basedir, '*/*/models/model_normalized.{}'.format(ext)), recursive=True))
            else:
                self._data_filenames = self.shuffle(glob(os.path.join(self.basedir, '{}/*/models/model_normalized.{}'.format(synset_id, ext)), recursive=True))
            return self._data_filenames
        else:
            return self._data_filenames

    def get_trainfiles(self, ext='ply', synset_id='*', use_all_cateogories=False):
        files = self._get_filenames('train', ext, synset_id, use_all_cateogories=use_all_cateogories)
        num_objects = len(files)
        self.trainfiles = files[:int(0.80 * num_objects)]

    def get_testfiles(self, ext='ply', synset_id='*', use_all_cateogories=False):
        files = self._get_filenames('test', ext, synset_id, use_all_cateogories=use_all_cateogories)
        num_objects = len(files)
        self.testfiles = files[int(0.80 * num_objects):]

    def set_trainfiles(self, files):
        self.trainfiles = files

    def set_testfiles(self, files):
        self.testfiles = files

    def build_h5_file(self, h5_filename, files, ext='obj', rotate=False, rot_type='z', objs_per_file=1024, check_symmetry=False):

        init_time = time.time()

        with tf.Session() as sess:

            all_face_normals = []
            all_triangles = []
            all_points = []
            all_cluster_labels = []
            all_inc_cluster_labels = []
            clusters = [2, 3, 4, 8]
            incomplete_data = {
                10: {'points': [], 'plane': [], 'cluster_labels': []},
                20: {'points': [], 'plane': [], 'cluster_labels': []},
                30: {'points': [], 'plane': [], 'cluster_labels': []},
                50: {'points': [], 'plane': [], 'cluster_labels': []}
            }
            all_symmetry_planes = []
            all_filenames = []

            triangle = tf.placeholder(tf.float32, shape=(None, None, 3, 3))
            normals = tf.placeholder(tf.float32, shape=(None, None, 3))

            total_meshes = len(files)

            num_valid_meshes = 0
            pb_files = tqdm.tqdm(files)
            for k, f in enumerate(pb_files):
                pb_files.set_description('Processing {} (valid={}/{})'.format(f, num_valid_meshes, total_meshes))

                # The default symmetry plane is YZ plane
                symmetry_plane = np.array([[1.0, 0.0, 0.0]]).T
                w = np.ones(3)
                num_planes = 1

                #try:
                mesh, (mean, radius) = ShapeNetSymmetryDataset.load_clean_mesh(f, ext, return_normal_pars=True)

                # For visualization purposes we set the z axis to go up
                rot_mat = np.array([
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]
                ])

                mesh.apply_transform(rot_mat)
                #except:
                #print(k, '(Error reading the mesh)', f)
                #continue
                    # mesh.show()

                # Only use meshes with a certain maximum number of triangles
                if mesh.triangles.shape[0] > self.max_num_triangles:
                    continue
                else:

                    # Check if the mesh is symmetric, otherwise assume that the mesh symmetric with respect to plane YZ
                    if check_symmetry:

                        tr1 = np.transpose(mesh.triangles, axes=[0, 2, 1])[None, ...]
                        norm = mesh.face_normals[None, ...]

                        # Creates a cylindrical representation of the objects to check the symmetry
                        r = tf_get_cyl_rep(triangle, normals, (32, 32), init_theta=tf.constant(np.pi / 2.0))

                        res = sess.run(r, feed_dict={triangle: tr1.astype(np.float32), normals: norm})
                        res = res[0, ...]
                        res /= max(np.sqrt((res**2).sum()), 1e-5)
                        flipped_res = np.flip(res, axis=1)

                        corr = np.sum(res*flipped_res)

                        if 1 - corr > 2e-2:
                            continue
                        else:
                            num_valid_meshes += 1
                    else:
                        num_valid_meshes += 1

                # Since in ShapeNet the symmetry plane is always the same plane, we add either azimuthal or random
                # rotations to add more variability to the data.
                if rotate:
                    if rot_type == 'so3':
                        mesh, rot_mat = ShapeNetSymmetryDataset.rotate_mesh_so3(mesh, return_rot_mat=True)
                    elif rot_type == 'z':
                        mesh, rot_mat = ShapeNetSymmetryDataset.rotate_mesh_azimuthal(mesh, return_rot_mat=True)
                    else:
                        raise ValueError('Rotation type {} is not valid.'.format(rot_type))

                    symmetry_plane = rot_mat[:3, :3] @ symmetry_plane

                face_normals = mesh.face_normals
                triangles = mesh.triangles

                points = self._sample_faces(mesh)

                filename, _ = os.path.splitext(os.path.basename(f))

                # Generates cluster labels for every k in clusters
                gamma = 0.1
                point_labels = np.zeros((points.shape[0], len(clusters)), dtype=int)
                for i, k in enumerate(clusters):
                    features = np.concatenate((gamma * points, np.linalg.norm(points, axis=1)[:, None]), axis=1)
                    c, d = scipy.cluster.vq.kmeans(features, k)
                    point_labels[:, i] = np.argmin(scipy.spatial.distance.cdist(points, c[:, 0:3]), axis=1)

                # Remove up to the x percent of points by chopping the point cloud with a random plane
                # the value of 'key' corresponds to the percentage of data we want to remove from the point cloud
                for key in incomplete_data.keys():
                    aux_points, aux_plane, inc_point_labels = self._remove_parts(points, symmetry_plane, point_labels, t_upper=key/100)
                    incomplete_data[key]['points'].append(aux_points)
                    incomplete_data[key]['plane'].append(aux_plane)
                    incomplete_data[key]['cluster_labels'].append(inc_point_labels)

                all_face_normals.append(face_normals)
                all_triangles.append(triangles)
                all_points.append(points)
                all_cluster_labels.append(point_labels)
                all_symmetry_planes.append(symmetry_plane)
                all_filenames.append(filename)

                # copies the mesh file
                if check_symmetry:
                    self.copy_mesh(f, os.path.join(os.path.dirname(h5_filename), 'dataset_symmetric_copy'))

                # exits after one hour of running
                if time.time() - init_time > self.MAX_RUNNING_TIME or num_valid_meshes >= objs_per_file:
                    self._write_h5_data(h5_filename,
                                        points=all_points,
                                        incomplete_point_clouds=incomplete_data,
                                        face_normals=all_face_normals,
                                        triangles=all_triangles,
                                        symmetry_planes=all_symmetry_planes,
                                        filenames=all_filenames,
                                        cluster_labels=all_cluster_labels)
                    return k + 1

        print('Num. processed meshes: {}'.format(num_valid_meshes))
        self._write_h5_data(h5_filename,
                            points=all_points,
                            incomplete_point_clouds=incomplete_data,
                            face_normals=all_face_normals,
                            triangles=all_triangles,
                            symmetry_planes=all_symmetry_planes,
                            filenames=all_filenames)
        return k + 1

    def _write_h5_data(self, h5_filename, points, incomplete_point_clouds, face_normals, triangles, symmetry_planes, filenames, cluster_labels):
        hf = h5py.File(h5_filename, 'w')
        hf.create_dataset('points', data=np.array(points), compression="gzip", compression_opts=9)
        #hf.create_dataset('face_normals', data=face_normals, compression="gzip", compression_opts=9)
        #hf.create_dataset('triangles', data=np.array(triangles, dtype=object), compression="gzip", compression_opts=9)
        hf.create_dataset('symmetry_planes', data=np.array(symmetry_planes), compression="gzip", compression_opts=9)
        hf.create_dataset('cluster_labels', data=np.array(cluster_labels), compression="gzip", compression_opts=9)
        #hf.create_dataset('filenames', data=filenames, compression="gzip", compression_opts=9)

        for key in incomplete_point_clouds.keys():
            hf.create_dataset(f'points_{key}', data=np.array(incomplete_point_clouds[key]['points']), compression="gzip", compression_opts=9)
            hf.create_dataset(f'cut_plane_{key}', data=np.array(incomplete_point_clouds[key]['plane']), compression="gzip", compression_opts=9)
            hf.create_dataset(f'cluster_labels_{key}', data=np.array(incomplete_point_clouds[key]['cluster_labels']), compression="gzip", compression_opts=9)

        hf.close()

    def _remove_parts(self, points, symmetry_plane, point_labels, t_upper=0.2, t_lower=0.1, show=False):

        # randomly generates the normal to the plane
        # the normal will be close to the plane of symmetry differing
        # in their normals by two rotations (azimuthal and elevation)
        # so that their angles of rotation are normally distributed. (az, el ~ N(0, pi/8))
        #v = np.random.uniform(-1.0, 1.0, 3)
        az = np.random.normal(0, np.pi/8)
        R_az = np_util.get_rotation([az, 0.0, 0.0])
        el = np.random.normal(0, np.pi/8)
        R_el = np_util.get_rotation([0.0, el, 0.0])

        v = (R_el @ R_az) @ symmetry_plane

        v /= np.linalg.norm(v)

        m = 50
        t1 = 0.0
        t2 = 1.0
        for i in range(0, m + 1):

            d = i * 1 / m
            a = points @ v - d

            right_side = points[np.where(a > 0)[0], :]

            r = right_side.shape[0] / points.shape[0]
            if r >= t_upper:
                t1 = d

            if r <= t_lower:
                t2 = d
                break

        d = np.random.uniform(t1, t2)
        a = points @ v - d

        right_side = points[np.where(a > 0)[0], :]
        left_side = points[np.where(a <= 0)[0], :]

        r = right_side.shape[0] / points.shape[0]

        if show:

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # for i in range(points.shape[0]):

            colors = plt.cm.get_cmap(lut=np.max(point_labels[:, 0]) + 1)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='.', color=colors.colors[point_labels[:, 0]])

            vis_util._set_unit_limits_in_3d_plot(ax)
            plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            vis_util._add_plane(ax, v[:, 0], color='green', alpha=0.1, show_normal=True, d=d)
            vis_util._add_plane(ax, symmetry_plane[:, 0], color='orange', alpha=0.1, show_normal=True, d=d)
            ax.scatter(left_side[:, 0], left_side[:, 1], left_side[:, 2], marker='.', color='blue')
            ax.scatter(right_side[:, 0], right_side[:, 1], right_side[:, 2], marker='.', color='red')
            vis_util._set_unit_limits_in_3d_plot(ax)
            plt.title(f'missing parts: {r}\nt1:{t1} t2:{t2}')

            plt.show()

        new_points = np.zeros_like(points)
        new_point_labels = np.zeros_like(point_labels)
        new_points[:left_side.shape[0], :] = left_side
        new_point_labels[:left_side.shape[0], :] = point_labels[np.where(a <= 0)[0], :]
        return new_points, np.append(v[:, 0], -d), point_labels


    def build_tfrecord(self, tfrecord_filename, files, ext='obj', rotate=False, rot_type='z'):

        init_time = time.time()

        with tf.python_io.TFRecordWriter(tfrecord_filename) as writer:
            with tf.Session() as sess:

                triangle = tf.placeholder(tf.float32, shape=(None, None, 3, 3))
                normals = tf.placeholder(tf.float32, shape=(None, None, 3))

                total_meshes = len(files)

                num_valid_meshes = 0
                pb_files = tqdm.tqdm(files)
                for k, f in enumerate(pb_files):
                    pb_files.set_description('Processing {} (valid={}/{})'.format(f, num_valid_meshes, total_meshes))

                    symmetry_planes = np.zeros((3, 3, 3))
                    w = np.ones(3)
                    num_planes = 1

                    try:
                        mesh, (mean, radius) = ShapeNetSymmetryDataset.load_clean_mesh(f, ext, return_normal_pars=True)

                        # For visualization purposes we set the z axis to go up
                        rot_mat = np.array([
                            [1, 0, 0, 0],
                            [0, 0, -1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]
                        ])

                        mesh.apply_transform(rot_mat)

                        # Then the default symmetry plane is YZ plane
                        symmetry_planes[0, ...] = np.array([[0, 1, -1], [0, 1, 1], [0, -1, -1]]).astype(float)
                    except:
                        print(k, '(Error reading the mesh)', f)
                        continue
                        # mesh.show()

                    # Only use meshes with a certain maximum number of triangles
                    if mesh.triangles.shape[0] > self.max_num_triangles:
                        continue
                    else:

                        tr1 = np.transpose(mesh.triangles, axes=[0, 2, 1])[None, ...]
                        norm = mesh.face_normals[None, ...]

                        r = tf_get_cyl_rep(triangle, normals, (32, 32), init_theta=tf.constant(np.pi / 2.0))

                        res = sess.run(r, feed_dict={triangle: tr1.astype(np.float32), normals: norm})
                        res = res[0, ...]
                        res /= max(np.sqrt((res**2).sum()), 1e-5)
                        flipped_res = np.flip(res, axis=1)

                        corr = np.sum(res*flipped_res)

                        if 1 - corr > 2e-2:
                            continue
                        else:
                            num_valid_meshes += 1

                    # Since in ShapeNet the symmetry plane is always the same plane, we add either azimuthal or random
                    # rotations to add more variability to the data.
                    if rotate:
                        if rot_type == 'so3':
                            mesh, rot_mat = ShapeNetSymmetryDataset.rotate_mesh_so3(mesh, return_rot_mat=True)
                        elif rot_type == 'z':
                            mesh, rot_mat = ShapeNetSymmetryDataset.rotate_mesh_azimuthal(mesh, return_rot_mat=True)
                        else:
                            raise ValueError('Rotation type {} is not valid.'.format(rot_type))

                        symmetry_planes[0, ...] = (rot_mat[:3, :3] @ symmetry_planes[0, ...].T).T

                    face_normals = mesh.face_normals
                    triangles = mesh.triangles

                    points = self._sample_faces(mesh)

                    filename, _ = os.path.splitext(os.path.basename(f))
                    example = self.to_tfrecord(face_normals, triangles, points, symmetry_planes, w, num_planes, filename)
                    writer.write(example.SerializeToString())

                    # copies the mesh file
                    self.copy_mesh(f, os.path.join(os.path.dirname(tfrecord_filename), 'dataset_symmetric_copy'))

                    # exits after one hour of running
                    if time.time() - init_time > self.MAX_RUNNING_TIME:
                        return k + 1

        print('Num. processed meshes: {}'.format(num_valid_meshes))
        return k + 1

    def copy_mesh(self, f, output_folder):
        els = splitall(f)
        output_folder = os.path.join(output_folder, *els[-4:-1])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        copyfile(f, os.path.join(output_folder, els[-1]))

    def generate_h5_train(self, tfrecord_path=None, shuffle_data=True, ext='off', rotate=False, offset=0,
                                file_counter=0, rot_type='z', objs_per_file=1024, check_symmetry=False):
        self.trainfiles = self.shuffle() if shuffle_data else self.trainfiles

        if tfrecord_path is None:
            h5_filename = self.basedir + '/train' + str(self.num_samples) + '_{}.h5'.format(file_counter)
        else:
            h5_filename = tfrecord_path + '/train' + str(self.num_samples) + '_{}.h5'.format(file_counter)

        return self.build_h5_file(h5_filename, self.trainfiles[offset:], ext, rotate, rot_type=rot_type, objs_per_file=objs_per_file, check_symmetry=check_symmetry)

    def generate_h5_test(self, tfrecord_path=None, ext='off', rotate=False, offset=0, file_counter=0,
                               rot_type='z', objs_per_file=1024, check_symmetry=False):
        if tfrecord_path is None:
            h5_filename = self.basedir + '/test' + str(self.num_samples) + '_{}.h5'.format(file_counter)
        else:
            h5_filename = tfrecord_path + '/test' + str(self.num_samples) + '_{}.h5'.format(file_counter)

        return self.build_h5_file(h5_filename, self.testfiles[offset:], ext, rotate, rot_type=rot_type, objs_per_file=objs_per_file, check_symmetry=check_symmetry)


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
    parser.add_argument('--synset-id', type=str, help='Category id of model inside shapenet', default='*')
    parser.add_argument('--max_num_triangles', type=int, help='Max num of triangles allowed.', default=10000)
    parser.add_argument('--ext', type=str, help='Files extension', default='ply')
    parser.add_argument('--rotate', type=str, help='Randomly rotate the mesh.', default='True')
    parser.add_argument('--rot_type', type=str, help='Rotation type [z, so3].', default='z')
    parser.add_argument('--make-a-copy', type=str, help='Copy the symmetric files.', default='True')
    parser.add_argument('--num_objs_per_file', type=int, help='number of h5 files to generate', default=1024)
    parser.add_argument('--dataset_type', type=str, help='Type of dataset: (tfrecord or h5).', default='h5')
    parser.add_argument('--use_all_categories', action='store_true', help='Uses all object categories in shapenet.')
    parser.add_argument('--check_symmetry', action='store_true', help='Whether to check if the objects have ' +
                                                                      'a plane of symmetry. The plane of symmetry ' +
                                                                      'will be assumed to be plane YZ if the option' +
                                                                      'is not present.')

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    dataset = ShapeNetSymmetryDataset(args.datadir, args.num_samples, args.max_num_triangles)
    progress_fn = os.path.join(Path(args.save_path), 'progress.pkl')

    if args.dataset_type == ['tfrecord']:
        generation_function_train = dataset.generate_tfRecord_train
        generation_function_test = dataset.generate_tfRecord_test
    else:
        generation_function_train = dataset.generate_h5_train
        generation_function_test = dataset.generate_h5_test

    # Continues where prior execution left
    if os.path.exists(progress_fn):

        # Reads the progress file
        with open(progress_fn, 'rb') as pickle_file:
            current_state = pickle.load(pickle_file)

        # set the list of files to process from the progress file
        dataset.set_trainfiles(current_state['train']['files'])
        dataset.set_testfiles(current_state['test']['files'])

        # set the progress variables
        train_offset = current_state['train']['offset']
        test_offset = current_state['test']['offset']
        train_file_counter = current_state['train']['file_counter']
        test_file_counter = current_state['test']['file_counter']
        train_done = train_offset >= len(dataset.trainfiles)
        test_done = test_offset >= len(dataset.testfiles)

        # if there still files left to process in the training set
        if not train_done:
            train_file_counter += 1
            train_offset += generation_function_train(args.save_path, ext=args.ext,
                                                rotate=str2bool(args.rotate),
                                                shuffle_data=False,
                                                offset=train_offset,
                                                file_counter=train_file_counter,
                                                rot_type=args.rot_type,
                                                objs_per_file=args.num_objs_per_file,
                                                check_symmetry=args.check_symmetry)
            train_done = train_offset >= len(dataset.trainfiles)

        if train_done and not test_done:
            test_offset += generation_function_test(args.save_path, ext=args.ext,
                                             rotate=str2bool(args.rotate),
                                             offset=test_offset,
                                             file_counter=test_file_counter,
                                             rot_type=args.rot_type,
                                             objs_per_file=args.num_objs_per_file,
                                             check_symmetry=args.check_symmetry)
            test_file_counter += 1
        test_done = test_offset >= len(dataset.testfiles)

    else:
        # get a list of training files
        dataset.get_trainfiles(ext=args.ext, synset_id=args.synset_id, use_all_cateogories=args.use_all_categories)
        dataset.get_testfiles(ext=args.ext, synset_id=args.synset_id, use_all_cateogories=args.use_all_categories)

        train_offset = 0
        test_offset = 0
        train_file_counter = 0
        test_file_counter = 0

        train_offset = generation_function_train(args.save_path, ext=args.ext, rotate=str2bool(args.rotate), shuffle_data=False, rot_type=args.rot_type, objs_per_file=args.num_objs_per_file, check_symmetry=args.check_symmetry)
        train_done = train_offset >= len(dataset.trainfiles)

        if train_done:
            test_offset = generation_function_test(args.save_path, ext=args.ext, rotate=str2bool(args.rotate), rot_type=args.rot_type, objs_per_file=args.num_objs_per_file, check_symmetry=args.check_symmetry)

        test_done = test_offset >= len(dataset.testfiles)

    current_state = {
        'train': {
            'done': train_done,
            'files': dataset.trainfiles,
            'offset': train_offset,
            'file_counter': train_file_counter,
        },
        'test': {
            'done': test_done,
            'files': dataset.testfiles,
            'offset': test_offset,
            'file_counter': test_file_counter
        }
    }

    with open(progress_fn, 'wb') as pickle_file:
        pickle.dump(current_state, pickle_file)

    # demonstrates use of the file
    if args.dataset_type == 'tfrecord':
        tfdataset = tf.data.TFRecordDataset(args.save_path + '/train1024_0.tfrecord')
        tfdataset = tfdataset.map(map_func=ShapeNetSymmetryDataset.from_tfrecord)

        iterator = tfdataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:
            for i in range(2):
                value = sess.run(next_element)
                print(value[0].shape, value[1].shape, value[2].shape, value[3], value[4])
                print(type(value[0]), type(value[1]), type(value[2]), type(value[3]))
                print(type(value[0][0]), type(value[1][0]), type(value[2][0]), type(value[3]), value[3])
