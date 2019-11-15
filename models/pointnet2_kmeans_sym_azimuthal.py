"""
    PointNet++ Model for point clouds classification
"""

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/nn_distance'))
import tensorflow as tf
import numpy as np
import tf_util
import vis_util
import math
from pointnet_util import pointnet_sa_module
from tf_nndistance import nn_distance

def placeholder_inputs(batch_size, num_point, num_clusters):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    cluster_labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_clusters))
    return pointclouds_pl, labels_pl, cluster_labels_pl

def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # Set abstraction layers
    # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
    # So we only use NCHW for layer 1 until this issue can be resolved.
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_nchw=True)
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, 2, activation_fn=None, scope='fc3')

    # attach a vector of zeros to account for the fixed value in the third dimension
    net = tf.concat([net, tf.zeros([batch_size, 1])], axis=-1)

    return net, end_points

def pairwise_l2_norm2(x, y, scope=None):
    with tf.op_scope([x, y], scope, 'pairwise_l2_norm2'):
        batch_size = tf.shape(x)[0]
        size_x = tf.shape(x)[1]
        size_y = tf.shape(y)[1]
        #xx = tf.expand_dims(x, -1)
        #xx = tf.tile(xx, tf.pack([1, 1, size_y]))
        xx = x[:, :, None, :]

        yy = tf.expand_dims(y, -1)
        #yy = tf.tile(yy, tf.pack([1, 1, size_x]))
        #yy = tf.transpose(yy, perm=[2, 1, 0])
        yy = y[:, None, :, :]

        diff = xx - yy

        #diff = tf.sub(xx, yy)
        square_dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1))

        return square_dist

def get_loss(pred_plane, gt_plane, input_points, cluster_labels_sparse):
    """ pred: B*N*3,
        label: B*N*3,

        Uses Householder transformation to reflect the original_points around the plane <pred>
    """
    n_points = tf.shape(input_points['l0_xyz'])[1]
    y_true = tf.nn.l2_normalize(gt_plane, axis=-1)
    y_pred = tf.nn.l2_normalize(pred_plane, axis=-1)

    # creates the reflexion matrix
    T = tf.eye(3) - 2 * tf.einsum('bi, bj -> bij', y_pred, y_pred)

    # reflects the original point cloud
    reflected_point_cloud = tf.einsum('bic, bpc -> bpi', T, input_points['l0_xyz'])
    input_points['reflected_l0_xyz'] = reflected_point_cloud


    cluster_labels_sparse = 2 * cluster_labels_sparse - tf.ones_like(cluster_labels_sparse)
    cluster_labels_sparse = cluster_labels_sparse[:, None, ...]
    #cluster_labels_sparse = tf.cast(cluster_labels_sparse, tf.float32)
    cdist = pairwise_l2_norm2(input_points['l0_xyz'], reflected_point_cloud)
    cdist = cdist[..., None]

    #removing the elements that do not belong to the cluster
    d_matrix = cdist * cluster_labels_sparse
    #d_matrix = d_matrix - tf.eye(n_points)[None, ..., None]
    d_matrix = tf.maximum(0.0, d_matrix)

    # computing the min distance in the cluster
    min_dist = tf.reduce_min(d_matrix, axis=2)

    cluster_chamfer_loss = tf.reduce_mean(min_dist)

    #dists_forward, _, dists_backward, _ = nn_distance(reflected_point_cloud, input_points['l0_xyz'])

    #chamfer_loss = tf.reduce_mean(dists_forward + dists_backward)


    cosine_similarity = tf.abs(tf.reduce_sum(y_true * y_pred, axis=-1))
    cosine_similarity_loss = 1 - tf.reduce_mean(cosine_similarity)
    average_error_angle = tf.reduce_mean(tf.math.acos(cosine_similarity) * 180 / math.pi)

    tf.summary.scalar('Cluster Chamfer loss', cluster_chamfer_loss)
    tf.summary.scalar('Mean error angle', average_error_angle)
    tf.add_to_collection('losses', cluster_chamfer_loss)

    return cluster_chamfer_loss


def create_figures(FLAGS, step, tb_logger, cur_batch_gt_points, cur_batch_cut_plane, points, reflected_points, pred_plane, gt_plane):
    figs_filenames = vis_util.gen_symmetry_fig(FLAGS, step, cur_batch_gt_points, cur_batch_cut_plane, points, reflected_points, pred_plane, gt_plane)
    tb_logger.log_images('figs', step, images_filenames=figs_filenames)


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        output, _ = get_model(inputs, tf.constant(True))
        print(output)
