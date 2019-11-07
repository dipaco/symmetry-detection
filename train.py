'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))
import provider
import tf_util
import tensorboard_logging
import modelnet_dataset
import modelnet_h5_dataset
import shapenet_symmetry

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name [default: pointnet2_cls_ssg]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--dataset_dir', default='data', help='Folder where the files are stores [default: data]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--train_size', type=int, help='Number of elements in the train size.')
parser.add_argument('--create_figures', action='store_true')
parser.add_argument('--augment', action='store_true')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
DATASET_DIR = FLAGS.dataset_dir
PC_IDX = 3

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 40

TIMEOUT_TERMINATION_SECS = 3600
script_starting_time = time.time()

'''# Shapenet official train/test split
if FLAGS.normal:
    assert(NUM_POINT<=10000)
    DATA_PATH = os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled')
    TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
    TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
else:
    assert(NUM_POINT<=2048)
    TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True)
    TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)
'''

# loads the dataset
assert(NUM_POINT<=2048)
TRAIN_DATASET = shapenet_symmetry.ShapenetSymmetryDataset(os.path.join(DATASET_DIR, 'train_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True)
TEST_DATASET = shapenet_symmetry.ShapenetSymmetryDataset(os.path.join(DATASET_DIR, 'test_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.000000001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():

    global EPOCH_CNT

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            global_step = tf.train.get_or_create_global_step()
            epoch_var = tf.Variable(0, trainable=False, name='epoch', dtype=tf.int32)   # epoch counter
            bn_decay = get_bn_decay(global_step)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            MODEL.get_loss(pred, labels_pl, end_points)
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
            for l in losses + [total_loss]:
                tf.summary.scalar(l.op.name, l)

            print("--- Get training operator")
            # Get training operator
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=global_step)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(save_relative_paths=True)
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # if a checkpoint exists, restore from the latest checkpoint
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(LOG_DIR, 'checkpoint')))
        print('----', os.path.abspath(os.path.join(LOG_DIR, 'checkpoint')), ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('restoring global_step={}'.format(tf.train.global_step(sess, global_step)))
            print('restoring epoch={}'.format(sess.run(epoch_var)))

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # gets current epoch integer
        cur_epoch = sess.run(epoch_var)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': global_step,
               'end_points': end_points}

        best_acc = -1
        for epoch in range(cur_epoch, MAX_EPOCH):
            EPOCH_CNT = epoch
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # increase epoch counter
            increase_epoch = tf.assign(epoch_var, epoch + 1)
            sess.run(increase_epoch)

            # Kills the process after one hour of processing.
            if time.time() - script_starting_time > TIMEOUT_TERMINATION_SECS:
                save_progress(global_step, saver, sess)
                print('Time out termination.')
                exit()

            elif (epoch + 1) % 10 == 0:   # Save the variables to disk.
                save_progress(global_step, saver, sess)


def save_progress(global_step, saver, sess):
    save_path = saver.save(sess, os.path.join(LOG_DIR, 'checkpoint'), global_step=global_step)
    log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string(str(datetime.now()))

    tb_logger = tensorboard_logging.Logger(train_writer)

    # Make sure batch data is of same size
    cur_batch_gt_points = np.zeros((BATCH_SIZE, NUM_POINT, TRAIN_DATASET.num_channel()))
    cur_batch_points = np.zeros((BATCH_SIZE, NUM_POINT, TRAIN_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE,  3))
    cur_batch_cut_plane = np.zeros((BATCH_SIZE, 4))

    loss_sum = 0
    batch_idx = 0
    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=FLAGS.augment)

        bsize = batch_data.shape[0]
        cur_batch_gt_points[0:bsize, ...] = batch_data[:, 0, ...]
        cur_batch_points[0:bsize,...] = batch_data[:, PC_IDX, ...]
        cur_batch_label[0:bsize, ...] = batch_label[:, 0, 0:3]
        cur_batch_cut_plane[0:bsize, ...] = batch_label[:, PC_IDX, :]

        feed_dict = {ops['pointclouds_pl']: cur_batch_points,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val, end_points = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred'], ops['end_points']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        loss_sum += loss_val
        if (batch_idx+1) % 50 == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            log_string('mean loss: %f' % (loss_sum / 50))
            loss_sum = 0

            if FLAGS.create_figures:
                MODEL.create_figures(FLAGS, step, tb_logger, cur_batch_gt_points, cur_batch_cut_plane, end_points['l0_xyz'], end_points['reflected_l0_xyz'], pred_val, cur_batch_label)

        batch_idx += 1

    TRAIN_DATASET.reset()
        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    tb_logger = tensorboard_logging.Logger(test_writer)

    # Make sure batch data is of same size

    cur_batch_gt_points = np.zeros((BATCH_SIZE, NUM_POINT, TRAIN_DATASET.num_channel()))
    cur_batch_points = np.zeros((BATCH_SIZE, NUM_POINT,TEST_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE, 3))
    cur_batch_cut_plane = np.zeros((BATCH_SIZE, 4))

    loss_sum = 0
    batch_idx = 0
    shape_ious = []
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    all_gt_points = None
    all_end_points = None
    all_reflected_points = None
    all_pred_vals = None
    all_labels = None
    all_cut_planes = None
    
    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_gt_points[0:bsize, ...] = batch_data[:, 0, ...]
        cur_batch_points[0:bsize,...] = batch_data[:, PC_IDX, ...]
        cur_batch_label[0:bsize, ...] = batch_label[:, 0, 0:3]
        cur_batch_cut_plane[0:bsize, ...] = batch_label[:, PC_IDX, :]

        feed_dict = {ops['pointclouds_pl']: cur_batch_points,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val, end_points = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred'], ops['end_points']], feed_dict=feed_dict)

        all_gt_points = cur_batch_gt_points if cur_batch_gt_points is None else np.vstack((all_gt_points, cur_batch_gt_points))
        all_end_points = end_points['l0_xyz'] if all_end_points is None else np.vstack((all_end_points, end_points['l0_xyz']))
        all_reflected_points = end_points['reflected_l0_xyz'] if all_reflected_points is None else np.vstack((all_reflected_points, end_points['reflected_l0_xyz']))
        all_pred_vals = pred_val if all_pred_vals is None else np.vstack((all_pred_vals, pred_val))
        all_cut_planes = cur_batch_cut_plane if all_cut_planes is None else np.vstack((all_cut_planes, cur_batch_cut_plane))

        test_writer.add_summary(summary, step)
        loss_sum += loss_val
        batch_idx += 1

    if FLAGS.create_figures:
        MODEL.create_figures(FLAGS, step, tb_logger, all_gt_points, all_cut_planes, all_end_points, all_reflected_points, all_pred_vals, all_labels)
    
    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    EPOCH_CNT += 1

    TEST_DATASET.reset()


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
