'''
    Evaluate classification performance
'''
import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import sys
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import tf_util
import dataloader
from dict_restore import DictRestore
import spec_transforms
import target_transforms

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name. [default: pointnet2_cls_ssg]')
parser.add_argument('--data', default='', help='Data dir [default: ]')
parser.add_argument('--num_segs', type=int, default=8, help='The number of frames unsed [default: 8]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--freqbins', type=int, default=128, help='Video image freqbins [default: 112]')
parser.add_argument('--timebins', type=int, default=80, help='Video image timebins [default: 112]')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes [default: 400]')
parser.add_argument('--num_threads', type=int, default=24, help='Number of threads to use in loading data [default: 24]')
parser.add_argument('--sn', type=int, default=4, help='Number of Semantic Neighbors [default: 4]')
parser.add_argument('--fcn', type=int, default=0, help='Whether to use all spatial in evaluation [default: 0]')
parser.add_argument('--command_file', default=None, help=' [Shell command file to use default: None]')
FLAGS = parser.parse_args()

sys.path.append(os.path.dirname(FLAGS.model_path))

random.seed(0)
np.random.seed(0)
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

NUM_GPUS = len(FLAGS.gpu.split(','))

MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_SEGS = FLAGS.num_segs
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
DATA = FLAGS.data
FREQBINS = FLAGS.freqbins
TIMEBINS = FLAGS.timebins
NUM_THREADS = FLAGS.num_threads
SN = FLAGS.sn
COMMAND_FILE = FLAGS.command_file
FCN = FLAGS.fcn

MODEL_FILE = os.path.join(os.path.dirname(FLAGS.model_path), FLAGS.model+'.py')
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
os.system('cp %s %s ' % (__file__, DUMP_DIR)) # bkp of evaluation file
os.system('cp %s %s ' % (COMMAND_FILE, DUMP_DIR)) # bkp of command shell file
os.system('cp %s %s' % (MODEL_FILE, DUMP_DIR)) # bkp of model def
os.system('cp utils/net_utils.py %s ' % (DUMP_DIR)) # bkp of net_utils file
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = FLAGS.num_classes

HOSTNAME = socket.gethostname()

# validation transform
normalize = spec_transforms.ToNormalizedTensor()
val_transform = spec_transforms.Compose([normalize])
target_transform = target_transforms.ClassLabel()

loader_bsize = 1

if FCN > 1:
    
    _, _, val_loader = dataloader.get_loader(root=DATA,
                                             train_transform=None,
                                             val_transform=val_transform,
                                             target_transform=target_transform,
                                             batch_size=loader_bsize,
                                             num_segs=NUM_SEGS,
                                             val_samples=1,
                                             n_threads=NUM_THREADS,
                                             training=False, val=False, test=True)
else:
    _, val_loader = dataloader.get_loader(root=DATA,
                                          train_transform=None,
                                          val_transform=val_transform,
                                          target_transform=target_transform,
                                          batch_size=loader_bsize,
                                          num_segs=NUM_SEGS,
                                          val_samples=1,
                                          n_threads=NUM_THREADS,
                                          training=False, val=True, test=False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    with tf.Graph().as_default():
        is_training = False

        if FCN == 5:
            pl_bsize = 5
        elif FCN == 10:
            pl_bsize = 10
        elif FCN == 2:
            pl_bsize = 2
        else:
            pl_bsize = 1
        assert(pl_bsize % NUM_GPUS == 0)
        DEVICE_BATCH_SIZE = pl_bsize // NUM_GPUS

        audio_pl, labels_pl = MODEL.placeholder_inputs(pl_bsize, NUM_SEGS, FREQBINS, TIMEBINS, evaluate=True)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        MODEL.get_model(audio_pl, is_training_pl, NUM_CLASSES, sn=SN)
        pred_gpu = []
        for i in range(NUM_GPUS):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                with tf.device('/gpu:%d'%(i)) as scope:
                    vd_batch = tf.slice(audio_pl,
                        [i*DEVICE_BATCH_SIZE,0,0,0,0], [DEVICE_BATCH_SIZE,-1,-1,-1,-1])
                    label_batch = tf.slice(labels_pl,
                        [i*DEVICE_BATCH_SIZE], [DEVICE_BATCH_SIZE])

                    pred, end_points = MODEL.get_model(vd_batch, is_training_pl, NUM_CLASSES, sn=SN)
                    pred_gpu.append(pred)
        pred = tf.concat(pred_gpu, 0)

        saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # Restore variables from disk.
        if MODEL_PATH is not None:
            if 'npz' not in MODEL_PATH:
                saver.restore(sess, MODEL_PATH)
                log_string("Model restored.")
            else:
                dict_file = np.load(MODEL_PATH)
                dict_for_restore = {}
                dict_file_keys = dict_file.keys()
                for k in dict_file_keys:
                    dict_for_restore[k] = dict_file[k]
                dict_for_restore = MODEL.name_mapping(dict_for_restore)
                dict_for_restore = MODEL.convert_2d_3d(dict_for_restore)
                dr = DictRestore(dict_for_restore, log_string)
                dr.run_init(sess)
                log_string("npz file restored.")

        ops = {'audio_pl': audio_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred}

        eval_one_epoch(sess, ops, val_loader)

def eval_one_epoch(sess, ops, val_loader, topk=1):
    is_training = False

    total_correct_top1 = 0
    total_correct_top5 = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    loss_sum = 0
    batch_idx = 0

    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_data = inputs.data.numpy()
        bsize = batch_data.shape[0]
        batch_label = targets.data.numpy()
        batch_data = np.transpose(batch_data, [0,2,3,4,1])

        if FCN == 10:
            preds = []
            for i in range(bsize):
                batch_data_split = np.expand_dims(batch_data[i], 0)
                batch_label_split = np.expand_dims(batch_label[i], 0)
                batch_data_split = np.concatenate([ \
                        batch_data_split[:,::2,:,:,:], \
                        batch_data_split[:,::2,:,:,:][:,::-1,:,:,:], \
                        batch_data_split[:,1::2,:,:,:], \
                        batch_data_split[:,1::2,:,:,:][:,::-1,:,:,:], \
                        batch_data_split[:,:NUM_SEGS,:,:,:], \
                        batch_data_split[:,:NUM_SEGS,:,:,:][:,::-1,:,:,:], \
                        batch_data_split[:,-NUM_SEGS:,:,:,:], \
                        batch_data_split[:,-NUM_SEGS:,:,:,:][:,::-1,:,:,:], \
                        batch_data_split[:,(NUM_SEGS // 2):(NUM_SEGS // 2 + NUM_SEGS),:,:,:], \
                        batch_data_split[:,(NUM_SEGS // 2):(NUM_SEGS // 2 + NUM_SEGS),:,:,:][:,::-1,:,:,:] ], \
                        axis=0)
                batch_label_split = np.concatenate([batch_label_split] * 10, axis=0)
                feed_dict = {ops['audio_pl']: batch_data_split,
                             ops['labels_pl']: batch_label_split,
                             ops['is_training_pl']: is_training}

                pred_val = sess.run(ops['pred'], feed_dict=feed_dict)
                preds.append(pred_val)
            pred_val = np.concatenate(preds, 0)
        elif FCN == 5:
            preds = []
            for i in range(bsize):
                batch_data_split = np.expand_dims(batch_data[i], 0)
                batch_label_split = np.expand_dims(batch_label[i], 0)
                batch_data_split = np.concatenate([ \
                        batch_data_split[:,::2,:,:,:], \
                        batch_data_split[:,1::2,:,:,:], \
                        batch_data_split[:,:NUM_SEGS,:,:,:], \
                        batch_data_split[:,-NUM_SEGS:,:,:,:], \
                        batch_data_split[:,(NUM_SEGS // 2):(NUM_SEGS // 2 + NUM_SEGS),:,:,:] ], \
                        axis=0)
                batch_label_split = np.concatenate([batch_label_split] * 5, axis=0)
                feed_dict = {ops['audio_pl']: batch_data_split,
                             ops['labels_pl']: batch_label_split,
                             ops['is_training_pl']: is_training}

                pred_val = sess.run(ops['pred'], feed_dict=feed_dict)
                preds.append(pred_val)
            pred_val = np.concatenate(preds, 0)
        elif FCN == 2:
            preds = []
            for i in range(bsize):
                batch_data_split = np.expand_dims(batch_data[i], 0)
                batch_label_split = np.expand_dims(batch_label[i], 0)
                batch_data_split = np.concatenate([ \
                        batch_data_split[:,:,:,:,:], \
                        batch_data_split[:,:,:,:,:][:,::-1,:,:,:] ], \
                        axis=0)
                batch_label_split = np.concatenate([batch_label_split] * 2, axis=0)
                feed_dict = {ops['audio_pl']: batch_data_split,
                             ops['labels_pl']: batch_label_split,
                             ops['is_training_pl']: is_training}

                pred_val = sess.run(ops['pred'], feed_dict=feed_dict)
                preds.append(pred_val)
            pred_val = np.concatenate(preds, 0)
        else:
            feed_dict = {ops['audio_pl']: batch_data,
                         ops['labels_pl']: batch_label,
                         ops['is_training_pl']: is_training}

            pred_val = sess.run(ops['pred'], feed_dict=feed_dict)

        pred_val = np.exp(pred_val - np.max(pred_val, axis=1, keepdims=True))
        pred_val = pred_val / np.sum(pred_val, axis=1, keepdims=True)

        pred_val = np.mean(pred_val, axis=0)

        pred_val_top5 = np.argsort(pred_val)[::-1][:5]
        pred_val_top1 = np.argmax(pred_val)

        correct_top1 = pred_val_top1 == batch_label[0]
        correct_top5 = np.sum(pred_val_top5 == batch_label[0])
        total_correct_top1 += correct_top1
        total_correct_top5 += correct_top5

        total_seen += 1
        log_string('batch accuracy top1 : %f, batch accuracy top5: %f'% (correct_top1, correct_top5))

        l = batch_label[0]
        total_seen_class[l] += 1
        total_correct_class[l] += correct_top1

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy top1 : %f' % (total_correct_top1 / float(total_seen)))
    log_string('eval accuracy top5 : %f' % (total_correct_top5 / float(total_seen)))
    np.savez_compressed(os.path.join(DUMP_DIR, 'pred_class.npz'), total_correct_class=np.array(total_correct_class), total_seen_class=np.array(total_seen_class))


if __name__=='__main__':
    evaluate()
    LOG_FOUT.close()
