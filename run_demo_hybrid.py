import argparse
import numpy as np
import os
import sys
import tensorflow as tf
import yaml

from lib.utils import load_graph_data
from model.hybrid_supervisor import HybridSupervisor


def run_demo(args):
    with open(args.config_filename) as f:
        config = yaml.load(f)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    graph_pkl_filename = config['data']['graph_pkl_filename']
    _, _, adj_mx = load_graph_data(graph_pkl_filename)
    with tf.Session(config=tf_config) as sess:
        supervisor = HybridSupervisor(adj_mx=adj_mx, **config)
        supervisor.load(sess, config['train']['model_filename'])
        if args.eval == 'spatiotemporal':
            supervisor.spatiotemporal_evaluate(sess, horizon=1)
        elif args.eval == 'r2':
            supervisor.r2_evaluate(sess, horizon=12)
        else:
            supervisor.evaluate(sess)
        


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='', type=str, help='Set GPU to use.')
    parser.add_argument('--config_filename', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--eval', default='', type=str, help='Which evaluate to use')
    args = parser.parse_args()
    run_demo(args)
