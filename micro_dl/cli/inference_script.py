#!/usr/bin/env/python
"""Model inference"""
import argparse
import os
import pandas as pd
import pickle
import yaml

from micro_dl.input.dataset import BaseDataSet, DataSetWithMask
from micro_dl.train.model_inference import ModelEvaluator
from micro_dl.utils.train_utils import check_gpu_availability
from micro_dl.train.model_inference import predict_on_larger_image

import numpy as np


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help=('specify the gpu to use: 0,1,...',
                              ', -1 for debugging'))
    parser.add_argument('--gpu_mem_frac', type=float, default=1.,
                        help='specify the gpu memory fraction to use')
    parser.add_argument('--config', type=str,
                        help='path to yaml configuration file')
    parser.add_argument('--model_fname', type=str, default=None,
                        help='fname with full path to model weights .hdf5')

    parser.add_argument('--num_batches', type=int, default=2,
                        help='run prediction on tiles for num_batches')

    parser.add_argument('--flat_field', dest='flat_field', action='store_true',
                        help='Indicator to correct for flat field')

    parser.add_argument('--no_flat_field', dest='flat_field',
                        action='store_false')

    parser.set_defaults(flat_field=True)

    parser.add_argument('--focal_plane_idx', type=int, default=0,
                        help='idx for focal plane')
    parser.add_argument('--base_image_dir', type=str, default=None,
                        help='base dir with whole/entire images')

    parser.add_argument('--image_meta_fname', type=str, default=None,
                        help='csv holding meta for all images in study')

    args = parser.parse_args()
    return args


def run_inference(args):
    """Evaluate model performance"""

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    df_test = pd.read_csv(os.path.join(config['trainer']['model_dir'],
                                       'test_metadata.csv'))

    if 'masked_loss' in config['trainer']:
        ds_test = DataSetWithMask(input_fnames=df_test['fpaths_input'],
                                  target_fnames=df_test['fpaths_target'],
                                  mask_fnames=df_test['fpaths_mask'],
                                  batch_size=config['trainer']['batch_size'])
    else:
        ds_test = BaseDataSet(input_fnames=df_test['fpaths_input'],
                              target_fnames=df_test['fpaths_target'],
                              batch_size=config['trainer']['batch_size'])

    ev_inst = ModelEvaluator(config,
                             model_fname=args.model_fname,
                             gpu_ids=args.gpu,
                             gpu_mem_frac=args.gpu_mem_frac)
    test_perf_metrics = ev_inst.evaluate_model(ds_test)

    ev_inst.predict_on_tiles(ds_test, nb_batches=args.num_batches)
    idx_fname = os.path.join(config['trainer']['model_dir'],
                             'split_samples.pkl')
    with open(idx_fname, 'rb') as f:
        split_samples = pickle.load(f)

    image_meta = pd.read_csv(args.image_meta_fname)
    # for regression tasks change place_operation to 'mean'
    print("config", config)

    image_test = np.load('/data/norah/small_set/split_images/timepoint_0/channel_3/image_n100_z1.npy')
    print(image_test.shape)
    image_test = np.expand_dims(image_test, axis=0)
    print(image_test.shape)

    predict_on_larger_image(config['network'], args.model_fname,image_test)
    # ev_inst.predict_on_larger_image(image_meta=image_meta,
    #                              test_samples=split_samples['test'],
    #                              focal_plane_idx=args.focal_plane_idx,
    #                             flat_field_correct=args.flat_field,
    #                             base_image_dir=args.base_image_dir,
    #                             place_operation='max')
    # ev_inst.predict_on_larger_image(image_meta=image_meta,
    #                               test_samples=split_samples['test'],
    #                               focal_plane_idx=args.focal_plane_idx,
    #                               flat_field_correct=args.flat_field,
    #                               base_image_dir=args.base_image_dir,
    #                               place_operation='max')

    if config['performance']['save_metrics']:
        save_metrics(test_perf_metrics, config)

    return test_perf_metrics


def save_metrics(test_perf_metrics, config):
    """Save model performance in separate csv

    :param test_perf_metrics: Model performance
    :param config: Config yml
    :return:
    """
    performance_csv = os.path.join(config['performance']['model_performance_dir'], 'final_models.csv')

    if not os.path.exists(performance_csv):
        df_performance = pd.DataFrame(columns=['file_path', 'model', 'loss', 'performance'])
        df_performance.to_csv(performance_csv, index=False)

    df_new_performance = pd.DataFrame(columns=['file_path', 'model', 'loss', 'performance'])
    df_new_performance.at[0, 'file_path'] = config['performance']['data_dir'].split('/')[-2]
    df_new_performance.at[0, 'model'] = args.model_fname.split('/')[-1]
    df_new_performance.at[0, 'loss'] = test_perf_metrics[0]
    df_new_performance.at[0, 'performance'] = test_perf_metrics[1]

    # metrics_dict = {metric:value for metric, value in zip(test_perf_metrics, config['trainer']['metrics'])}

    df_new_performance.to_csv(performance_csv, mode='a', header=False, index=False)


if __name__ == '__main__':
    args = parse_args()
    gpu_available = False
    assert isinstance(args.gpu, int)
    if args.gpu >= 0:
        gpu_available = check_gpu_availability(args.gpu, args.gpu_mem_frac)
    if gpu_available:
        model_perf = run_inference(args)
        print('model performance on test images:', model_perf)
