import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from PIL import Image
import pickle
import numpy as np
import os
from tqdm import tqdm
import random
import time
import datetime
import yaml
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam
from torch.multiprocessing import set_start_method
from torch.nn.utils import clip_grad_norm_

try:
    set_start_method('spawn')
except RuntimeError:
    pass

from bevodom2.utils import geom
from bevodom2.modules.utils.monitor import MonitorBase

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.getLogger('mmcv').setLevel(logging.WARNING)

def list_and_sort_files(folder_path):
    file_names = os.listdir(folder_path)
    sorted_file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))
    sorted_file_names_without_extension = [os.path.splitext(file_name)[0] for file_name in sorted_file_names]
    return sorted_file_names_without_extension

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def merge_configs(defaults, overrides):
    for key, value in overrides.items():
        if key in defaults and isinstance(defaults[key], dict) and isinstance(value, dict):
            merge_configs(defaults[key], value)
        else:
            defaults[key] = value
    return defaults

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="load yaml")
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='yaml path'
    )
    parser.add_argument(
        '-g', '--gpu',
        type=str,
        required=True,
        help='Comma-separated list of GPU IDs to use, e.g. "0,1,2"'
    )
    args = parser.parse_args()
    gpu_ids = args.gpu.split(',')
    print("num of gpus:", len(gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_ids)
    MORE_GPU = len(gpu_ids) > 1
    config = load_config(args.config)

    # Extract configurations
    model_conf = config.get('model_conf', {'use_pretrained_model': True})
    backbone_conf = config.get('backbone_conf', {})
    head_conf = config.get('head_conf', {})
    matching_conf = config.get('matching_conf', {})
    training_params = config.get('training_params', {})

    # ========== unified model import ==========
    from bevodom2.models.model import BaseBEVODOM2, DepthLoss

    use_6dof = training_params.get('use_6dof', True)
    corr_patch_size = training_params.get('corr_patch_size', 11)
    use_leakyrelu_bn = training_params.get('use_leakyrelu_bn', False)

    from bevodom2.train_model.dataset_nclt import NCLTDataset, NCLTDataset_sequences
    from bevodom2.train_model.dataset_oxford import OxfordSequence, OxfordSequences
    # ==================================================================

    # Set random seeds for reproducibility
    seed = training_params.get('seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime("%m%d_%H%M%S")

    NUM_WORKERS = training_params.get('NUM_WORKERS', 6)
    batch_size_train = training_params.get('batch_size_train', 4)
    batch_size_test = training_params.get('batch_size_test', 1)
    epoch_save = training_params.get('epoch_save', 1)

    NO_DEPTH = training_params.get('NO_DEPTH', True)

    DEBUG = training_params.get('DEBUG', False)
    JUST_TEST = training_params.get('JUST_TEST', False)

    feature_size = training_params.get('feature_size', 128)
    division = training_params.get('division', 0.8)
    matched_depth = training_params.get('matched_depth', False)
    dataset_type = training_params.get('dataset_type', 'NCLT')
    dataset_item_range = training_params.get('dataset_item_range', '0-2')
    pickle_name = training_params.get('pickle_name', 'pickle_d0_d2_r15_r45.txt')
    featuresize_before_corr = training_params.get('featuresize_before_corr', '64*64_31')
    max_dis = training_params.get('max_dis', 4.0)

    IS_MONO = training_params.get('IS_MONO', True)
    IS_MONO_CUT = training_params.get('IS_MONO_CUT', True)

    other_downsample_factor = training_params.get('other_downsample_factor', 16)


    USE_PICKLE = training_params.get('USE_PICKLE', False)
    ALL_DATA_INPUT = training_params.get('ALL_DATA_INPUT', True)
    FREEZE_BEV = training_params.get('FREEZE_BEV', False)
    NO_BN = training_params.get('NO_BN', False)
    USE_TRAIN_EVAL = training_params.get('USE_TRAIN_EVAL', True)
    NO_WEIGHT_SCORES = training_params.get('NO_WEIGHT_SCORES', False)
    USE_R_ENHANCE = training_params.get('USE_R_ENHANCE', False)
    max_norm = training_params.get('max_norm', 1.0)
    USE_GRAD_CLIP = training_params.get('USE_GRAD_CLIP', False)
    USE_MORE_SEQUENCES = training_params.get('USE_MORE_SEQUENCES', True)
    USE_REVERSE = training_params.get('USE_REVERSE', False)
    depthloss_rate = training_params.get('depthloss_rate', 1)
    USE_TEST_DIS_THRES = training_params.get('USE_TEST_DIS_THRES', False)
    TEST_DIS_THRES = training_params.get('TEST_DIS_THRES', 1)
    SPLIT_R_t = training_params.get('SPLIT_R_t', False)
    lr_start = training_params.get('lr_start', 0.0001)

    wrap_level_num = training_params.get('wrap_level_num', 1)
    freeze_bev = training_params.get('freeze_bev', False)
    freeze_steps = training_params.get('freeze_steps', 0)
    pvloss_rate = training_params.get('pvloss_rate', 0.2)

    if MORE_GPU:
        batch_size_train *= len(gpu_ids)
        batch_size_test *= len(gpu_ids)

    if FREEZE_BEV:
        feature_size = 128

    # Experiment name
    NAME = "25" + current_time_str + "_" + dataset_type + "_BEVLOC_" + "_" + str(division) + "m-pix_" + str(feature_size) + "_" + dataset_item_range
    if NO_DEPTH:
        NAME = NAME + "_NO_DEPTH"
    if IS_MONO:
        if IS_MONO_CUT:
            NAME = NAME + "_MONO_CUT"
        else:
            NAME = NAME + "_MONO"
    if other_downsample_factor != 16:
        NAME = NAME + "_DOWN_" + str(other_downsample_factor)
    if freeze_bev:
        NAME = NAME + "_freeze_bevNsteps"

    NAME = NAME + "_max_dis_" + str(max_dis)

    NAME = NAME + "_USE_" + featuresize_before_corr

    if matched_depth:
        NAME = NAME + "_MD"

    ckpt_path = training_params.get('ckpt_path', None)
    START = 0
    output_root = training_params.get('output_root', str(PROJECT_ROOT / "outputs"))
    model_save_root = os.path.join(output_root, "model_save")
    evo_output_root = os.path.join(output_root, "evo")
    data_root_nclt = training_params.get('data_root_nclt', "/data/wyf/NCLT/format_data")
    data_root_oxford = training_params.get('data_root_oxford', "/data/wyf/oxford")
    pickle_root_nclt = training_params.get('pickle_root_nclt', "./data/pickle/NCLT")
    pickle_root_oxford = training_params.get('pickle_root_oxford', "./data/pickle/oxford")
    os.makedirs(model_save_root, exist_ok=True)
    os.makedirs(evo_output_root, exist_ok=True)

    if ckpt_path is None and 'NAME' in training_params and training_params.get('START', 0) > 0:
        NAME = training_params['NAME']
        START = training_params['START']
        ckpt_path = os.path.join(model_save_root, "model_" + str(NAME), "model_" + str(START) + ".pth")
        if os.path.exists(ckpt_path):
            freeze_bev = False
        else:
            print(f"Resume checkpoint not found, start from scratch: {ckpt_path}")
            ckpt_path = None

    matching_conf["log_dir"] = os.path.join(output_root, f"log_{NAME}") + "/"
    os.makedirs(matching_conf["log_dir"], exist_ok=True)
    log_file_path = os.path.join(matching_conf["log_dir"], "loss_record.txt")

    if feature_size == 256 and division == 0.1:
        matching_conf["cart_resolution"] = 0.1
        matching_conf["cart_pixel_width"] = 256
        matching_conf["networks"]["keypoint_block"]["patch_size"] = 8
        backbone_conf['x_bound'] = [-12.8, 12.8, 0.1]
        backbone_conf['y_bound'] = [-12.8, 12.8, 0.1]
        backbone_conf['d_bound'] = [2.0, 16, 0.125]
    if feature_size == 256 and division == 0.2:
        matching_conf["cart_resolution"] = 0.2
        matching_conf["cart_pixel_width"] = 256
        matching_conf["networks"]["keypoint_block"]["patch_size"] = 8
        backbone_conf['x_bound'] = [-25.6, 25.6, 0.2]
        backbone_conf['y_bound'] = [-25.6, 25.6, 0.2]
        backbone_conf['d_bound'] = [2.0, 30, 0.25]
    if feature_size == 256 and division == 0.4:
        matching_conf["cart_resolution"] = 0.4
        matching_conf["cart_pixel_width"] = 256
        matching_conf["networks"]["keypoint_block"]["patch_size"] = 8
        backbone_conf['x_bound'] = [-51.2, 51.2, 0.4]
        backbone_conf['y_bound'] = [-51.2, 51.2, 0.4]
        backbone_conf['d_bound'] = [2.0, 58, 0.5]
    if feature_size == 128 and division == 0.4:
        matching_conf["cart_resolution"] = 0.4
        matching_conf["cart_pixel_width"] = 128
        matching_conf["networks"]["keypoint_block"]["patch_size"] = 8
        backbone_conf['x_bound'] = [-25.6, 25.6, 0.4]
        backbone_conf['y_bound'] = [-25.6, 25.6, 0.4]
        backbone_conf['d_bound'] = [2.0, 30, 0.25]
        # d_bound override for matched_depth
        if dataset_type == 'oxford':
            backbone_conf['d_bound'] = [2.0, 30, 0.25]
        elif dataset_type == 'NCLT':
            backbone_conf['d_bound'] = [0.4, 38.4, 0.15]
        else:
            backbone_conf['d_bound'] = [0.4, 38.4, 0.15]
        print(f"matched_depth d_bound set to {backbone_conf['d_bound']} for {dataset_type}")
    if feature_size == 128 and division == 0.8:
        matching_conf["cart_resolution"] = 0.8
        matching_conf["cart_pixel_width"] = 128
        matching_conf["networks"]["keypoint_block"]["patch_size"] = 8
        backbone_conf['x_bound'] = [-51.2, 51.2, 0.8]
        backbone_conf['y_bound'] = [-51.2, 51.2, 0.8]
        backbone_conf['d_bound'] = [2.0, 58, 0.5]
    if feature_size == 128 and division == 0.8 and matched_depth:
        matching_conf["cart_resolution"] = 0.8
        matching_conf["cart_pixel_width"] = 128
        matching_conf["networks"]["keypoint_block"]["patch_size"] = 8
        backbone_conf['x_bound'] = [-51.2, 51.2, 0.8]
        backbone_conf['y_bound'] = [-51.2, 51.2, 0.8]
        
        if dataset_type == 'oxford':
            backbone_conf['d_bound'] = [2.0, 30, 0.25]
        elif dataset_type == 'NCLT':
            backbone_conf['d_bound'] = [0.4, 38.4, 0.15]
        else:
            backbone_conf['d_bound'] = [0.4, 38.4, 0.15]
        print(f"matched_depth d_bound set to {backbone_conf['d_bound']} for {dataset_type}")

    if matched_depth and (feature_size != 128 or division != 0.8):
        print("matched_depth only support feature_size=128 and division=0.8!!!")
        exit()

    if dataset_type == 'oxford':
        backbone_conf['final_dim'] = (320, 640)
    if IS_MONO:
        if IS_MONO_CUT:
            matching_conf["cart_pixel_width"] = int(matching_conf["cart_pixel_width"] / 2)
    else:
        IS_MONO_CUT = False

    if other_downsample_factor != 16:
        backbone_conf['downsample_factor'] = other_downsample_factor
        if other_downsample_factor == 8:
            backbone_conf['img_neck_conf'] = dict(
                type='SECONDFPN',
                in_channels=[256, 512, 1024, 2048],
                upsample_strides=[0.5, 1, 2, 4],
                out_channels=[128, 128, 128, 128]
            )
            model_conf['use_pretrained_model'] = False
        if other_downsample_factor == 4:
            backbone_conf['img_neck_conf'] = dict(
                type='SECONDFPN',
                in_channels=[256, 512, 1024, 2048],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[128, 128, 128, 128]
            )
            model_conf['use_pretrained_model'] = False
    
    print(matching_conf, backbone_conf, model_conf)
    print("batch_size_train, NUM_WORKERS:", batch_size_train, NUM_WORKERS)

    backbone_conf["dataset_type"] = dataset_type

    test_net = BaseBEVODOM2(backbone_conf=backbone_conf,
                head_conf=head_conf,
                matching_conf=matching_conf,
                model_conf=model_conf,
                is_train_depth=True,
                IS_MONO_CUT=IS_MONO_CUT,
                ALL_DATA_INPUT=ALL_DATA_INPUT,
                FREEZE_BEV=FREEZE_BEV,
                NO_BN=NO_BN,
                NO_WEIGHT_SCORES=NO_WEIGHT_SCORES,
                dataset_type=dataset_type,
                featuresize_before_corr=featuresize_before_corr,
                wrap_level_num=wrap_level_num,
                max_dis=max_dis,
                freeze_bev=freeze_bev,
                corr_patch_size=corr_patch_size,
                use_leakyrelu_bn=use_leakyrelu_bn).cuda()

    monitor = MonitorBase(test_net, matching_conf)

    optimizer = Adam(test_net.parameters(), lr=lr_start, weight_decay=1e-4)

    scheduler = ExponentialLR(optimizer, gamma=0.95)
    start_epoch = None

    if True:
        if ckpt_path is not None:
            try:
                print('Loading from checkpoint: ' + ckpt_path)
                checkpoint = torch.load(ckpt_path, map_location=torch.device(matching_conf['gpuid']))

                new_state_dict = {}
                for key, value in checkpoint['model_state_dict'].items():
                    if key.startswith('module.'):
                        new_key = key[7:]  # remove 'module.' prefix
                    else:
                        new_key = key
                    new_state_dict[new_key] = value

                test_net.load_state_dict(new_state_dict, strict=False)
                
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except (ValueError, KeyError):
                    print("Warning: optimizer state_dict mismatch (model structure changed), using fresh optimizer")
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except (ValueError, KeyError):
                    print("Warning: scheduler state_dict mismatch, using fresh scheduler")
                start_epoch = checkpoint['epoch'] - 1
                if JUST_TEST:
                    monitor.counter = checkpoint['counter']
                else:
                    monitor.counter = checkpoint['counter']
                print('success')
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint: {ckpt_path}") from e

        if dataset_type == 'NCLT':
            with open(os.path.join(data_root_nclt, "image_meta.pkl"), 'rb') as handle:
                image_meta = pickle.load(handle)
                S = 5  # 5 cameras
        if dataset_type == 'oxford':
            with open(os.path.join(data_root_oxford, "image_meta.pkl"), 'rb') as handle:
                image_meta = pickle.load(handle)
                S = 3  # 3 cameras

        if IS_MONO:
            S = 1

        mats_dict = {}
        if not ALL_DATA_INPUT:
            B = 2
        if ALL_DATA_INPUT:
            B = 2 * batch_size_train
        __p = lambda x: geom.pack_seqdim(x, B)
        __u = lambda x: geom.unpack_seqdim(x, B)

        if IS_MONO and dataset_type == 'NCLT':
            intrins = torch.from_numpy(np.array(image_meta['K'])).float()[-1:, ...]
            pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
            cams_T_body = torch.from_numpy(np.array(image_meta['T'])).unsqueeze(0).float()[:1, -1:, ...]
        elif IS_MONO and dataset_type == 'oxford':
            intrins = torch.from_numpy(np.array(image_meta['K'][:3])).float()[-1:, ...]
            pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
            cams_T_body = torch.from_numpy(np.array(image_meta['T'][:3])).unsqueeze(0).float()[:1, -1:, ...]
        else:
            intrins = torch.from_numpy(np.array(image_meta['K'][:S])).float()
            pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
            cams_T_body = torch.from_numpy(np.array(image_meta['T'][:S])).unsqueeze(0).float()

        pix_T_cams = pix_T_cams.repeat(B,1,1,1).cuda()
        cams_T_body = cams_T_body.repeat(B,1,1,1).cuda()
        body_T_cams = __u(geom.safe_inverse(__p(cams_T_body)))
        pix_T_cams = pix_T_cams.view(B,1,S,4,4)
        cams_T_body = cams_T_body.view(B,1,S,4,4)
        body_T_cams = body_T_cams.view(B,1,S,4,4)
        ida_mats = torch.from_numpy(np.eye(4)).repeat(B*S,1,1).cuda().view(B,1,S,4,4)
        bda_mat = torch.from_numpy(np.eye(4)).repeat(B,1,1).cuda()

        mats_dict['sensor2ego_mats'] = body_T_cams.float()
        mats_dict['intrin_mats'] = pix_T_cams.float()
        mats_dict['ida_mats'] = ida_mats.float()
        mats_dict['bda_mat'] = bda_mat.float()

        if USE_MORE_SEQUENCES:
            if dataset_type == 'NCLT':
                base_dir = data_root_nclt
                
                train_dates = ['2013-04-05', '2012-01-08', '2012-02-04']
                
                root_dirs = [f"{base_dir}/{date}/lb3_u_s_384" for date in train_dates]
                csv_paths = [f"{base_dir}/{date}/ground_truth/groundtruth_{date}.csv" for date in train_dates]
                pickle_names = [os.path.join(pickle_root_nclt, date, pickle_name) for date in train_dates]

                nclt_dataset = NCLTDataset_sequences(
                    root_dirs=root_dirs, 
                    csv_paths=csv_paths, 
                    phase="train",
                    dataset_item_range=dataset_item_range,
                    pickle_names=pickle_names,
                    IS_MONO=IS_MONO,
                    NO_DEPTH=NO_DEPTH,
                )

                dataloader_params = {
                    'batch_size': batch_size_train,
                    'shuffle': True,
                    'drop_last': True,
                    'pin_memory': True
                }
                
                if (not JUST_TEST) and (not DEBUG):
                    dataloader_params['num_workers'] = NUM_WORKERS
                
                dataloader = DataLoader(nclt_dataset, **dataloader_params)

                test_date = '2012-03-17'
                test_root = [f"{base_dir}/{test_date}/lb3_u_s_384"]
                test_csv = [f"{base_dir}/{test_date}/ground_truth/groundtruth_{test_date}.csv"]
                pickle_names = [os.path.join(pickle_root_nclt, test_date, pickle_name)]

                test_date_2 = '2012-02-02'
                test_data_3 = '2012-02-19'
                test_data_4 = '2012-08-20'
                test_root_2 = [f"{base_dir}/{test_date_2}/lb3_u_s_384"]
                test_root_3 = [f"{base_dir}/{test_data_3}/lb3_u_s_384"]
                test_root_4 = [f"{base_dir}/{test_data_4}/lb3_u_s_384"]
                test_csv_2 = [f"{base_dir}/{test_date_2}/ground_truth/groundtruth_{test_date_2}.csv"]
                test_csv_3 = [f"{base_dir}/{test_data_3}/ground_truth/groundtruth_{test_data_3}.csv"]
                test_csv_4 = [f"{base_dir}/{test_data_4}/ground_truth/groundtruth_{test_data_4}.csv"]
                pickle_names_2 = [os.path.join(pickle_root_nclt, test_date_2, pickle_name)]
                pickle_names_3 = [os.path.join(pickle_root_nclt, test_data_3, pickle_name)]
                pickle_names_4 = [os.path.join(pickle_root_nclt, test_data_4, pickle_name)]

                nclt_dataset_test = NCLTDataset_sequences(
                    root_dirs=test_root,
                    csv_paths=test_csv,
                    phase="test",
                    dataset_item_range=dataset_item_range,
                    pickle_names=pickle_names,
                    IS_MONO=IS_MONO,
                    NO_DEPTH=NO_DEPTH,
                )
                nclt_dataset_test_2 = NCLTDataset_sequences(
                    root_dirs=test_root_2,
                    csv_paths=test_csv_2,
                    phase="test",
                    dataset_item_range=dataset_item_range,
                    pickle_names=pickle_names_2,
                    IS_MONO=IS_MONO,
                    NO_DEPTH=NO_DEPTH,
                )
                nclt_dataset_test_3 = NCLTDataset_sequences(
                    root_dirs=test_root_3,
                    csv_paths=test_csv_3,
                    phase="test",
                    dataset_item_range=dataset_item_range,
                    pickle_names=pickle_names_3,
                    IS_MONO=IS_MONO,
                    NO_DEPTH=NO_DEPTH,
                )
                nclt_dataset_test_4 = NCLTDataset_sequences(
                    root_dirs=test_root_4,
                    csv_paths=test_csv_4,
                    phase="test",
                    dataset_item_range=dataset_item_range,
                    pickle_names=pickle_names_4,
                    IS_MONO=IS_MONO,
                    NO_DEPTH=NO_DEPTH,
                )
                
                test_loader_params = {
                    'batch_size': batch_size_test,
                    'shuffle': False,
                    'drop_last': True,
                    'pin_memory': True
                }
                
                if (not DEBUG):
                    test_loader_params['num_workers'] = (NUM_WORKERS - 2) if NUM_WORKERS > 2 else 1
                    
                dataloader_test = DataLoader(nclt_dataset_test, **test_loader_params)
                dataloader_test_2 = DataLoader(nclt_dataset_test_2, **test_loader_params)
                dataloader_test_3 = DataLoader(nclt_dataset_test_3, **test_loader_params)
                dataloader_test_4 = DataLoader(nclt_dataset_test_4, **test_loader_params)

            if dataset_type == 'oxford':
                root_dir = data_root_oxford
                sequences_name = ["2019-01-11-13-24-51", "2019-01-14-14-15-12", "2019-01-15-14-24-38"]
                pickle_names = [os.path.join(pickle_root_oxford, date, pickle_name) for date in sequences_name]

                oxford_dataset = OxfordSequences(
                    dataset_root=root_dir, 
                    sequence_names=sequences_name, 
                    split="train", 
                    dataset_item_range=dataset_item_range, 
                    pickle_names=pickle_names, 
                    IS_MONO=IS_MONO, 
                    NO_DEPTH=NO_DEPTH)
                
                dataloader_params = {
                    'batch_size': batch_size_train,
                    'shuffle': True,
                    'drop_last': True,
                    'pin_memory': True
                }

                if (not JUST_TEST) and (not DEBUG):
                    dataloader_params['num_workers'] = NUM_WORKERS
                
                dataloader = DataLoader(oxford_dataset, **dataloader_params)

                root_dir_test = data_root_oxford
                sequence_name_test = ["2019-01-15-13-06-37"]
                pickle_names_test = [os.path.join(pickle_root_oxford, date, pickle_name) for date in sequence_name_test]
                sequence_name_test_2 = ["2019-01-11-12-26-55"]
                sequence_name_test_3 = ["2019-01-16-14-15-33"]
                sequence_name_test_4 = ["2019-01-17-12-48-25"]
                pickle_names_test_2 = [os.path.join(pickle_root_oxford, date, pickle_name) for date in sequence_name_test_2]
                pickle_names_test_3 = [os.path.join(pickle_root_oxford, date, pickle_name) for date in sequence_name_test_3]
                pickle_names_test_4 = [os.path.join(pickle_root_oxford, date, pickle_name) for date in sequence_name_test_4]

                oxford_dataset_test = OxfordSequences(
                    dataset_root=root_dir_test, 
                    sequence_names=sequence_name_test, 
                    split="test", 
                    dataset_item_range=dataset_item_range, 
                    pickle_names=pickle_names_test, 
                    IS_MONO=IS_MONO, 
                    NO_DEPTH=NO_DEPTH)
                oxford_dataset_test_2 = OxfordSequences(
                    dataset_root=root_dir_test,
                    sequence_names=sequence_name_test_2,
                    split="test",
                    dataset_item_range=dataset_item_range,
                    pickle_names=pickle_names_test_2,
                    IS_MONO=IS_MONO,
                    NO_DEPTH=NO_DEPTH)
                oxford_dataset_test_3 = OxfordSequences(
                    dataset_root=root_dir_test,
                    sequence_names=sequence_name_test_3,
                    split="test",
                    dataset_item_range=dataset_item_range,
                    pickle_names=pickle_names_test_3,
                    IS_MONO=IS_MONO,
                    NO_DEPTH=NO_DEPTH)
                oxford_dataset_test_4 = OxfordSequences(
                    dataset_root=root_dir_test,
                    sequence_names=sequence_name_test_4,
                    split="test",
                    dataset_item_range=dataset_item_range,
                    pickle_names=pickle_names_test_4,
                    IS_MONO=IS_MONO,
                    NO_DEPTH=NO_DEPTH)
                
                test_loader_params = {
                    'batch_size': batch_size_test,
                    'shuffle': False,
                    'drop_last': True,
                    'pin_memory': True
                }

                if (not DEBUG):
                    test_loader_params['num_workers'] = (NUM_WORKERS - 2) if NUM_WORKERS > 2 else 1

                dataloader_test = DataLoader(oxford_dataset_test, **test_loader_params)
                dataloader_test_2 = DataLoader(oxford_dataset_test_2, **test_loader_params)
                dataloader_test_3 = DataLoader(oxford_dataset_test_3, **test_loader_params)
                dataloader_test_4 = DataLoader(oxford_dataset_test_4, **test_loader_params)


        folder_path = os.path.join(model_save_root, "model_" + str(NAME))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        depth_loss_fn = DepthLoss(backbone_conf)

        current_time = datetime.datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

        with open(log_file_path, "a") as f:
            f.write("Current Time: {}\n".format(current_time_str))

        num_epoch = 0
        total_steps = 0
        if MORE_GPU:
            test_net = nn.DataParallel(test_net)
        while num_epoch < 1000:
            print(NAME)
            if freeze_bev and total_steps > freeze_steps:
                freeze_bev = False
                test_net.freeze_bev = False
                for param in test_net.backbone.parameters():
                    param.requires_grad = True

            mats_dict = {}
            if not ALL_DATA_INPUT:
                B = 2
            if ALL_DATA_INPUT:
                B = 2 * batch_size_train

            __p = lambda x: geom.pack_seqdim(x, B)
            __u = lambda x: geom.unpack_seqdim(x, B)
            
            if IS_MONO and dataset_type == 'NCLT':
                intrins = torch.from_numpy(np.array(image_meta['K'])).float()[-1:, ...]
                pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
                cams_T_body = torch.from_numpy(np.array(image_meta['T'])).unsqueeze(0).float()[:1, -1:, ...]
            elif IS_MONO and dataset_type == 'oxford':
                intrins = torch.from_numpy(np.array(image_meta['K'][:3])).float()[-1:, ...]
                pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
                cams_T_body = torch.from_numpy(np.array(image_meta['T'][:3])).unsqueeze(0).float()[:1, -1:, ...]
            else:
                intrins = torch.from_numpy(np.array(image_meta['K'][:S])).float()
                pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
                cams_T_body = torch.from_numpy(np.array(image_meta['T'][:S])).unsqueeze(0).float()

            pix_T_cams = pix_T_cams.repeat(B,1,1,1).cuda()
            cams_T_body = cams_T_body.repeat(B,1,1,1).cuda()
            body_T_cams = __u(geom.safe_inverse(__p(cams_T_body)))
            pix_T_cams = pix_T_cams.view(B,1,S,4,4)
            cams_T_body = cams_T_body.view(B,1,S,4,4)
            body_T_cams = body_T_cams.view(B,1,S,4,4)
            ida_mats = torch.from_numpy(np.eye(4)).repeat(B*S,1,1).cuda().view(B,1,S,4,4)
            bda_mat = torch.from_numpy(np.eye(4)).repeat(B,1,1).cuda()

            mats_dict['sensor2ego_mats'] = body_T_cams.float()
            mats_dict['intrin_mats'] = pix_T_cams.float()
            mats_dict['ida_mats'] = ida_mats.float()
            mats_dict['bda_mat'] = bda_mat.float()


            if start_epoch is not None:
                print(f"training from epoch {start_epoch + 2}")
                num_epoch = start_epoch + 1
                start_epoch = None
            print("num_epoch:", num_epoch + 1)
            total_svd_loss = 0
            total_R_loss = 0
            total_t_loss = 0
            total_flow_loss = 0
            total_R_loss_pv = 0
            total_t_loss_pv = 0
            total_depth_loss = 0
            total_num = 0
            
            total_svd_loss_val = 0
            total_R_loss_val = 0
            total_t_loss_val = 0
            total_num_val = 0

            temp_svd_loss = 0
            temp_R_loss = 0
            temp_t_loss = 0
            temp_flow_loss = 0
            temp_R_loss_pv = 0
            temp_t_loss_pv = 0
            temp_depth_loss = 0
            temp_num = 0

            model_save_path = os.path.join(model_save_root, "model_" + str(NAME), f"model_{num_epoch + 1}.pth")
            if USE_TRAIN_EVAL:
                test_net.train()
            if (num_epoch + 1) % epoch_save == 0:
                print("----------------------------------------")
                time_last = time.time()
            for batch_data in tqdm(dataloader):
                batch_all_2_5_images, poses, batch_all_2_5_depth, timestamp_train_all, idx1_all, idx2_all, poses_3dof = batch_data
                batch_all_2_5_images = batch_all_2_5_images.cuda()
                batch_all_2_5_depth = batch_all_2_5_depth.cuda()
                if total_num >= 500 and DEBUG:
                    break
                if JUST_TEST:
                    break

                pose0 = poses[:, 0]  # pose of frame t in world coordinates

                pose1 = poses[:, 1]  # pose of frame t+1 in world coordinates

                pose1_inv = torch.from_numpy(np.linalg.inv(pose1))
                t1_T_t = torch.matmul(pose1_inv, pose0).cuda()


                batch_all_2_5_images_new = batch_all_2_5_images.reshape(batch_all_2_5_images.shape[0] * batch_all_2_5_images.shape[1], batch_all_2_5_images.shape[2], batch_all_2_5_images.shape[3], batch_all_2_5_images.shape[4], batch_all_2_5_images.shape[5])

                # if saveimg_1.max() <= 1.0:
                # else:
                # if saveimg_2.max() <= 1.0:
                # else:


                batch_all_2_5_images_new = batch_all_2_5_images_new.unsqueeze(1)

                if SPLIT_R_t:
                    rotation_matrices, translation_matrices, dict_results, depth_pred_all, cut_x_save, flow_pred_temp, pvflow_rotation_matrices_out, pvflow_translation_matrices_out = test_net(batch_all_2_5_images_new, mats_dict, gt_Rt=t1_T_t)
                else:
                    rotation_matrices, translation_matrices, dict_results, depth_pred_all, cut_x_save, flow_pred_temp, pvflow_rotation_matrices_out, pvflow_translation_matrices_out = test_net(batch_all_2_5_images_new, mats_dict)
                
                depth_pred_all = depth_pred_all.reshape(batch_size_train, depth_pred_all.shape[0] // batch_size_train, depth_pred_all.shape[1], depth_pred_all.shape[2], depth_pred_all.shape[3])

                R_tgt_src_pred_all = [rotation_matrices[i:i+1] for i in range(rotation_matrices.shape[0])]
                t_tgt_src_pred_all = [translation_matrices[i:i+1] for i in range(translation_matrices.shape[0])]
                if flow_pred_temp is not None:
                    flow_pred_all = [flow_pred_temp[i:i+1] for i in range(flow_pred_temp.shape[0])]
                if pvflow_rotation_matrices_out is not None:
                    pvflow_rotation_matrices_all = [pvflow_rotation_matrices_out[i:i+1] for i in range(pvflow_rotation_matrices_out.shape[0])]
                    pvflow_translation_matrices_all = [pvflow_translation_matrices_out[i:i+1] for i in range(pvflow_translation_matrices_out.shape[0])]
                dict_matching_all = [{'R': rotation_matrices[i:i+1], 't': translation_matrices[i:i+1]} for i in range(rotation_matrices.shape[0])]
                cut_x_save = Image.fromarray(cut_x_save.cpu().detach().numpy().astype('uint8'), mode='L')


                for iii in range(batch_size_train):
                    R_tgt_src_pred, t_tgt_src_pred, dict_matching, depth_pred, timestamp_train = R_tgt_src_pred_all[iii], t_tgt_src_pred_all[iii], dict_matching_all[iii], depth_pred_all[iii], timestamp_train_all[iii]
                    # Flow + PV flow loss
                    flow_pred = flow_pred_all[iii] if flow_pred_temp is not None else None
                    flow_gt = test_net.compute_flow_gt(t1_T_t[iii:iii+1]) if flow_pred is not None else None
                    svd_loss, dict_loss = test_net.RTloss(R_tgt_src_pred, t_tgt_src_pred, t1_T_t[iii:iii+1],
                                                          flow_pred=flow_pred, flow_gt=flow_gt,
                                                          testmode=(flow_pred is None))

                    if pvflow_rotation_matrices_out is not None:
                        pvflow_r = pvflow_rotation_matrices_all[iii]
                        pvflow_t_pred = pvflow_translation_matrices_all[iii]
                        svd_loss_pv, dict_loss_pv = test_net.RTloss(pvflow_r, pvflow_t_pred, t1_T_t[iii:iii+1])
                    else:
                        svd_loss_pv = None
                        dict_loss_pv = None

                    if NO_DEPTH == False:
                        depth_loss, _ = depth_loss_fn(batch_all_2_5_depth[iii], depth_pred)
                        total_depth_loss += depth_loss.detach().cpu().item()*depthloss_rate
                        temp_depth_loss += depth_loss.detach().cpu().item()*depthloss_rate
                        temp_svd_loss += (svd_loss + depth_loss*depthloss_rate)
                        if svd_loss_pv is not None:
                            temp_svd_loss += svd_loss_pv*pvloss_rate
                        total_svd_loss += (svd_loss.detach().cpu().item() + depth_loss.detach().cpu().item()*depthloss_rate)
                        if svd_loss_pv is not None:
                            total_svd_loss += svd_loss_pv.detach().cpu().item()
                    else:
                        temp_svd_loss += svd_loss
                        if svd_loss_pv is not None:
                            temp_svd_loss += svd_loss_pv*pvloss_rate
                        total_svd_loss += svd_loss.detach().cpu().item()
                        if svd_loss_pv is not None:
                            total_svd_loss += svd_loss_pv.detach().cpu().item()

                    total_R_loss += dict_loss["R_loss"].detach().cpu().item()
                    total_t_loss += dict_loss["t_loss"].detach().cpu().item()
                    if "flow_loss" in dict_loss:
                        total_flow_loss += dict_loss["flow_loss"].detach().cpu().item()
                    if dict_loss_pv is not None:
                        total_R_loss_pv += dict_loss_pv["R_loss"].detach().cpu().item()
                        total_t_loss_pv += dict_loss_pv["t_loss"].detach().cpu().item()
                    total_num += 1

                    temp_R_loss += dict_loss["R_loss"].detach().cpu().item()
                    temp_t_loss += dict_loss["t_loss"].detach().cpu().item()
                    if "flow_loss" in dict_loss:
                        temp_flow_loss += dict_loss["flow_loss"].detach().cpu().item()
                    if dict_loss_pv is not None:
                        temp_R_loss_pv += dict_loss_pv["R_loss"].detach().cpu().item()
                        temp_t_loss_pv += dict_loss_pv["t_loss"].detach().cpu().item()
                    temp_num += 1
                
                if temp_num == 0:
                    temp_svd_loss = 0
                    temp_R_loss = 0
                    temp_t_loss = 0
                    temp_depth_loss = 0
                    temp_flow_loss = 0
                    time_last = time.time()
                    continue

                back_loss_record = temp_svd_loss / temp_num
                R_loss_record = temp_R_loss / temp_num
                t_loss_record = temp_t_loss / temp_num
                depth_loss_record = temp_depth_loss / temp_num
                flow_loss_record = temp_flow_loss / temp_num
                R_loss_pv_record = temp_R_loss_pv / temp_num
                t_loss_pv_record = temp_t_loss_pv / temp_num

                optimizer.zero_grad()
                back_loss_record.backward()

                if USE_GRAD_CLIP:
                    clip_grad_norm_(test_net.parameters(), max_norm)

                optimizer.step()

                monitor_count = monitor.step(back_loss_record, R_loss_record, t_loss_record, depth_loss_record, flow_loss_record, R_loss_pv_record, t_loss_pv_record)
                total_steps += 1

                temp_svd_loss = 0
                temp_R_loss = 0
                temp_t_loss = 0
                temp_depth_loss = 0
                temp_flow_loss = 0
                temp_R_loss_pv = 0
                temp_t_loss_pv = 0
                temp_num = 0

                time_last = time.time()

            if not JUST_TEST:   
                scheduler.step()
                

            mats_dict = {}
            B = 2 * batch_size_test
            __p = lambda x: geom.pack_seqdim(x, B)
            __u = lambda x: geom.unpack_seqdim(x, B)
            
            if IS_MONO and dataset_type == 'NCLT':
                intrins = torch.from_numpy(np.array(image_meta['K'])).float()[-1:, ...]
                pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
                cams_T_body = torch.from_numpy(np.array(image_meta['T'])).unsqueeze(0).float()[:1, -1:, ...]
            elif IS_MONO and dataset_type == 'oxford':
                intrins = torch.from_numpy(np.array(image_meta['K'][:3])).float()[-1:, ...]
                pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
                cams_T_body = torch.from_numpy(np.array(image_meta['T'][:3])).unsqueeze(0).float()[:1, -1:, ...]
            else:
                intrins = torch.from_numpy(np.array(image_meta['K'][:S])).float()
                pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
                cams_T_body = torch.from_numpy(np.array(image_meta['T'][:S])).unsqueeze(0).float()

            pix_T_cams = pix_T_cams.repeat(B,1,1,1).cuda()
            cams_T_body = cams_T_body.repeat(B,1,1,1).cuda()
            body_T_cams = __u(geom.safe_inverse(__p(cams_T_body)))
            pix_T_cams = pix_T_cams.view(B,1,S,4,4)
            cams_T_body = cams_T_body.view(B,1,S,4,4)
            body_T_cams = body_T_cams.view(B,1,S,4,4)
            ida_mats = torch.from_numpy(np.eye(4)).repeat(B*S,1,1).cuda().view(B,1,S,4,4)
            bda_mat = torch.from_numpy(np.eye(4)).repeat(B,1,1).cuda()

            mats_dict['sensor2ego_mats'] = body_T_cams.float()
            mats_dict['intrin_mats'] = pix_T_cams.float()
            mats_dict['ida_mats'] = ida_mats.float()
            mats_dict['bda_mat'] = bda_mat.float()

            if not JUST_TEST:
                print(f"Model saved at epoch {num_epoch + 1}")
                torch.save({
                    'model_state_dict': test_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'counter': monitor.counter,
                    'epoch': num_epoch + 1,
                    }, model_save_path)

            if USE_TRAIN_EVAL:
                test_net.eval()


            # Test on 4 validation sequences
            all_test = [dataloader_test, dataloader_test_2, dataloader_test_3, dataloader_test_4]
            test_num_seq = 0
            t_err_avg_all = []
            R_err_avg_all = []
            kitti_t_err_all = []
            kitti_R_err_all = []

            for dataloader_test_now in all_test:
                test_num_seq += 1
                all_t1_T_t = []
                all_t1_T_t_3dof = []
                all_homogeneous_transform = []
                timestamps_val = []
                cut_x_save_vis = None
                test_num = 0

                total_svd_loss_val = 0
                total_R_loss_val = 0
                total_t_loss_val = 0
                total_num_val = 0

                for batch_data in tqdm(dataloader_test_now):
                    batch_all_2_5_images, poses, batch_all_2_5_depth, timestamp_val, poses_3dof = batch_data
                    batch_all_2_5_images = batch_all_2_5_images.cuda()
                    # Append all timestamps in the batch
                    timestamps_val.extend(timestamp_val)
                    
                    pose0 = poses[:, 0]  # pose of frame t in world coordinates

                    pose1 = poses[:, 1]  # pose of frame t+1 in world coordinates

                    pose1_inv = torch.from_numpy(np.linalg.inv(pose1))
                    t1_T_t = torch.matmul(pose1_inv, pose0).cuda()

                    pose0_3dof = poses_3dof[:, 0]  # pose of frame t in world coordinates

                    pose1_3dof = poses_3dof[:, 1]  # pose of frame t+1 in world coordinates

                    pose1_inv_3dof = torch.from_numpy(np.linalg.inv(pose1_3dof))
                    t1_T_t_3dof = torch.matmul(pose1_inv_3dof, pose0_3dof).cuda()
                    
                    batch_all_2_5_images_new = batch_all_2_5_images.reshape(batch_all_2_5_images.shape[0] * batch_all_2_5_images.shape[1], 
                                                                        batch_all_2_5_images.shape[2], 
                                                                        batch_all_2_5_images.shape[3], 
                                                                        batch_all_2_5_images.shape[4], 
                                                                        batch_all_2_5_images.shape[5])

                    batch_all_2_5_images_new = batch_all_2_5_images_new.unsqueeze(1)
                    
                    rotation_matrices, translation_matrices, dict_results, depth_pred_all, cut_x_save, flow_pred, pvflow_rotation_matrices, pvflow_translation_matrices = test_net(batch_all_2_5_images_new, mats_dict, count_num=test_num)
                    test_num += batch_size_test
                    
                    R_tgt_src_pred_all = [rotation_matrices[i:i+1] for i in range(rotation_matrices.shape[0])]
                    t_tgt_src_pred_all = [translation_matrices[i:i+1] for i in range(translation_matrices.shape[0])]
                    dict_results_all = [{'R': rotation_matrices[i:i+1], 't': translation_matrices[i:i+1]} for i in range(rotation_matrices.shape[0])]
                    cut_x_save = Image.fromarray(cut_x_save.cpu().detach().numpy().astype('uint8'), mode='L')

                    for iii in range(batch_size_test):
                        # if (np.sqrt(t1_T_t[iii][0,3].cpu().numpy()**2 + t1_T_t[iii][1,3].cpu().numpy()**2)) < 0.2:
                        #     continue
                            
                        R_tgt_src_pred, t_tgt_src_pred = R_tgt_src_pred_all[iii], t_tgt_src_pred_all[iii]
                        
                        svd_loss_val, dict_loss_val = test_net.RTloss(R_tgt_src_pred, t_tgt_src_pred, t1_T_t[iii:iii+1], testmode=True)
                        
                        all_t1_T_t.append(t1_T_t[iii:iii+1].cpu().numpy().reshape((4, 4)))
                        all_t1_T_t_3dof.append(t1_T_t_3dof[iii:iii+1].cpu().numpy().reshape((4, 4)))
                        
                        homogeneous_transform = np.eye(4)
                        homogeneous_transform[:3, :3] = R_tgt_src_pred.cpu().detach().numpy()
                        homogeneous_transform[:3, 3] = t_tgt_src_pred.cpu().detach().numpy().reshape((3,))
                        all_homogeneous_transform.append(homogeneous_transform)
                        
                        total_svd_loss_val += svd_loss_val.detach().cpu().item()
                        total_R_loss_val += dict_loss_val["R_loss"].detach().cpu().item()
                        total_t_loss_val += dict_loss_val["t_loss"].detach().cpu().item()
                        total_num_val += 1
                    
                    if not DEBUG and (total_num_val >= 1000 or total_num_val >= len(dataloader_test_now) // 5) and cut_x_save_vis is None:
                        cut_x_save_vis = cut_x_save
                    if DEBUG and total_num_val >= len(dataloader_test_now) // 200 and cut_x_save_vis is None:
                        cut_x_save_vis = cut_x_save
                    
                    if (total_num_val >= 2000) and (not JUST_TEST) and (not USE_MORE_SEQUENCES):
                        cut_x_save_vis = cut_x_save
                        break
                    if (total_num_val >= (len(dataloader_test_now) // 200 + 10)) and DEBUG:
                        break

                with open(log_file_path, "a") as f:
                    original_stdout = sys.stdout
                    sys.stdout = f

                    if not JUST_TEST and total_num != 0 and test_num_seq == 1:
                        avg_svd_loss = total_svd_loss / total_num
                        avg_R_loss = total_R_loss / total_num
                        avg_t_loss = total_t_loss / total_num
                        avg_flow_loss = total_flow_loss / total_num
                        avg_R_pv_loss = total_R_loss_pv / total_num
                        avg_t_pv_loss = total_t_loss_pv / total_num
                        print("num_epoch: {:d}\tavg_svd_loss: {:.6f}\tavg_R_loss: {:.6f}\tavg_t_loss: {:.6f}\tavg_flow_loss: {:.6f}\tavg_R_pv_loss: {:.6f}\tavg_t_pv_loss: {:.6f}".format((num_epoch + 1), avg_svd_loss, avg_R_loss, avg_t_loss, avg_flow_loss, avg_R_pv_loss, avg_t_pv_loss))

                    if total_num_val == 0:
                        print(f"[WARN] Empty validation loader for sequence {test_num_seq}, skip.")
                        continue
                    avg_svd_loss_val = total_svd_loss_val / total_num_val
                    avg_R_loss_val = total_R_loss_val / total_num_val
                    avg_t_loss_val = total_t_loss_val / total_num_val

                    t_err_avg, R_err_avg, kitti_t_err, kitti_r_err = monitor.step_val_iros(all_t1_T_t_3dof, all_homogeneous_transform, timestamps_val, \
                        file_path_gt=os.path.join(evo_output_root, 'tum_trajectory_gt_'+str(test_num_seq)+'_'+str(NAME)+'_'+str(num_epoch)+'.txt'), \
                        file_path_pred=os.path.join(evo_output_root, 'tum_trajectory_pred_'+str(test_num_seq)+'_'+str(NAME)+'_'+str(num_epoch)+'.txt'), \
                        cut_x_save_vis=cut_x_save_vis, test_num_seq=test_num_seq)
                    
                    t_err_avg_all.append(t_err_avg)
                    R_err_avg_all.append(R_err_avg)
                    kitti_t_err_all.append(kitti_t_err)
                    kitti_R_err_all.append(kitti_r_err)

                    if test_num_seq == len(all_test):
                        t_err_avg_record = np.mean(t_err_avg_all)
                        R_err_avg_record = np.mean(R_err_avg_all)
                        kitti_t_err_record = np.mean(kitti_t_err_all)
                        kitti_R_err_record = np.mean(kitti_R_err_all)
                        monitor.step_val_iros_record(t_err_avg_record, R_err_avg_record, kitti_t_err_record, kitti_R_err_record)

                    print("num_epoch: {:d}\tavg_svd_loss_val: {:.6f}\tavg_R_loss_val: {:.6f}\tavg_t_loss_val: {:.6f}\n----------------------------------------" \
                    .format((num_epoch + 1), avg_svd_loss_val, avg_R_loss_val, avg_t_loss_val))
                    print("t_err_avg: {:.6f}\tR_err_avg: {:.6f}\tkitti_t_err: {:.6f}\tkitti_R_err: {:.6f}".format(t_err_avg, R_err_avg, kitti_t_err, kitti_r_err))

                    sys.stdout = original_stdout

            torch.cuda.empty_cache()
            num_epoch += 1
            if JUST_TEST:
                print("finish test")
                if len(t_err_avg_all) > 0:
                    print("================ FINAL TEST RESULTS ================")
                    for idx in range(len(t_err_avg_all)):
                        print(f"Sequence {idx+1} - t_err_avg: {t_err_avg_all[idx]:.6f}, R_err_avg: {R_err_avg_all[idx]:.6f}, kitti_t_err: {kitti_t_err_all[idx]:.6f}, kitti_R_err: {kitti_R_err_all[idx]:.6f}")
                    if len(t_err_avg_all) > 1:
                        print("---------------- AVERAGE RESULTS ----------------")
                        print(f"Mean t_err_avg: {np.mean(t_err_avg_all):.6f}, Mean R_err_avg: {np.mean(R_err_avg_all):.6f}, Mean kitti_t_err: {np.mean(kitti_t_err_all):.6f}, Mean kitti_R_err: {np.mean(kitti_R_err_all):.6f}")
                    print("====================================================")
                break