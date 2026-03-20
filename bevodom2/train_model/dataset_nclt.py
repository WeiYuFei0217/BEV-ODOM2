import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import struct
import matplotlib.pyplot as plt
import transforms3d.euler as euler

import pickle
from tqdm import tqdm

from PIL import Image
import numpy as np
import random

seed = 1024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class NCLTDataset(Dataset):
    def __init__(self, root_dir, csv_path, phase, image_transform=None, dataset_item_range="0.0-2.0", pickle_name="pickle.txt", IS_MONO=False, NO_DEPTH=True):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_path, header=None)
        self.df_list = self.df[0].tolist()
        self.phase = phase
        self.image_transform = image_transform
        self.dataset_item_range = dataset_item_range
        self.IS_MONO = IS_MONO
        self.NO_DEPTH = NO_DEPTH
        if self.phase == "prepare_gt" or self.phase == "prepare_pickle":
            self.nearby_points = {}
            self.nearby_points_R = {}
        else:
            save_pickle_dir = os.path.join(pickle_name)
            self.pickle = pickle.load(open(save_pickle_dir, 'rb'))
            self.nearby_points = self.pickle[0]
            self.nearby_points_R = self.pickle[1]

        self.image_names = sorted(os.listdir(os.path.join(root_dir, "Cam1")))

        if self.phase == "prepare_gt" or self.phase == "prepare_pickle":
            self.image_names_part = self.image_names
        if self.phase == "train":
            self.image_names_part = self.image_names[:-1]
        if self.phase == "test":
            if self.dataset_item_range == "0.0-2.0":
                self.image_names_part = self.image_names[::5]

    def __len__(self):
        if self.phase == "prepare_gt" or self.phase == "prepare_pickle":
            return len(self.image_names_part)
        else:
            if self.phase == "train":
                return len(self.image_names_part)
            if self.phase == "test":
                return len(self.image_names_part) - 1

    def __getitem__(self, idx):

        if self.IS_MONO:
            folder_paths = [os.path.join(self.root_dir, "Cam5")]
            folder_pc_path = os.path.join(self.root_dir, "../velodyne_sync")
            folder_paths_depth = [os.path.normpath(os.path.join(self.root_dir, "../depth_img/Cam5"))]

        else:
            folder_paths = [
                os.path.join(self.root_dir, "Cam1"),
                os.path.join(self.root_dir, "Cam2"),
                os.path.join(self.root_dir, "Cam3"),
                os.path.join(self.root_dir, "Cam4"),
                os.path.join(self.root_dir, "Cam5")
            ]
            folder_pc_path = os.path.join(self.root_dir, "../velodyne_sync")
            folder_paths_depth = [
                os.path.normpath(os.path.join(self.root_dir, "../depth_img/Cam1")),
                os.path.normpath(os.path.join(self.root_dir, "../depth_img/Cam2")),
                os.path.normpath(os.path.join(self.root_dir, "../depth_img/Cam3")),
                os.path.normpath(os.path.join(self.root_dir, "../depth_img/Cam4")),
                os.path.normpath(os.path.join(self.root_dir, "../depth_img/Cam5"))
            ]

        all_2_5_images = []
        all_2_5_depth = []

        if self.phase == "train":
            idx1 = idx

            correct_idx2_list = []
            correct_idx2_list_rot = []

            correct_idx2_list = self.nearby_points[idx1]
            correct_idx2_list_rot = self.nearby_points_R[idx1]

            if len(correct_idx2_list_rot) != 0 and len(correct_idx2_list) != 0:
                if random.randint(1, 10) <= 7:
                    idx2 = random.choice(correct_idx2_list_rot)
                else:
                    idx2 = random.choice(correct_idx2_list)
            elif len(correct_idx2_list_rot) != 0:
                idx2 = random.choice(correct_idx2_list_rot)
            elif len(correct_idx2_list) != 0:
                idx2 = random.choice(correct_idx2_list)
            else:
                while len(correct_idx2_list) == 0 and len(correct_idx2_list_rot) == 0:
                    idx1 = random.randint(0, len(self.image_names_part)-1)
                    correct_idx2_list = self.nearby_points[idx1]
                    correct_idx2_list_rot = self.nearby_points_R[idx1]
                    if len(correct_idx2_list_rot) != 0 and len(correct_idx2_list) != 0:
                        if random.randint(1, 10) <= 7:
                            idx2 = random.choice(correct_idx2_list_rot)
                        else:
                            idx2 = random.choice(correct_idx2_list)
                    elif len(correct_idx2_list_rot) != 0:
                        idx2 = random.choice(correct_idx2_list_rot)
                    elif len(correct_idx2_list) != 0:
                        idx2 = random.choice(correct_idx2_list)

        if self.phase == "test":
            idx1 = idx
            idx2 = idx + 1

        if self.phase == "train":
            poses = np.zeros((2, 4, 4), dtype=np.float64)
            poses_3dof = np.zeros((2, 4, 4), dtype=np.float64)
            timestamp = None
            for ndx, idx_temp in enumerate([idx1, idx2]):
                all_5_images = []
                image_name = self.image_names[idx_temp].split('.')[0]

                target_value = float(image_name)
                nearest_row = self.find_nearest_value(target_value)
                temp = nearest_row.iloc[:].tolist()
                if timestamp is None:
                    timestamp = float(temp[0])
                poses[ndx] = self.RPY2Rot(float(temp[1]), float(temp[2]), float(temp[3]), float(temp[4]), float(temp[5]), float(temp[6]))
                poses_3dof[ndx] = self.RPY2Rot_3dof(float(temp[1]), float(temp[2]), float(temp[3]), float(temp[4]), float(temp[5]), float(temp[6]))

                images = self.load_images_from_folder(folder_paths, image_name)
                tensors = self.images_to_tensor(images)
                all_5_images = torch.stack(tensors)
                all_2_5_images.append(all_5_images)

                depth_maps = []
                if self.NO_DEPTH:
                    if self.IS_MONO:
                        depth_map = torch.zeros_like(tensors[0][0])
                        depth_maps.append(depth_map)
                        all_2_5_depth.append(torch.stack(depth_maps))
                    else:
                        for tensor in tensors:
                            depth_map = torch.zeros_like(tensor[0])
                            depth_maps.append(depth_map)
                        all_2_5_depth.append(torch.stack(depth_maps))
                else:
                    if self.IS_MONO:
                        for cam_num in range(1):
                            file_path = os.path.normpath(os.path.join(folder_paths_depth[cam_num], f"{image_name}.npy"))
                            depth_map = np.load(file_path)
                            depth_map = torch.tensor(depth_map, dtype=torch.float)
                            depth_maps.append(depth_map)
                        all_2_5_depth.append(torch.stack(depth_maps))
                    else:
                        for cam_num in range(5):
                            file_path = os.path.normpath(os.path.join(folder_paths_depth[cam_num], f"{image_name}.npy"))
                            depth_map = np.load(file_path)
                            depth_map = torch.tensor(depth_map, dtype=torch.float)
                            depth_maps.append(depth_map)
                        all_2_5_depth.append(torch.stack(depth_maps))

            all_2_5_images = torch.stack(all_2_5_images)
            all_2_5_depth = torch.stack(all_2_5_depth)
            return all_2_5_images, poses, all_2_5_depth, timestamp, idx1, idx2, poses_3dof

        if self.phase == "test":
            poses = np.zeros((2, 4, 4), dtype=np.float64)
            poses_3dof = np.zeros((2, 4, 4), dtype=np.float64)
            timestamp = None
            for ndx, idx_temp in enumerate([idx1, idx2]):
                all_5_images = []
                image_name = self.image_names_part[idx_temp].split('.')[0]

                target_value = float(image_name)
                nearest_row = self.find_nearest_value(target_value)
                temp = nearest_row.iloc[:].tolist()
                if timestamp is None:
                    timestamp = float(temp[0])
                poses[ndx] = self.RPY2Rot(float(temp[1]), float(temp[2]), float(temp[3]), float(temp[4]), float(temp[5]), float(temp[6]))  # , poses_yaw_x_y[ndx]
                poses_3dof[ndx] = self.RPY2Rot_3dof(float(temp[1]), float(temp[2]), float(temp[3]), float(temp[4]), float(temp[5]), float(temp[6]))  # , poses_yaw_x_y[ndx]

                images = self.load_images_from_folder(folder_paths, image_name)
                tensors = self.images_to_tensor(images)
                all_5_images = torch.stack(tensors)
                all_2_5_images.append(all_5_images)

                depth_maps = []
                if self.NO_DEPTH:
                    if self.IS_MONO:
                        depth_map = torch.zeros_like(tensors[0][0])
                        depth_maps.append(depth_map)
                        all_2_5_depth.append(torch.stack(depth_maps))
                    else:
                        for tensor in tensors:
                            depth_map = torch.zeros_like(tensor[0])
                            depth_maps.append(depth_map)
                        all_2_5_depth.append(torch.stack(depth_maps))
                else:
                    if self.IS_MONO:
                        for cam_num in range(1):
                            file_path = os.path.normpath(os.path.join(folder_paths_depth[cam_num], f"{image_name}.npy"))
                            depth_map = np.load(file_path)
                            depth_map = torch.tensor(depth_map, dtype=torch.float)
                            depth_maps.append(depth_map)
                        all_2_5_depth.append(torch.stack(depth_maps))
                    else:
                        for cam_num in range(5):
                            file_path = os.path.normpath(os.path.join(folder_paths_depth[cam_num], f"{image_name}.npy"))
                            depth_map = np.load(file_path)
                            depth_map = torch.tensor(depth_map, dtype=torch.float)
                            depth_maps.append(depth_map)
                        all_2_5_depth.append(torch.stack(depth_maps))

            all_2_5_images = torch.stack(all_2_5_images)
            all_2_5_depth = torch.stack(all_2_5_depth)
            return all_2_5_images, poses, all_2_5_depth, timestamp, poses_3dof

    def load_images_from_folder(self, folder_paths, image_name):
        images = []
        for folder_path in folder_paths:
            image_path = os.path.join(folder_path, f"{image_name}.jpg")
            if os.path.exists(image_path):
                input_image = cv2.imread(image_path)
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                input_image = cv2.rotate(input_image, cv2.ROTATE_90_CLOCKWISE)
                images.append(input_image)
            else:
                print(f"Warning: {image_path} not found.")
        return images

    def images_to_tensor(self, images):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tensors = [transform(image) for image in images]
        return tensors

    def find_nearest_value(self, target_value):
        return self.df.iloc[(self.df[0] - target_value).abs().idxmin()]
        # return self.df.iloc[np.searchsorted(self.df_list, target_value)]

    # nclt pointcloud utils
    def convert(self, x_s, y_s, z_s):
        scaling = 0.005 # 5 mm
        offset = -100.0

        x = x_s * scaling + offset
        y = y_s * scaling + offset
        z = z_s * scaling + offset

        return x, y, z

    # load lidar file in nclt dataset
    def load_lidar_file_nclt(self, file_path):
        f_bin = open(file_path,'rb')
        hits = []
        while True:
            x_str = f_bin.read(2)
            if x_str == b"": # eof
                break

            x = struct.unpack('<H', x_str)[0]
            y = struct.unpack('<H', f_bin.read(2))[0]
            z = struct.unpack('<H', f_bin.read(2))[0]
            i = struct.unpack('B', f_bin.read(1))[0]
            l = struct.unpack('B', f_bin.read(1))[0]
            x, y, z = self.convert(x, y, z)
            hits += [[x, y, z]]

        f_bin.close()
        hits = np.asarray(hits)

        return hits

    def ssc_to_homo(self,ssc):

        # Convert 6-DOF ssc coordinate transformation to 4x4 homogeneous matrix
        # transformation

        sr = np.sin(np.pi/180.0 * ssc[3])
        cr = np.cos(np.pi/180.0 * ssc[3])

        sp = np.sin(np.pi/180.0 * ssc[4])
        cp = np.cos(np.pi/180.0 * ssc[4])

        sh = np.sin(np.pi/180.0 * ssc[5])
        ch = np.cos(np.pi/180.0 * ssc[5])

        H = np.zeros((4, 4))

        H[0, 0] = ch*cp
        H[0, 1] = -sh*cr + ch*sp*sr
        H[0, 2] = sh*sr + ch*sp*cr
        H[1, 0] = sh*cp
        H[1, 1] = ch*cr + sh*sp*sr
        H[1, 2] = -ch*sr + sh*sp*cr
        H[2, 0] = -sp
        H[2, 1] = cp*sr
        H[2, 2] = cp*cr

        H[0, 3] = ssc[0]
        H[1, 3] = ssc[1]
        H[2, 3] = ssc[2]

        H[3, 3] = 1

        return H

    def project_vel_to_cam(self, hits, cam_num):

        # Load camera parameters
        cam_params_dir = os.environ.get("NCLT_CAM_PARAMS_DIR", "/data/wyf/NCLT/cam_params")
        K = np.loadtxt(os.path.join(cam_params_dir, 'K_cam%d.csv' % (cam_num)), delimiter=',')
        factor_x = 224. / 600.
        factor_y = 384. / 900.
        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]

        cx = 1616. - cx
        cx -= 616.  # cx
        cy -= 150.  # cy
        cx = cx * factor_x
        cy = cy * factor_y

        K[0][0] = fy * factor_y
        K[0][2] = cy
        K[1][1] = fx * factor_x
        K[1][2] = cx
        # image_meta :K
        x_lb3_c = np.loadtxt(os.path.join(cam_params_dir, 'x_lb3_c%d.csv' % (cam_num)), delimiter=',')

        # Other coordinate transforms we need
        x_body_lb3 = [0.035, 0.002, -1.23, -179.93, -0.23, 0.50]

        x_camNormal_cam = [0.0, 0.0, 0.0, 0.0, 0.0, -90]
        T_camNormal_cam = self.ssc_to_homo(x_camNormal_cam)

        # Now do the projection
        T_lb3_c = self.ssc_to_homo(x_lb3_c)
        T_body_lb3 = self.ssc_to_homo(x_body_lb3)

        T_lb3_body = np.linalg.inv(T_body_lb3)
        T_c_lb3 = np.linalg.inv(T_lb3_c)

        T_c_body = np.matmul(T_c_lb3, T_lb3_body)
        T_body_c = np.linalg.inv(T_c_body)

        T_camNormal_body = np.matmul(T_camNormal_cam, T_c_body)
        # image_meta :T

        # exit()
        hits_c = np.matmul(T_camNormal_body, hits)
        hits_im = np.matmul(K, hits_c[0:3, :])

        return hits_im

    def RPY2Rot(self, x, y, z, roll, pitch, yaw):
        R = [[np.cos(pitch)*np.cos(yaw), -np.cos(pitch)*np.sin(yaw), np.sin(pitch), x],
             [np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(roll)*np.sin(yaw),
              -np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(roll)*np.cos(yaw),
              -np.sin(roll)*np.cos(pitch), y],
             [-np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll),
              np.cos(roll)*np.sin(pitch)*np.sin(yaw) + np.sin(roll)*np.cos(yaw),
              np.cos(roll)*np.cos(pitch), z],
             [0, 0, 0, 1]]
        return np.array(R)
    
    def RPY2Rot_3dof(self, x, y, z, roll, pitch, yaw):
        T = np.identity(4, dtype=np.float32)
        T[0:2, 0:2] = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        T[0, 3] = x
        T[1, 3] = y
        return T

    def RPY2Rot_savegt(self, x, y, z, roll, pitch, yaw):
        R = [[np.cos(pitch)*np.cos(yaw), -np.cos(pitch)*np.sin(yaw), np.sin(pitch), x],
             [np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(roll)*np.sin(yaw),
              -np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(roll)*np.cos(yaw),
              -np.sin(roll)*np.cos(pitch), y],
             [-np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll),
              np.cos(roll)*np.sin(pitch)*np.sin(yaw) + np.sin(roll)*np.cos(yaw),
              np.cos(roll)*np.cos(pitch), z],
             [0, 0, 0, 1]]
        return R

    def euler_to_quaternion(self, x, y, z, roll, pitch, yaw):
        quaternion = euler.euler2quat(roll, pitch, yaw)

        return x, y, z, quaternion[0], quaternion[1], quaternion[2], quaternion[3]

    def save_tum_trajectory(self, file_path_notime, file_path, jiange=1):
        translation = [0,0,0]
        quaternion = [0,0,0,0]
        with open(file_path, 'w') as f, open(file_path_notime, 'w') as f_notime:
            for idx in range(0, len(self.image_names_part), jiange):
                image_name = self.image_names_part[idx].split('.')[0]
                target_value = float(image_name)
                nearest_row = self.find_nearest_value(target_value)
                temp = nearest_row.iloc[:].tolist()
                timestamp = float(temp[0])

                translation[0], translation[1], translation[2], quaternion[3], quaternion[0], quaternion[1], quaternion[2] = self.euler_to_quaternion(float(temp[1]), float(temp[2]), float(temp[3]), float(temp[4]), float(temp[5]), float(temp[6]))
                pose_str = f"{translation[0]} {translation[1]} {translation[2]} {quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[3]}"

                f.write(f"{timestamp} {pose_str}\n")
                f_notime.write(f"{pose_str}\n")

class NCLTDataset_sequences(Dataset):
    def __init__(self, root_dirs, csv_paths, phase, image_transform=None, dataset_item_range="0.0-2.0", pickle_names=["pickle.txt"], IS_MONO=False, NO_DEPTH=True):
    
        sequences = []
        for root_dir, csv_path, pickle_name in zip(root_dirs, csv_paths, pickle_names):
            print(csv_path)
            ds = NCLTDataset(root_dir, csv_path, phase, image_transform, dataset_item_range, pickle_name, IS_MONO, NO_DEPTH)
            sequences.append(ds)
            print(f'!!!!!!more_sequences: {len(ds)}!!!!!!')

        self.dataset = ConcatDataset(sequences)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ndx):
        return self.dataset[ndx]