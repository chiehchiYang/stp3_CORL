import os
import json
import numpy as np
import cv2
import torch
import torch.utils.data
import torchvision
from PIL import Image
import PIL
from pyquaternion import Quaternion

from stp3.utils.geometry import (
    update_intrinsics,
    mat2pose_vec,
    invert_matrix_egopose_numpy,
)

from stp3.utils.instance import convert_instance_mask_to_center_and_offset_label, conver_goal_map_to_center_and_offset

import stp3.utils.sampler as trajectory_sampler


class CarlaDataset(torch.utils.data.Dataset):
    SAMPLE_INTERVAL = 0.5  # SECOND
    def __init__(self, root_dir, is_train, cfg, is_test=False):
        super(CarlaDataset, self).__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        self.is_test = is_test
        self.sequence_length = cfg.TIME_RECEPTIVE_FIELD + cfg.N_FUTURE_FRAMES # 3 + 4 
        self.receptive_field = cfg.TIME_RECEPTIVE_FIELD # 3 
        self.cfg = cfg
        self.n_samples = self.cfg.PLANNING.SAMPLE_NUM

        self.normalise_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ]
        )

        self.front = []
        self.left = []
        self.right = []
        self.rear = []
        self.front_depth = []
        self.left_depth = []
        self.right_depth = []
        self.rear_depth = []
        self.x = []
        self.y = []
        self.x_command = []
        self.y_command = []
        self.theta = []
        self.steer = []
        self.throttle = []
        self.brake = []
        self.command = []
        self.velocity = []
        self.hdmap = []
        self.instance_map = []
        self.goal_points = []
        
        self.get_train_val()

    def get_train_val(self):
        
        train_towns =  ['Town01', 'Town02', 'Town03', 'Town05'] 
        
        if self.is_test:
            val_towns = ['Town10HD']  
        else:
            val_towns = [ 'Town01_val', 'Town02_val', 'Town03_val', 'Town05_val']
        
        train_data, val_data = [], []
        
        for town in train_towns:
            train_data.append(os.path.join(self.root_dir, town))
        for town in val_towns:
            val_data.append(os.path.join(self.root_dir, town))

        require_data = train_data if self.is_train else val_data

        for subroot in require_data:
            
            preload_front = []
            preload_left = []
            preload_right = []
            preload_rear = []
            preload_front_depth = []
            preload_left_depth = []
            preload_right_depth = []
            preload_rear_depth = []
            preload_x = []
            preload_y = []
            preload_x_command = []
            preload_y_command = []
            preload_theta = []
            preload_steer = []
            preload_throttle = []
            preload_brake = []
            preload_command = []
            preload_velocity = []
            preload_hdmap = []
            preload_instance_map = []
            preload_goal_points = []

            root_files = os.listdir(subroot)
            scenarios_files = [folder for folder in root_files if not os.path.isfile(os.path.join(subroot, folder))]
            
            for scenario in scenarios_files:
                routes = os.listdir(os.path.join(subroot, scenario))
                for route in routes:
                    if route[-4:] == 'json':
                        continue
                    route_dir = os.path.join(subroot, scenario, route)
                    num_seq = len(os.listdir(route_dir + "/goal_points/")) - self.sequence_length
                    
                    for seq in range(num_seq):
                        fronts, lefts, rights, rears = [], [], [], []
                        fr_depths, le_depths, ri_depths, re_depths = [], [], [], []
                        xs, ys, thetas = [], [], []
                        hdmap = []
                        instance_map = []
                        
                        goal_points = []
                        
                        
                        for i in range(self.receptive_field): # 0,3 
                            filename = f"{str(seq+1+i).zfill(4)}.png"
                            
                            fronts.append(route_dir + "/rgb_front/" + filename)
                            lefts.append(route_dir + "/rgb_left/" + filename)
                            rights.append(route_dir + "/rgb_right/" + filename)
                            rears.append(route_dir + "/rgb_rear/" + filename)
                            fr_depths.append(route_dir + "/depth_front/" + filename)
                            le_depths.append(route_dir + "/depth_left/" + filename)
                            ri_depths.append(route_dir + "/depth_right/" + filename)
                            re_depths.append(route_dir + "/depth_rear/" + filename)
                            
                            hdmap.append(route_dir + "/new_birdview_npy/" + f"{str(seq+1+i).zfill(4)}.npy")
                            instance_map.append(route_dir + "/instance_np/" + f"{str(seq+1+i).zfill(4)}.npy")
                            
                            goal_points.append(route_dir + "/goal_points/" + f"{str(seq+1+i).zfill(4)}.npy")
                            # goal_points.append(route_dir + "/goal_points_with_motion/" + f"{str(seq+1+i).zfill(4)}.npy")
                            
                            # position
                            with open(route_dir + f"/measurements/{str(seq+1+i).zfill(4)}.json","r") as read_file:
                                data = json.load(read_file)
                            xs.append(data['x'])
                            ys.append(data['y'])
                            thetas.append(data['theta'])

                        preload_x_command.append(data['x_command'])
                        preload_y_command.append(data['y_command'])
                        preload_steer.append(data['steer'])
                        preload_throttle.append(data['throttle'])
                        preload_brake.append(data['brake'])
                        preload_command.append(data['command'])
                        preload_velocity.append(data['speed'])

                        for i in range(self.receptive_field, self.sequence_length): # ( 3, 7)
                            with open(route_dir + f"/measurements/{str(seq+1+i).zfill(4)}.json","r") as read_file:
                                data = json.load(read_file)
                            xs.append(data['x'])
                            ys.append(data['y'])
                            if np.isnan(data['theta']):
                                thetas.append(0)
                            else:
                                thetas.append(data['theta'])
                            
                            hdmap.append(route_dir + "/new_birdview_npy/" + f"{str(seq+1+i).zfill(4)}.npy")
                            instance_map.append(route_dir + "/instance_np/" + f"{str(seq+1+i).zfill(4)}.npy")
                            goal_points.append(route_dir + "/goal_points/" + f"{str(seq+1+i).zfill(4)}.npy")
                            # goal_points.append(route_dir + "/goal_points_with_motion/" + f"{str(seq+1+i).zfill(4)}.npy")
                            
                        preload_front.append(fronts)
                        preload_left.append(lefts)
                        preload_right.append(rights)
                        preload_rear.append(rears)
                        preload_front_depth.append(fr_depths)
                        preload_left_depth.append(le_depths)
                        preload_right_depth.append(ri_depths)
                        preload_rear_depth.append(re_depths)
                        preload_x.append(xs)
                        preload_y.append(ys)
                        preload_theta.append(thetas)
                        preload_hdmap.append(hdmap)
                        preload_instance_map.append(instance_map)
                        preload_goal_points.append(goal_points)
                        
                preload_dict = {}
                preload_dict['front'] = preload_front
                preload_dict['left'] = preload_left
                preload_dict['right'] = preload_right
                preload_dict['rear'] = preload_rear
                preload_dict['front_depth'] = preload_front_depth
                preload_dict['left_depth'] = preload_left_depth
                preload_dict['right_depth'] = preload_right_depth
                preload_dict['rear_depth'] = preload_rear_depth
                preload_dict['x'] = preload_x
                preload_dict['y'] = preload_y
                preload_dict['x_command'] = preload_x_command
                preload_dict['y_command'] = preload_y_command
                preload_dict['theta'] = preload_theta
                preload_dict['steer'] = preload_steer
                preload_dict['throttle'] = preload_throttle
                preload_dict['brake'] = preload_brake
                preload_dict['command'] = preload_command
                preload_dict['velocity'] = preload_velocity
                preload_dict['hdmap'] = preload_hdmap
                preload_dict['instance'] = preload_instance_map
                preload_dict['goal_points'] = preload_goal_points
                
            self.front += preload_dict['front']
            self.left += preload_dict['left']
            self.right += preload_dict['right']
            self.rear += preload_dict['rear']
            self.front_depth += preload_dict['front_depth']
            self.left_depth += preload_dict['left_depth']
            self.right_depth += preload_dict['right_depth']
            self.rear_depth += preload_dict['rear_depth']
            self.x += preload_dict['x']
            self.y += preload_dict['y']
            self.x_command += preload_dict['x_command']
            self.y_command += preload_dict['y_command']
            self.theta += preload_dict['theta']
            self.steer += preload_dict['steer']
            self.throttle += preload_dict['throttle']
            self.brake += preload_dict['brake']
            self.command += preload_dict['command']
            self.velocity += preload_dict['velocity']
            self.hdmap += preload_dict['hdmap']
            self.instance_map += preload_dict['instance']
            self.goal_points += preload_dict['goal_points']
            
            print("Preloading " + str(len(preload_dict['goal_points'])) + " sequences" )

    def __len__(self):
        return len(self.front)

    def get_future_egomotion(self, seq_x, seq_y, seq_theta):
        future_egomotions = []

        def convert_to_matrix_numpy(x, y, theta):
            matrix = np.zeros((4,4), dtype=np.float32)
            matrix[:2, :2] = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            matrix[2,2] = 1
            matrix[0,3] = x
            matrix[1,3] = y
            matrix[3,3] = 1
            return matrix

        for i in range(len(seq_x) - 1 ) :#-1):
            egopose_t0 = convert_to_matrix_numpy(seq_x[i], seq_y[i], seq_theta[i])
            egopose_t1 = convert_to_matrix_numpy(seq_x[i+1], seq_y[i+1], seq_theta[i+1])

            future_egomotion = invert_matrix_egopose_numpy(egopose_t1).dot(egopose_t0)
            future_egomotion[3, :3] = 0.0
            future_egomotion[3, 3] = 1.0

            future_egomotion = torch.Tensor(future_egomotion).float()
            future_egomotion = mat2pose_vec(future_egomotion)
            future_egomotions.append(future_egomotion.unsqueeze(0))

        return torch.cat(future_egomotions, dim=0)

    def get_hdmap(self, path ):
        
        def crop_center(img,cropx,cropy):
            _, y, x = img.shape
            startx = x//2 
            starty = y//2    
            return img[:, starty - cropy:starty+cropy, startx - cropx:startx+cropx]

        COLOR_ON = 1
        nonzero_indices = lambda arr: arr == COLOR_ON
        
        img = np.load(path, allow_pickle=True)
        img = crop_center(img, 100, 100)
        
        _, y, x = img.shape
        
        lane = np.zeros(shape=(y, x), dtype=np.uint8)
        drivable = np.zeros(shape=(y, x), dtype=np.uint8)
        
        ## get HD Map - Lane
        # CENTERLINES = 2,  LANES = 1
        index = [2, 1]
        for i in index:
            lane[nonzero_indices(img[i, :, :])] = 1
            
        ## get HD map - Drivable area 
        # ROAD = 0
        drivable[nonzero_indices(img[0, :, :])] = 1
        
        # visualization
        # plt.imshow(lane, cmap='gray')
        # plt.savefig('map.png')
        
        # down, right is the positive
        lane = lane[::-1,::-1]
        drivable = drivable[::-1,::-1]
        hdmap = np.concatenate([lane[None], drivable[None]], axis=0)
        
        return hdmap
        
    def get_labels(self, path, scale, crop):
        
        def crop_center(img,cropx,cropy):
            _, y, x = img.shape
            startx = x//2 
            starty = y//2    

            return img[:, starty - cropy:starty+cropy, startx - cropx:startx+cropx]

        COLOR_ON = 1
        nonzero_indices = lambda arr: arr == COLOR_ON
        
        img = np.load(path, allow_pickle=True)
        img = crop_center(img, 100, 100)
        
        _, y, x = img.shape
        
        pedestrian = np.zeros(shape=(y, x), dtype=np.uint8)
        vehicle = np.zeros(shape=(y, x), dtype=np.uint8)
        
        # PEDESTRIANS = 8
        pedestrian[nonzero_indices(img[8, :, :])] = 1
            
        #  VEHICLES = 3
        vehicle[nonzero_indices(img[3, :, :])] = 1
        
        vehicle = vehicle[::-1,::-1]
        pedestrian = pedestrian[::-1,::-1]
        return vehicle.copy(), pedestrian.copy()

    def get_heatmap(self, goal_points_path, hd_map_path):
        
        def create_gauss_kernel(size=5, sigma=1.):
            """
            Creates gauss kernel of given size
            Args:
                size: Square matrix size
                sigma: Deviation parameter
            Returns: Gauss Kernel of size: [size]x[size]
            """

            ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
            gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
            kernel = np.outer(gauss, gauss)
            return kernel / np.sum(kernel)
                
                    
        goal_points_index = np.load(goal_points_path)       
        hd_map = np.load(hd_map_path)
        driviable_area  = hd_map[0, :,:]

        # empty heatmap 
        heatmap = np.zeros((400, 400))

        for i, j in goal_points_index:
            heatmap = cv2.circle(heatmap, (j, i), 10, 1, -1)
            
            
        kernel_size = 20
        sigma = 6 # 4

        gaussian_filter = create_gauss_kernel(kernel_size, sigma=sigma)
        heatmap = cv2.filter2D(heatmap, -1, gaussian_filter)

        for i, j in goal_points_index:
            heatmap[i, j] = 1.0
        heatmap = heatmap * driviable_area  # Set probability to 0 on in non-driveable area

        def crop_center(img,cropx,cropy):
            y, x = img.shape
            startx = x//2 
            starty = y//2    

            return img[starty - cropy:starty+cropy, startx - cropx:startx+cropx]
        
        
        heatmap = crop_center(heatmap, 100, 100)
        heatmap = heatmap[::-1,::-1]
        
        return heatmap.copy()

    def get_goals_map(self, path):
                
        index = np.load(path)
        tmp = np.zeros((400, 400))
            
        for i, j in index:
            
            if i == 300:
                i = 299
            if j == 300:
                j = 299
        
            tmp = cv2.circle(tmp, (j, i), 3, 1, -1)
        
        def crop_center(img,cropx,cropy):
            y, x = img.shape
            startx = x//2 
            starty = y//2    

            return img[starty - cropy:starty+cropy, startx - cropx:startx+cropx]
        
        goals = crop_center(tmp, 100, 100)
        goals = goals[::-1,::-1]
        
        return goals.copy()
    
    def get_goal_points_map(self, path):
                
        index = np.load(path)
        tmp = np.zeros((400, 400))

        for i, j in index:
            
            if i == 300:
                i = 299
            if j == 300:
                j = 299
            
            tmp[i][j] = 1.0

        def crop_center(img,cropx,cropy):
            y, x = img.shape
            startx = x//2 
            starty = y//2    

            return img[starty - cropy:starty+cropy, startx - cropx:startx+cropx]
        
        goals = crop_center(tmp, 100, 100)
        goals = goals[::-1,::-1]
        
        return goals.copy()
    
    
    def get_instance_goals_map(self, path):
                
        index = np.load(path)
        tmp = np.zeros((400, 400))
        
        counter = 1
        for i, j in index:
            
            if i == 300:
                i = 299
            if j == 300:
                j = 299
        
            tmp = cv2.circle(tmp, (j, i), 3, counter, -1)
            counter += 1
        
        def crop_center(img,cropx,cropy):
            y, x = img.shape
            startx = x//2 
            starty = y//2    

            return img[starty - cropy:starty+cropy, startx - cropx:startx+cropx]
        
        goals = crop_center(tmp, 100, 100)
        goals = goals[::-1,::-1]
        
        return goals.copy()
    

    
    def get_instance_map(self, path, instance_len):
        
        def crop_center(img,cropx,cropy):
            y, x = img.shape
            startx = x//2 
            starty = y//2    

            return img[starty - cropy:starty+cropy, startx - cropx:startx+cropx]
        
        img = np.load(path, allow_pickle=True)
        instance_np = crop_center(img, 100, 100)
        
        if instance_len < int(np.max(instance_np)) :
            instance_len = int(np.max(instance_np))
        instance_np = instance_np[::-1,::-1]

        return instance_np.copy() , instance_len
    
    def get_trajectory_sampling(self, v0, steering):

        Kappa = 2 * steering / 2.588

        # initial state
        T0 = np.array([0.0, 1.0])  # define front
        N0 = np.array([1.0, 0.0]) if Kappa <= 0 else np.array([-1.0, 0.0])  # define side

        t_start = 0  # second
        t_end = self.cfg.N_FUTURE_FRAMES * self.SAMPLE_INTERVAL  # second
        t_interval = self.SAMPLE_INTERVAL / 10
        tt = np.arange(t_start, t_end + t_interval, t_interval)
        sampled_trajectories_fine = trajectory_sampler.sample(v0, Kappa, T0, N0, tt, self.n_samples)
        sampled_trajectories = sampled_trajectories_fine[:, ::10]
        return sampled_trajectories

    def get_cam_para(self):
        def get_cam_to_ego(dof):
            yaw = dof[5]
            rotation = Quaternion(scalar=np.cos(yaw/2), vector=[0, 0, np.sin(yaw/2)])
            translation = np.array(dof[:3])[:, None]
            cam_to_ego = np.vstack([
                np.hstack((rotation.rotation_matrix,translation)),
                np.array([0,0,0,1])
            ])
            return cam_to_ego

        cam_front = [1.3, 0.0, 2.3, 0.0, 0.0, 0.0] # x,y,z,roll,pitch, yaw
        cam_left = [1.3, 0.0, 2.3, 0.0, 0.0, -60.0]
        cam_right = [1.3, 0.0, 2.3, 0.0, 0.0, 60.0]
        cam_rear = [-1.3, 0.0, 2.3, 0.0, 0.0, 180.0]
        front_to_ego = torch.from_numpy(get_cam_to_ego(cam_front)).float().unsqueeze(0)
        left_to_ego = torch.from_numpy(get_cam_to_ego(cam_left)).float().unsqueeze(0)
        right_to_ego = torch.from_numpy(get_cam_to_ego(cam_right)).float().unsqueeze(0)
        rear_to_ego = torch.from_numpy(get_cam_to_ego(cam_rear)).float().unsqueeze(0)
        extrinsic = torch.cat([front_to_ego, left_to_ego, right_to_ego, rear_to_ego], dim=0)

        sensor_data = {
            'width': 320,
            'height': 160,
            'fov': 60
        }
        w = sensor_data['width']
        h = sensor_data['height']
        fov = sensor_data['fov']
        f = w / (2 * np.tan(fov * np.pi/ 360))
        Cu = w / 2
        Cv = h / 2
        intrinsic = torch.Tensor([
            [f, 0, Cu],
            [0, f, Cv],
            [0, 0, 1]
        ])
        intrinsic = update_intrinsics(
            intrinsic, (h-256)/2, (w-256)/2,
            scale_width=1,
            scale_height=1
        )
        
        ############################# rear camear
        
        sensor_data_rear = {
            'width': 320,
            'height': 160,
            'fov': 120
        }
        w = sensor_data_rear['width']
        h = sensor_data_rear['height']
        fov = sensor_data_rear['fov']
        f = w / (2 * np.tan(fov * np.pi/ 360))
        Cu = w / 2
        Cv = h / 2
        intrinsic_rear = torch.Tensor([
            [f, 0, Cu],
            [0, f, Cv],
            [0, 0, 1]
        ])
        intrinsic_rear = update_intrinsics(
            intrinsic_rear, (h-256)/2, (w-256)/2,
            scale_width=1,
            scale_height=1
        )
        
        intrinsic = torch.cat([intrinsic, intrinsic, intrinsic, intrinsic_rear], dim=0)
        intrinsic = intrinsic.reshape((4, 3, 3))
        
        return extrinsic, intrinsic

    def get_depth(self, data):
        """
        Computes the normalized depth
        """
        data = data.astype(np.float32)

        normalized = np.dot(data, [65536.0, 256.0, 1.0])
        normalized /= (256 * 256 * 256 - 1)
        return torch.from_numpy(normalized * 1000)

    def __getitem__(self, index):
        data = {}
        keys = ['image', 'depths', 'segmentation', 'pedestrian', 'extrinsics', 'intrinsics', 'hdmap', 'gt_trajectory', 'instance', 'instance_goal', 'centerness', 'offset','flow', 'heatmap', 'goal_points' ]

        for key in keys:
            data[key] = []

        seq_fronts = self.front[index]
        seq_lefts = self.left[index]
        seq_rights = self.right[index]
        seq_rears = self.rear[index]
        seq_front_depths = self.front_depth[index]
        seq_left_depths = self.left_depth[index]
        seq_right_depths = self.right_depth[index]
        seq_rear_depths = self.rear_depth[index]
        seq_hdmaps = self.hdmap[index]
        seq_instance = self.instance_map[index]
        seq_goal_points = self.goal_points[index]

        seq_x = self.x[index]
        seq_y = self.y[index]
        seq_theta = self.theta[index]
        

        for i in range(self.receptive_field):
            images = []
            
            images.append(self.normalise_image(np.array(
                scale_and_crop_image(Image.open(seq_fronts[i]), scale=1.25, crop=256))).unsqueeze(0))
            images.append(self.normalise_image(np.array(
                scale_and_crop_image(Image.open(seq_lefts[i]), scale=1.25, crop=256))).unsqueeze(0))
            images.append(self.normalise_image(np.array(
                scale_and_crop_image(Image.open(seq_rights[i]), scale=1.25, crop=256))).unsqueeze(0))
            images.append(self.normalise_image(np.array(
                scale_and_crop_image(Image.open(seq_rears[i]), scale=1.25,crop=256))).unsqueeze(0))
            images = torch.cat(images, dim=0)
            data['image'].append(images.unsqueeze(0))
            depths = []
            depths.append(self.get_depth(np.array(
                scale_and_crop_image(Image.open(seq_front_depths[i]), scale=1.25, crop=256))).unsqueeze(0))
            depths.append(self.get_depth(np.array(
                scale_and_crop_image(Image.open(seq_left_depths[i]), scale=1.25, crop=256))).unsqueeze(0))
            depths.append(self.get_depth(np.array(
                scale_and_crop_image(Image.open(seq_right_depths[i]), scale=1.25, crop=256))).unsqueeze(0))
            depths.append(self.get_depth(np.array(
                scale_and_crop_image(Image.open(seq_rear_depths[i]), scale=1.25, crop=256))).unsqueeze(0))  
            
            
            depths = torch.cat(depths, dim=0)
            data['depths'].append(depths.unsqueeze(0))
            extrinsics, intrinsics = self.get_cam_para()
            data['extrinsics'].append(extrinsics.unsqueeze(0))
            data['intrinsics'].append(intrinsics.unsqueeze(0))
            
            
            data['hdmap'].append(torch.from_numpy(self.get_hdmap(seq_hdmaps[i])).unsqueeze(0))
            
            # fix for theta=nan in some measurements
            if np.isnan(seq_theta[i]):
                seq_theta[i] = 0.
                
        ego_x = seq_x[self.receptive_field-1]
        ego_y = seq_y[self.receptive_field-1]
        ego_theta = seq_theta[self.receptive_field-1]
        
        instance_len = 0

        for i in range(self.sequence_length):
            if i >= self.receptive_field-1:
                local_waypoint = transform_2d_points(np.zeros((1, 3)),
                                                     np.pi / 2 - seq_theta[i], -seq_x[i], -seq_y[i],
                                                     np.pi / 2 - ego_theta, -ego_x, -ego_y)
                local_waypoint = local_waypoint * [1.0, -1.0, 1.0]
                data['gt_trajectory'].append(torch.from_numpy(local_waypoint))
            
            segmentation, pedestrian = self.get_labels(seq_hdmaps[i], 1.1, 200)
            data['segmentation'].append(torch.from_numpy(segmentation).unsqueeze(0).unsqueeze(0))
            
            # change pedestrain to goal prediction 
            
            goal = self.get_goals_map(seq_goal_points[i])
            data['pedestrian'].append(torch.from_numpy(goal).unsqueeze(0).unsqueeze(0))
            
            heatmap = self.get_heatmap(seq_goal_points[i], seq_hdmaps[i])
            data['heatmap'].append(torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0))
            
            instance, instance_len = self.get_instance_map(seq_instance[i], instance_len)
            data['instance'].append(torch.from_numpy(instance).unsqueeze(0))
            
            instance_goal = self.get_instance_goals_map(seq_goal_points[i])
            data['instance_goal'].append(torch.from_numpy(instance_goal).unsqueeze(0))
            
            # for val ( points of goal )
            # goal_points
            
            goal_points = self.get_goal_points_map(seq_goal_points[i])
            data['goal_points'].append(torch.from_numpy(goal_points).unsqueeze(0))
            
        R = np.array([
            [np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
            [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)]
        ])
        local_command_point = np.array([self.x_command[index] - ego_x, self.y_command[index] - ego_y])
        local_command_point = R.T.dot(local_command_point)
        local_command_point = local_command_point * [1.0, -1.0]
        data['target_point'] = torch.from_numpy(local_command_point)

        if self.command[index] == 1:
            data['command'] = 'LEFT'
        elif self.command[index] == 2:
            data['command'] = 'RIGHT'
        elif self.command[index] == 3:
            data['command'] = 'FORWARD'
        else:
            data['command'] = 'LANE'
            
        data['steer'] = self.steer[index]
        data['throttle'] = self.throttle[index]
        data['brake'] = self.brake[index]
        data['velocity'] = self.velocity[index]
        data['future_egomotion'] = self.get_future_egomotion(seq_x, seq_y, seq_theta)
        data['sample_trajectory'] = torch.from_numpy(self.get_trajectory_sampling(self.velocity[index], self.steer[index])).float()

                
        for key, value in data.items():
            
            if key in keys:
                if key == 'centerness' or key == 'offset' or key == 'flow': #  and self.cfg.LIFT.GT_DEPTH is False:
                    continue
                data[key] = torch.cat(value, dim=0)
                
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])
                
        instance_centerness, instance_offset = conver_goal_map_to_center_and_offset(data['instance_goal'])
        
        # change centerness and offset to goal heatmap and off set 
        data['centerness'] = instance_centerness
        data['offset'] = instance_offset # 2 channel ( offset x and offset y)
        
        """
        Returns
        -------
            data: dict with the following keys:
                image: torch.Tensor<float> (T, N, 3, H, W)
                    normalised cameras images with T the sequence length, and N the number of cameras.
                intrinsics: torch.Tensor<float> (T, N, 3, 3)
                    intrinsics containing resizing and cropping parameters.
                extrinsics: torch.Tensor<float> (T, N, 4, 4)
                    6 DoF pose from world coordinates to camera coordinates.
                segmentation: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                    (H_bev, W_bev) are the pixel dimensions in bird's-eye view.
                instance: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                centerness: torch.Tensor<float> (T, 1, H_bev, W_bev)
                offset: torch.Tensor<float> (T, 2, H_bev, W_bev)
                flow: torch.Tensor<float> (T, 2, H_bev, W_bev)
                future_egomotion: torch.Tensor<float> (T, 6)
                    6 DoF egomotion t -> t+1
                    
                goal instance: torch.Tensor<int64> (T, 1, H_bev, W_bev) # 

        """

        return data
    

def scale_and_crop_image(image, scale=1., crop=256):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    (width, height) = (int(image.width // scale), int(image.height // scale))
    # origina size : 160, 320 
    # Final dimension: 128, 256 
    
    # width = 256 
    # height  = 128
    
    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    start_x = height//2 - crop//2
    start_y = width//2 - crop//2
    cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]
    
    return image # cropped_image


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:, 2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T

    # reset z-coordinate
    out[:, 2] = xyz[:, 2]

    return out
