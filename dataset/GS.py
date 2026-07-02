import torch
import gin
import numpy as np 
from typing import Optional
import pickle
import cv2, os
from utils.transform_utils import remove_outliers, MinMaxScaler
from dataset import colmap_utils
from utils import gs_utils
from utils import gpu_utils
import glob, yaml
from pathlib import Path
import random
from PIL import Image
from time import time
from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

@gin.configurable
class SplatfactoDataset(torch.utils.data.IterableDataset):
    def __init__(self, 
                 train_or_test,
                 nerfstudio_folder,
                 colmap_folder,
                 dataset_folder,
                 load_pose_src, #[colmap or nerfstudio]
                 sample_ratio_test: Optional[float],
                 image_per_scene: Optional[int],
                 remove_outlier_ndevs: float,
                 max_gs_num: int,
                 cache_steps: int,
                 cache_num_scenes: int, #Default: cache_num_scenes=1, cache_steps=1
                 split_across_gpus: bool,
                 low_resolution: int = 256,
                 high_resolution: int = 1024,
                 input_ckpt_step: int = 14999,
                 background_color: list=[0,0,0],
                 ):
        self.train_or_test = train_or_test
        self.image_per_scene = image_per_scene
        self.sample_ratio_test = sample_ratio_test
        self.global_step = 0
        self.low_resolution = str(low_resolution)
        self.high_resolution = str(high_resolution)
        self.input_ckpt_step = input_ckpt_step
        
        # self.nerfstudio_folders = sorted([os.path.join(nerfstudio_folder, ls, 'splatfacto') for ls in os.listdir(nerfstudio_folder)]) 
        # if colmap_folder.endswith('.txt'):
        #     self.colmap_folders = [os.path.join(colmap_path, ls) for ls in open(colmap_folder).read().splitlines()]
        # elif os.path.isdir(colmap_folder):
        #     self.colmap_folders = sorted([os.path.join(colmap_folder, ls) for ls in os.listdir(colmap_folder)])
        
        
        resolution_list = [[self.low_resolution, self.high_resolution]]
        ckpt_name = f'ckpt_{self.input_ckpt_step}_rank0.pt'
        self.nerfstudio_folders = []
        self.colmap_folders = []
        dataset_list = sorted(os.listdir(dataset_folder))
        for ls in dataset_list:
            for i in range(len(resolution_list)):
                if os.path.exists(os.path.join(dataset_folder, ls, f'{resolution_list[i][0]}','gaussian_splatting','ckpts', ckpt_name)):
                    if os.path.exists(os.path.join(dataset_folder, ls, f'{resolution_list[i][1]}')):
                        if os.path.exists(os.path.join(dataset_folder, ls, f'{resolution_list[i][1]}', 'images.txt')) and os.path.getsize(os.path.join(dataset_folder, ls, f'{resolution_list[i][1]}', 'images.txt')) > 0:
                            self.nerfstudio_folders.append(os.path.join(dataset_folder, ls, f'{resolution_list[i][0]}','gaussian_splatting'))
                            self.colmap_folders.append(os.path.join(dataset_folder, ls, f'{resolution_list[i][1]}'))
                        else:
                            print(f'Warning: {os.path.join(dataset_folder, ls, f"{resolution_list[i][1]}", "images.txt")} does not exist or is empty')

        assert len(self.nerfstudio_folders) == len(self.colmap_folders), 'The number of folders in nerfstudio and colmap should be the same'
        self.folders = list(zip(self.nerfstudio_folders, self.colmap_folders))
        self.unlocked_scene_indices = list(range(len(self.folders)))
        self.unlocked_scene_num = len(self.folders)
        self.increment_steps = 0
        self.load_pose_src = load_pose_src

        self.remove_outlier_ndevs = remove_outlier_ndevs
        self.cache_steps, self.cache_num_scenes = cache_steps, cache_num_scenes
        self.split_across_gpus = split_across_gpus
        self.max_gs_num = max_gs_num
        self.cache_scenes = []
        self.background_color = background_color

        if train_or_test in ['test']: 
            # For test set, we need to split data across device deterministically
            self.remaining_scenes = list(range(len(self.folders)))
            assert self.cache_num_scenes==1 and self.cache_steps==1, 'For test, we do not cache'
            # For DDP evaluation, we need to chunk the data
            try:
                world_size = torch.cuda.device_count()
                rank = torch.distributed.get_rank()
            except:
                world_size, rank = 1, 0
            chunk_size = len(self.remaining_scenes)//world_size
            if rank == world_size-1:
                self.remaining_scenes = self.remaining_scenes[rank*chunk_size:]
            else:
                self.remaining_scenes = self.remaining_scenes[rank*chunk_size:(rank+1)*chunk_size]
        else:
            self.counter = 0

    def refresh_remaining_training(self,):
        unlocked_indices = self.unlocked_scene_indices
        if self.split_across_gpus:
            self.remaining_scenes = self.get_thisworker_split(N=len(unlocked_indices))
            self.remaining_scenes = [unlocked_indices[i] for i in self.remaining_scenes]
        else:
            self.remaining_scenes = self.get_thisworker_split(N=len(unlocked_indices))
            self.remaining_scenes = [unlocked_indices[i] for i in self.remaining_scenes]
            random.shuffle(self.remaining_scenes)
        self.counter += 1
        return
    
    def step_and_unlock(self):
        if self.increment_steps <= 0:
            return
        self.global_step += 1
        if self.global_step % self.increment_steps == 0:
            new_num = len(self.folders)
            self.unlocked_scene_indices = list(range(new_num))
            self.unlocked_scene_num = new_num
            self.refresh_remaining_training()

    def get_thisworker_split(self, N):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return list(range(N))
        per_worker = N // worker_info.num_workers
        worker_id = worker_info.id
        if worker_id == worker_info.num_workers-1:
            return list(range(worker_id*per_worker, N))
        else:
            return list(range(worker_id*per_worker, (worker_id+1)*per_worker))
        
    def random_split_to_remaining(self, ):
        """
        #Generate a new permutation of the folders
        """
        # 1. Split self.folders across processes
        np.random.seed(torch.distributed.get_rank())
        torch.manual_seed(torch.distributed.get_rank())
        rng_state = np.random.get_state()
        world_size = torch.cuda.device_count()
        rank = torch.distributed.get_rank()     
        np.random.seed(self.counter)
        #Pad to world_size*k
        permutation = np.random.permutation(len(self.folders))
        pad_num = world_size - len(self.folders)%world_size
        if pad_num > 0 and world_size > 1:
            permutation = np.concatenate([permutation, permutation[:pad_num]])
        np.random.set_state(rng_state)

        chunk_size = len(permutation)//world_size
        if rank == world_size-1:
            remaining_scenes_for_thisprocess = permutation[rank*chunk_size:]
        else:
            remaining_scenes_for_thisprocess = permutation[rank*chunk_size:(rank+1)*chunk_size]

        # 2. Split scenes across works
        split_id = self.get_thisworker_split(N=len(remaining_scenes_for_thisprocess))
        self.remaining_scenes = [remaining_scenes_for_thisprocess[i] for i in split_id]
        worker_info = torch.utils.data.get_worker_info()
        return 

    @gin.configurable
    def read_image(self, path, background):
        try:
            pil_image = Image.open(path)
        except:
            print(f'Warning: {path} cannot be opened')

        image = np.array(pil_image, dtype="uint8").astype(np.float32) / 255.0
        mask = None
        if 'real' in path.lower():
            possible_mask_filename = path.replace('images','masks') #TODO: hardcoded
            if os.path.exists(possible_mask_filename):
                mask = np.array(Image.open(possible_mask_filename)).astype(image.dtype)/255.0
                mask = torch.from_numpy(mask)
        # else:
        #     mask = None

        image = torch.from_numpy(image)
        if image.shape[2] == 4:
            image = image[:, :, :3] * image[:, :, -1:] + background * (1.0 - image[:, :, -1:])
        elif mask is not None:
            image_rgb = image * mask[...,None] + background * (1.0 - mask[...,None])
            # As we need to preserve the mask for evaluation, we save the image RGBA
            image = torch.concat([image_rgb, mask[...,None]], axis=-1) # Hardcoded here, only for the real dataset
        return image

    def load_gs_params_fromnerfstudio(self, nerfstudio_dir, idx):
        skip_params = gin.query_parameter('FeaturePredictor.input_features')
        if gin.query_parameter("training.pretrain_steps") > 0:
            skip_params = skip_params+gin.query_parameter('create_pseudo_target.take_from_input')
        ckpt_file = os.path.join(
            nerfstudio_dir,
            'ckpts',
            f'ckpt_{self.input_ckpt_step}_rank0.pt',
        )
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(f'Cannot find input 3DGS checkpoint: {ckpt_file}')
            
        # try:
        #     ckpt_file = glob.glob(nerfstudio_dir + 'ckpts/ckpt_*_rank0.pt')[-1] #Take the last checkpoint
        # except:
        #     print(f'Warning: {nerfstudio_dir} does not have ckpts/ckpt_*_rank0.pt')
        #     exit()
        
        # ckpt = torch.load(ckpt_file, map_location='cpu')
        # ckpt = {k.replace('_model.gauss_params.',''):v for k,v in ckpt.items() if 'gauss_params' in k}
        # gs_params = {k:ckpt[k] for k in set(skip_params)}
        
        pt = torch.load(ckpt_file, map_location='cpu')['splats']
        key_mapping = {'sh0': 'features_dc', 'shN': 'features_rest'}
        new_pt = {key_mapping.get(k, k): v for k, v in pt.items()}
        gs_params = {k:new_pt[k] for k in set(skip_params)}
        gs_params['features_dc'] = gs_params['features_dc'].squeeze(dim=1)
        gs_params['opacities'] = gs_params['opacities'].unsqueeze(dim=1)
        
        # Convert means from COLMAP (OpenCV) to OpenGL
        # Flip Y and Z coordinates
        # gs_params['means'][:, 1] *= -1
        # gs_params['means'][:, 2] *= -1
        
        # Convert rotation (quaternions) to OpenGL
        # q_new = q_flip * q_old = [0, 1, 0, 0] * [w, x, y, z] = [-x, w, -z, y]
        # (Rotation of 180 degrees around X axis)
        # w = gs_params['quats'][:, 0].clone()
        # x = gs_params['quats'][:, 1].clone()
        # y = gs_params['quats'][:, 2].clone()
        # z = gs_params['quats'][:, 3].clone()
        
        # gs_params['quats'][:, 0] = -x
        # gs_params['quats'][:, 1] = w
        # gs_params['quats'][:, 2] = -z
        # gs_params['quats'][:, 3] = y
        
        # Convert means from COLMAP coordinate system to OpenGL coordinate system
        # COLMAP uses right-handed coordinate system with Y pointing down and Z pointing forward
        # OpenGL uses right-handed coordinate system with Y pointing up and Z pointing backward
        # gs_params['means'][:, 1:3] *= -1
        
        # Remove inf or nanf
        select = torch.ones(gs_params['means'].shape[0], dtype=torch.bool)
        for key in gs_params:
            if key=='features_rest':
                select = select & ~torch.isnan(gs_params[key].sum(dim=1)).any(dim=1)
            else:
                select = select & ~torch.isnan(gs_params[key]).any(dim=1)
        for key in gs_params:
            gs_params[key] = gs_params[key][select]

        # Filter the outliers
        if self.remove_outlier_ndevs > 0:
            _,inlier_mask = remove_outliers(gs_params['means'], n_devs=self.remove_outlier_ndevs)
            for key in gs_params:
                gs_params[key] = gs_params[key][inlier_mask]

        # Truncate gs params if num > self.max_gs_num (randomly sample to preserve spatial distribution)
        N = gs_params['means'].shape[0]
        if N > self.max_gs_num:
            random_indices = torch.randperm(N)[:self.max_gs_num]
            inlier_mask = torch.zeros(N, dtype=torch.bool)
            inlier_mask[random_indices] = True
            for key in gs_params:
                gs_params[key] = gs_params[key][inlier_mask]

        # Normalize the means and scales (we need to use the scaler to transform the camera later)
        # Auto-detect coordinate system based on data range (refer to inference_lgm.py)
        means_mean = gs_params['means'].mean(dim=0)
        if abs(means_mean.mean().item()) < 0.1:
            scaler = MinMaxScaler(preserve_ratio=True, already_centered=True, already_scaled=False)
        else:
            scaler = MinMaxScaler(preserve_ratio=True, already_centered=False, already_scaled=False)

        gs_params['means'] = scaler.fit_transform(gs_params['means']) 
        gs_params['scales'] = gs_params['scales'] + torch.log(scaler.scale_)

        inf_mask = torch.isinf(gs_params['scales']).sum(dim=1).bool()
        valid_mask = (~inf_mask).bool()
        inrange_mask = torch.all((gs_params['means'] >= 0) & (gs_params['means'] <= 1), dim=1)
        valid_mask = valid_mask & inrange_mask
        for key in gs_params:
            gs_params[key] = gs_params[key][valid_mask]
            if torch.isnan(gs_params[key]).any():
                print(f'Warning: {key} contains nan', nerfstudio_dir)

        return gs_params, scaler

    def load_images_cameras_fromnerfstudio(self, nerfstudio_dir, colmap_dir):
        with open(nerfstudio_dir + '/camera_for-3d-denoise.pkl', 'rb') as f:
            meta = pickle.load(f)
        train_imgs_path, test_imgs_path = [], []

        image_names = os.listdir(colmap_dir + '/images')
        # Hard coded, only used for real-world dataset
        if os.path.isfile(os.path.join(colmap_dir,'ood-test_split.txt')):
            ood_test_img_names = []
            with open(os.path.join(colmap_dir,'ood-test_split.txt')) as f:
                for line in f.readlines():
                    ood_test_img_names.append(line.strip())
        else:
            ood_test_img_names = None

        TESTSET_ELEVATION = False
        for i, name in enumerate(sorted(image_names)): # TODO [The order is not necessarily aligned with the camera_to_worlds]
            if 'elevation' in name: # Hardcoded: test set (TODO)
                assert self.train_or_test == 'test'
                # We only use elevation-70/80/90
                TESTSET_ELEVATION = True
                if 'elevation90' in name or 'elevation80' in name or 'elevation70' in name:
                    test_imgs_path.append(os.path.join(colmap_dir, 'images', name))
            else:
                if name.startswith('test') or name.startswith('frame_eval'):
                    test_imgs_path.append(os.path.join(colmap_dir, 'images', name))
                else:
                    train_imgs_path.append(os.path.join(colmap_dir, 'images', name))
        # Check: align the order of camer poses with the images 
        # print(test_imgs_path)
        # print(train_imgs_path)
        if TESTSET_ELEVATION:
            meta['test_camera_to_worlds'] = meta['test_camera_to_worlds'][-3*3:]
        if ood_test_img_names!=None:
            ood_ids = [i for i, path in enumerate(test_imgs_path) if os.path.basename(path) in ood_test_img_names]
            # selected test_imgs_path and test_camera_to_worlds
            test_imgs_path = [test_imgs_path[i] for i in ood_ids]
            meta['test_camera_to_worlds'] = meta['test_camera_to_worlds'][ood_ids]
        return meta, train_imgs_path, test_imgs_path

    def load_images_cameras_fromcolmap(self, colmap_dir, nerfstudio_dir):
        recon_dir = Path(os.path.join(colmap_dir, "sparse/0"))
        recon2_dir = Path(os.path.join(nerfstudio_dir.replace('/gaussian_splatting',''), "sparse/0"))
        if (Path(colmap_dir) / "cameras.txt").exists():
            cam_id_to_camera = colmap_utils.read_cameras_text(Path(colmap_dir) / "cameras.txt")
        # if (Path(colmap_dir) / "images.txt").exists():
        #     im_id_to_image = colmap_utils.read_images_text(Path(colmap_dir) / "images.txt")
        if (recon2_dir / "images.bin").exists():
            im_id_to_image = colmap_utils.read_images_binary(recon2_dir / "images.bin")
        
        # if (Path(colmap_dir) / "cameras.txt").exists():
        #     cam_id_to_camera = colmap_utils.read_cameras_text(Path(colmap_dir) / "cameras.txt")
        #     im_id_to_image = colmap_utils.read_images_text(Path(colmap_dir) / "images.txt")
        # elif (recon_dir / "cameras.bin").exists():
        #     cam_id_to_camera = colmap_utils.read_cameras_binary(recon_dir / "cameras.bin")
        #     im_id_to_image = colmap_utils.read_images_binary(recon_dir / "images.bin")
              
        # if (recon_dir / "cameras.bin").exists():
        #     cam_id_to_camera = colmap_utils.read_cameras_binary(recon_dir / "cameras.bin")
        #     im_id_to_image = colmap_utils.read_images_binary(recon_dir / "images.bin")
        # elif (Path(colmap_dir) / "cameras.txt").exists():
        #     cam_id_to_camera = colmap_utils.read_cameras_text(Path(colmap_dir) / "cameras.txt")
        #     im_id_to_image = colmap_utils.read_images_text(Path(colmap_dir) / "images.txt")
        # else:
        #     raise ValueError(f"Could not find cameras.txt or cameras.bin in {recon_dir}")

        # Parse cameras
        assert len(cam_id_to_camera) == 1, "Only one camera is supported"
        for cam_id, cam_data in cam_id_to_camera.items():
            camera = colmap_utils.parse_colmap_camera_params(cam_data)
            # print('camera:', camera)
        assert camera['camera_model'] in ['SIMPLE_PINHOLE', 'PINHOLE'], "Only pinhole camera is supported"

        meta = {
            'fx':camera['fl_x'], 'fy':camera['fl_y'],
            'cx':camera['cx'], 'cy':camera['cy'],
            'width':camera['w'], 'height':camera['h']
        }

        for key in meta:
            meta[key] = torch.tensor(meta[key], dtype=torch.float32)
        c2ws, image_names = [], []

        ordered_im_id = sorted(im_id_to_image.keys(), key=lambda x: im_id_to_image[x].name)
        for im_id in ordered_im_id:
            im_data = im_id_to_image[im_id]
            # NB: COLMAP uses Eigen / scalar-first quaternions
            # * https://colmap.github.io/format.html
            # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
            # the `rotation_matrix()` handles that format for us.
            rotation = colmap_utils.qvec2rotmat(im_data.qvec)
            translation = im_data.tvec.reshape(3, 1)
            w2c = np.concatenate([rotation, translation], 1)
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
            c2w = np.linalg.inv(w2c)
            # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
            c2w[0:3, 1:3] *= -1
            c2ws.append(c2w)
            image_names.append(im_data.name) #test_XXX.png
            # print(f'Image {im_data.name} with id {im_id} has camera pose:\n{c2w}')
        
        
        
        # ordered_im_id = sorted(im_id_to_image.keys(), key=lambda x: im_id_to_image[x].name)

        # camtoworlds = []
        # for im_id in ordered_im_id:
        #     im_data = im_id_to_image[im_id]
        #     rotation = colmap_utils.qvec2rotmat(im_data.qvec)
        #     translation = im_data.tvec.reshape(3, 1)
        #     w2c = np.concatenate([rotation, translation], 1)
        #     w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        #     c2w = np.linalg.inv(w2c)
        #     camtoworlds.append(c2w)
        #     image_names.append(im_data.name)
        # camtoworlds = np.array(camtoworlds)

        # points = []
        # points3D_file = os.path.join(recon2_dir, "points3D.bin")
        # if os.path.exists(points3D_file):
        #     id_to_point = colmap_utils.read_points3D_binary(points3D_file)
        # else:
        #     print(f'Warning: {points3D_file} does not exist!')
        # for point_id in id_to_point:
        #     point = id_to_point[point_id]
        #     points.append(point.xyz)
        # points = np.array(points)

        # T1 = similarity_from_cameras(camtoworlds)
        # camtoworlds = transform_cameras(T1, camtoworlds)
        # points = transform_points(T1, points)

        # T2 = align_principal_axes(points)
        # camtoworlds = transform_cameras(T2, camtoworlds)
        # points = transform_points(T2, points)

        # transform = T2 @ T1

        # # Fix for up side down. We assume more points towards
        # # the bottom of the scene which is true when ground floor is
        # # present in the images.
        # if np.median(points[:, 2]) > np.mean(points[:, 2]):
        #     # rotate 180 degrees around x axis such that z is flipped
        #     T3 = np.array(
        #         [
        #             [1.0, 0.0, 0.0, 0.0],
        #             [0.0, -1.0, 0.0, 0.0],
        #             [0.0, 0.0, -1.0, 0.0],
        #             [0.0, 0.0, 0.0, 1.0],
        #         ]
        #     )
        #     camtoworlds = transform_cameras(T3, camtoworlds)
        #     points = transform_points(T3, points)
        #     transform = T3 @ transform

        # for im_id in ordered_im_id:
        #     im_data = im_id_to_image[im_id]
        #     # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        #     c2w = camtoworlds[im_id-1]
        #     c2w[0:3, 1:3] *= -1
        #     c2ws.append(c2w)



        poses = torch.from_numpy(np.array(c2ws).astype(np.float32))
        train_poses, test_poses = [], []
        train_imgs_path, test_imgs_path = [], []

        # # 方法1：正确的随机打乱
        # shuffled_data = list(zip(image_names, poses))
        # random.shuffle(shuffled_data)  # 直接打乱
        # split_point = len(shuffled_data) // 2

        # for i, (name, pose) in enumerate(shuffled_data):
        #     if i < split_point:
        #         test_poses.append(pose)
        #         test_imgs_path.append(os.path.join(colmap_dir, 'images', name))
        #     else:
        #         train_poses.append(pose)
        #         train_imgs_path.append(os.path.join(colmap_dir, 'images', name))
        
        # for i, name in enumerate(image_names[1:-5]):
        #     test_poses.append(poses[i+1])
        #     test_imgs_path.append(os.path.join(colmap_dir, 'images', name))
        #     # print(f'Image {name} with {i} is in the test set')
        
        if len(image_names) > 6:
             start_idx = 1
             end_idx = -5
        else:
             start_idx = 0
             end_idx = len(image_names)

        for i, name in enumerate(image_names[start_idx:end_idx]):
            test_poses.append(poses[i+start_idx])
            test_imgs_path.append(os.path.join(colmap_dir, 'images', name))

        
        # for i, name in enumerate(image_names):
        #     if i < len(image_names) // 2:
        #         test_poses.append(poses[i])
        #         test_imgs_path.append(os.path.join(colmap_dir, 'images', name))
        #         print(f'Image {name} with {i} is in the test set')
        #     else:
        #         train_poses.append(poses[i])
        #         train_imgs_path.append(os.path.join(colmap_dir, 'images', name))
        #         print(f'Image {name} with {i} is in the train set')
        
        

        # for i, name in enumerate(image_names):
        #     if name.startswith('test'):
        #         test_poses.append(poses[i])
        #         test_imgs_path.append(os.path.join(colmap_dir, 'images', name))
        #     else:
        #         train_poses.append(poses[i])
        #         train_imgs_path.append(os.path.join(colmap_dir, 'images', name))
        
        if len(train_poses)!=0:
            meta['train_camera_to_worlds'] = torch.stack(train_poses, dim=0)
        else:
            # print("Warning: No training images, set the 1st test poses as training poses as placeholder")
            meta['train_camera_to_worlds'] = torch.stack(test_poses[:1], dim=0)
            train_imgs_path = test_imgs_path[:1]
        meta['test_camera_to_worlds'] = torch.stack(test_poses, dim=0)
        # print("meta train_imgs_path and test_imgs_path:", meta, train_imgs_path, test_imgs_path)
        return meta, train_imgs_path, test_imgs_path

    def load_scene(self, idx):
        nerfstudio_dir, colmap_dir = self.folders[idx]
        gs_params, scaler = self.load_gs_params_fromnerfstudio(nerfstudio_dir, idx)
        if self.load_pose_src == 'colmap':
            meta, train_imgs_path, test_imgs_path = self.load_images_cameras_fromcolmap(colmap_dir, nerfstudio_dir)
            # Apply World transformation (OpenCV -> OpenGL) to cameras as well
            # The load_images_cameras_fromcolmap only flips the local camera axes (columns)
            # We also need to flip the World axes (rows) to match the Gaussian transformation
            # Flip Y and Z of the world coordinate (Rows 1 and 2 of C2W)
            # meta['train_camera_to_worlds'][:, 1:3, :] *= -1
            # meta['test_camera_to_worlds'][:, 1:3, :] *= -1
        if self.load_pose_src == 'nerfstudio':
            meta, train_imgs_path, test_imgs_path = self.load_images_cameras_fromnerfstudio(nerfstudio_dir, colmap_dir)
        meta['train_camera_to_worlds'][:,:3,-1] = scaler.transform(meta['train_camera_to_worlds'][:,:3,-1])
        meta['test_camera_to_worlds'][:,:3,-1] = scaler.transform(meta['test_camera_to_worlds'][:,:3,-1])

        outputs = {'gs_params': gs_params, 'meta': meta, 'idx': idx, 
                'scene_name': nerfstudio_dir, #The basename is splatfacto
                'train_imgs_path': train_imgs_path,
                'test_imgs_path': test_imgs_path}
        return outputs

    def get_scene_from_cache(self):
        # 1. If the cache is not full, we just append the new data
        if len(self.cache_scenes) < self.cache_num_scenes:
            idx = self.remaining_scenes.pop(0)
            if self.train_or_test=='train' and len(self.remaining_scenes)==0:
                self.refresh_remaining_training()
            new_scene = self.load_scene(idx)
            if self.cache_steps!=1: #cache_steps=-1: cache forever, cache_steps>1: cache for more than one step
                self.cache_scenes.append([new_scene,1]) # Set the counter to 1, otherwise keep the cache empty
            return new_scene
        else:
            # 2. If the cache is full, 
            scene_i = random.randint(0, len(self.cache_scenes)-1) # We need not to worry about the test set, since cache_num_scenes=1
            scene  = self.cache_scenes[scene_i][0]
            self.cache_scenes[scene_i][1] += 1
            if self.cache_scenes[scene_i][1] == self.cache_steps: #cache_steps=-1 means we cache forever
                self.cache_scenes.pop(scene_i)
                self.get_scene_from_cache()
            return scene

    # 原版
    # def __iter__(self):
    #     if self.train_or_test == 'train':
    #         self.refresh_remaining_training()
    #     if len(self.remaining_scenes) < self.cache_num_scenes:
    #         print(f'Warning: The number of scenes is less than the cache_num_scenes, {len(self.remaining_scenes)} < {self.cache_num_scenes}')
    #         self.cache_num_scenes = len(self.remaining_scenes)
    #         print(f'cache_num_scenes is set to {self.cache_num_scenes}')
    #     while len(self.remaining_scenes) > 0:
    #         scene = self.get_scene_from_cache()
    #         gs_params, meta, scene_name = scene['gs_params'], scene['meta'], scene['scene_name']
    #         train_imgs_path, test_imgs_path = scene['train_imgs_path'], scene['test_imgs_path']
    #         train_imgs_name = [os.path.basename(path) for path in train_imgs_path]
    #         test_imgs_name = [os.path.basename(path) for path in test_imgs_path]          

    #         total_train_num, total_test_num = len(meta['train_camera_to_worlds']), len(meta['test_camera_to_worlds'])
    #         cameras = {}
    #         if self.train_or_test == 'train':
    #             sample_test = np.random.rand(self.image_per_scene) < self.sample_ratio_test
    #             sample_test_num = min(np.sum(sample_test), total_test_num) #previously here it's max
    #             sample_train_num = self.image_per_scene - sample_test_num
    #             sample_train_num = min(sample_train_num, total_train_num)
    #             images, images_names = [], []
    #             cameras['camera_to_worlds'] = []
    #             #decide background_color
    #             if self.background_color == 'random':
    #                 background = torch.rand(3)
    #             else:
    #                 background = torch.tensor(self.background_color)/255.
    #             if sample_train_num > 0: #TODO not enough training or test views
    #                 train_cam_ids = np.random.permutation(total_train_num)[:sample_train_num]
    #                 images.extend([self.read_image(train_imgs_path[i], background=background) for i in train_cam_ids])
    #                 images_names.extend([train_imgs_name[i] for i in train_cam_ids])
    #                 cameras['camera_to_worlds'].append(meta['train_camera_to_worlds'][train_cam_ids])
    #             if sample_test_num > 0:
    #                 # test_cam_ids = np.random.permutation(total_test_num)[:sample_test_num]
    #                 test_cam_ids = np.arange(sample_test_num)
    #                 # print(test_cam_ids)
    #                 images.extend([self.read_image(test_imgs_path[i], background=background) for i in test_cam_ids])
    #                 images_names.extend([test_imgs_name[i] for i in test_cam_ids])
    #                 cameras['camera_to_worlds'].append(meta['test_camera_to_worlds'][test_cam_ids])
    #             cameras['camera_to_worlds'] = torch.concatenate(cameras['camera_to_worlds'], axis=0)
    #         elif self.train_or_test == 'test':
    #             assert self.background_color!='random', 'For test set, background_color cannot be random'
    #             background = torch.tensor(self.background_color)/255.
    #             test_cam_ids = np.arange(total_test_num) #We take all the test images
    #             images = [self.read_image(test_imgs_path[i], background=background) for i in test_cam_ids]
    #             images_names = [test_imgs_name[i] for i in test_cam_ids]
    #             cameras['camera_to_worlds'] = meta['test_camera_to_worlds'][test_cam_ids] 
    #         else:
    #             raise ValueError
    #         for key in ['fx','fy','cx','cy','width','height']:
    #             cameras[key] = meta[key]
    #         cameras['background_color'] = background
    #         output_dict = {'gs_params': gs_params, 'images': images, 'cameras': cameras, 
    #                     'scene_idx': scene['idx'],
    #                     'scene_name': scene_name} #..../XX/splatfacto
    #         output_dict['images_name'] = images_names
    #         yield output_dict
    
    
    def __iter__(self):
        if self.train_or_test == 'train':
            self.refresh_remaining_training()
        if len(self.remaining_scenes) < self.cache_num_scenes:
            print(f'Warning: The number of scenes is less than the cache_num_scenes, {len(self.remaining_scenes)} < {self.cache_num_scenes}')
            self.cache_num_scenes = len(self.remaining_scenes)
            print(f'cache_num_scenes is set to {self.cache_num_scenes}')
        while len(self.remaining_scenes) > 0:
            scene = self.get_scene_from_cache()
            gs_params, meta, scene_name = scene['gs_params'], scene['meta'], scene['scene_name']
            train_imgs_path, test_imgs_path = scene['train_imgs_path'], scene['test_imgs_path']
            train_imgs_name = [os.path.basename(path) for path in train_imgs_path]
            test_imgs_name = [os.path.basename(path) for path in test_imgs_path]          

            total_train_num, total_test_num = len(meta['train_camera_to_worlds']), len(meta['test_camera_to_worlds'])
            cameras = {}
            if self.train_or_test == 'train':
                sample_test = np.random.rand(self.image_per_scene) < self.sample_ratio_test
                sample_test_num = min(np.sum(sample_test), total_test_num)
                sample_train_num = self.image_per_scene - sample_test_num
                sample_train_num = min(sample_train_num, total_train_num)
                images, images_names = [], []
                cameras['camera_to_worlds'] = []
                if self.background_color == 'random':
                    background = torch.rand(3)
                else:
                    background = torch.tensor(self.background_color)/255.
                if sample_train_num > 0:
                    train_cam_ids = np.random.permutation(total_train_num)[:sample_train_num]
                    images.extend([self.read_image(train_imgs_path[i], background=background) for i in train_cam_ids])
                    images_names.extend([train_imgs_name[i] for i in train_cam_ids])
                    cameras['camera_to_worlds'].append(meta['train_camera_to_worlds'][train_cam_ids])
                if sample_test_num > 0:
                    test_cam_ids = np.arange(sample_test_num)
                    images.extend([self.read_image(test_imgs_path[i], background=background) for i in test_cam_ids])
                    images_names.extend([test_imgs_name[i] for i in test_cam_ids])
                    cameras['camera_to_worlds'].append(meta['test_camera_to_worlds'][test_cam_ids])
                cameras['camera_to_worlds'] = torch.concatenate(cameras['camera_to_worlds'], axis=0)
            elif self.train_or_test == 'test':
                assert self.background_color!='random', 'For test set, background_color cannot be random'
                background = torch.tensor(self.background_color)/255.
                test_cam_ids = np.arange(total_test_num)
                images = [self.read_image(test_imgs_path[i], background=background) for i in test_cam_ids]
                images_names = [test_imgs_name[i] for i in test_cam_ids]
                cameras['camera_to_worlds'] = meta['test_camera_to_worlds'][test_cam_ids] 
            else:
                raise ValueError
            for key in ['fx','fy','cx','cy','width','height']:
                cameras[key] = meta[key]
            cameras['background_color'] = background
            output_dict = {'gs_params': gs_params, 'images': images, 'cameras': cameras, 
                        'scene_idx': scene['idx'],
                        'scene_name': scene_name}
            output_dict['images_name'] = images_names

            # 增量式解锁逻辑
            self.step_and_unlock()

            yield output_dict        
