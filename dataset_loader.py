import numpy as np
import random
import torch
from pathlib import Path
import torch.utils.data as data
import utils.utils_video as utils_video
from basicsr.data import degradations as degradations
import math
import random
import cv2

def add_jpg_compression(img, quality=90):
    """Add JPG compression artifacts.
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        quality (float): JPG compression quality. 0 for lowest quality, 100 for
            best quality. Default: 90.
    Returns:
        (Numpy array): Returned image after JPG, shape (h, w, c), range[0, 1],
            float32.
    """
    img = np.clip(img, 0, 1)
    # print(quality)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    _, encimg = cv2.imencode('.jpg', img * 255., encode_param)
    img = np.float32(cv2.imdecode(encimg, 1)) / 255.
    return img


def random_add_jpg_compression(img, quality_range=(90, 100)):
    """Randomly add JPG compression artifacts.
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        quality_range (tuple[float] | list[float]): JPG compression quality
            range. 0 for lowest quality, 100 for best quality.
            Default: (90, 100).
    Returns:
        (Numpy array): Returned image after JPG, shape (h, w, c), range[0, 1],
            float32.
    """
    quality = np.random.uniform(quality_range[0], quality_range[1])
    return add_jpg_compression(img, quality)

def random_mixed_kernels(kernel_list,
                         kernel_prob,
                         kernel_size=21,
                         sigma_x_range=(0.6, 5),
                         sigma_y_range=(0.6, 5),
                         rotation_range=(-math.pi, math.pi),
                         betag_range=(0.5, 8),
                         betap_range=(0.5, 8),
                         rand_num =1,
                         noise_range=None):
    """Randomly generate mixed kernels.
    Args:
        kernel_list (tuple): a list name of kernel types,
            support ['iso', 'aniso', 'skew', 'generalized', 'plateau_iso',
            'plateau_aniso']
        kernel_prob (tuple): corresponding kernel probability for each
            kernel type
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        beta_range (tuple): [0.5, 8]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None
    Returns:
        kernel (ndarray):
    """
    x = sigma_x_range[0] + (sigma_x_range[1]-sigma_x_range[0])*rand_num 
    sigma_x_range = (x, x+0.00001)
    y = sigma_y_range[0] + (sigma_y_range[1]-sigma_y_range[0])*rand_num 
    sigma_y_range = (y, y+0.00001)
    r = rotation_range[0] + (rotation_range[1]-rotation_range[0])*rand_num 
    rotation_range = (r, r+0.00001)

    kernel_type = 'aniso'
    if kernel_type == 'iso':
        kernel = degradations.random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=noise_range, isotropic=True)
    elif kernel_type == 'aniso':
        kernel = degradations.random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=noise_range, isotropic=False)
    elif kernel_type == 'generalized_iso':
        kernel = random_bivariate_generalized_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            betag_range,
            noise_range=noise_range,
            isotropic=True)
    elif kernel_type == 'generalized_aniso':
        kernel = random_bivariate_generalized_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            betag_range,
            noise_range=noise_range,
            isotropic=False)
    elif kernel_type == 'plateau_iso':
        kernel = random_bivariate_plateau(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, betap_range, noise_range=None, isotropic=True)
    elif kernel_type == 'plateau_aniso':
        kernel = random_bivariate_plateau(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, betap_range, noise_range=None, isotropic=False)
    return kernel

def random_bivariate_Gaussian(kernel_size,
                              sigma_x_range,
                              sigma_y_range,
                              rotation_range,
                              noise_range=None,
                              isotropic=True):
    """Randomly generate bivariate isotropic or anisotropic Gaussian kernels.
    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.
    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None
    Returns:
        kernel (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    kernel = bivariate_Gaussian(kernel_size, sigma_x, sigma_y, rotation, isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    return kernel


class PortraitVideoRecurrentTrainDataset(data.Dataset):


    def __init__(self, opt):
        super(PortraitVideoRecurrentTrainDataset, self).__init__()
        self.opt = opt
        self.scale = opt.get('scale', 4)
        self.gt_size = opt.get('gt_size', 256)
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.filename_tmpl = opt.get('filename_tmpl', '08d')
        self.filename_ext = opt.get('filename_ext', 'png')
        self.num_frame = opt['num_frame']

        keys = []
        total_num_frames = [] 
        start_frames = [] 
        rand_num = []
        with open(opt['meta_info_file'], 'r') as fin:
            
            for line in fin:
                tmp = random.uniform(0, 1)
                folder, frame_num, _, start_frame = line.split(' ')
                keys.extend([f'{folder}/{i:{self.filename_tmpl}}' for i in range(int(start_frame), int(start_frame)+int(frame_num))])
                total_num_frames.extend([int(frame_num) for i in range(int(frame_num))])
                start_frames.extend([int(start_frame) for i in range(int(frame_num))])
                rand_num.extend([tmp for i in range(int(frame_num))])


        self.keys = []
        self.total_num_frames = [] # some clips may not have 100 frames
        self.start_frames = []
        self.rand_num = []
        if opt['test_mode']:
            for i, v in zip(range(len(keys)), keys):
                if v.split('/')[0] in val_partition:
                    self.keys.append(keys[i])
                    self.total_num_frames.append(total_num_frames[i])
                    self.start_frames.append(start_frames[i])
                    self.rand_num.append(rand_num[i])
        else:
            for i, v in zip(range(len(keys)), keys):
                if v.split('/')[0] not in val_partition:
                    self.keys.append(keys[i])
                    self.total_num_frames.append(total_num_frames[i])
                    self.start_frames.append(start_frames[i])
                    self.rand_num.append(rand_num[i])

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if hasattr(self, 'flow_root') and self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)

        self.blur_kernel_size = 15
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = (0.1,3)
        self.downsample_range = (0.8,2.5)
        self.noise_range = (0,0.1)
        self.jpeg_range = (70,100)

        interval_str = ','.join(str(x) for x in self.interval_list)
        print(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        # pdb.set_trace()
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index]
        total_num_frames = self.total_num_frames[index]
        start_frames = self.start_frames[index]
        random_num = self.rand_num[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        endmost_start_frame_idx = start_frames + total_num_frames - self.num_frame * interval
        if start_frame_idx > endmost_start_frame_idx:
            start_frame_idx = random.randint(start_frames, endmost_start_frame_idx)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))


        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []



        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:{self.filename_tmpl}}'
                img_gt_path = f'{clip_name}/{neighbor:{self.filename_tmpl}}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = utils_video.imfrombytes(img_bytes, float32=True)
            

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = utils_video.imfrombytes(img_bytes, float32=True)
            h, w, _ = img_gt.shape
            img_gts.append(img_gt)

            # ------------------------ generate lq image ------------------------ #
            # blur
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                random_num,
                noise_range=None)
            img_lq = cv2.filter2D(img_gt, -1, kernel)
            # downsample
            scale = self.downsample_range[0] +  (self.downsample_range[1]-self.downsample_range[0])*random_num
            jpeg_rand = self.jpeg_range[0] +  (self.jpeg_range[1]-self.jpeg_range[0])*random_num

            img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
            
            if self.jpeg_range is not None:
                img_lq = add_jpg_compression(img_lq, jpeg_rand)

            # resize to original size
            
            img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
            img_lqs.append(img_lq)

        # randomly crop
        # img_gts, img_lqs = utils_video.paired_random_crop(img_gts, img_lqs, self.gt_size, self.scale, img_gt_path)


        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = utils_video.augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = utils_video.img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'L': img_lqs, 'H': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)


    """Vimeo90K dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains:
    1. clip name; 2. frame number; 3. image shape, separated by a white space.
    Examples:
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    Key examples: "00001/0001"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    The neighboring frame list for different num_frame:
    num_frame | frame list
             1 | 4
             3 | 3,4,5
             5 | 2,3,4,5,6
             7 | 1,2,3,4,5,6,7

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(VideoRecurrentTrainVimeoDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])

        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # indices of input images
        self.neighbor_list = [i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])]

        # temporal augmentation configs
        self.random_reverse = opt['random_reverse']
        print(f'Random reverse is {self.random_reverse}.')

        self.flip_sequence = opt.get('flip_sequence', False)
        self.pad_sequence = opt.get('pad_sequence', False)
        self.neighbor_list = [1, 2, 3, 4, 5, 6, 7]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the neighboring LQ and  GT frames
        img_lqs = []
        img_gts = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip}/{seq}/im{neighbor}'
                img_gt_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path = self.lq_root / clip / seq / f'im{neighbor}.png'
                img_gt_path = self.gt_root / clip / seq / f'im{neighbor}.png'
            # LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = utils_video.imfrombytes(img_bytes, float32=True)
            # GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = utils_video.imfrombytes(img_bytes, float32=True)

            img_lqs.append(img_lq)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = utils_video.paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = utils_video.augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = utils_video.img2tensor(img_results)
        img_lqs = torch.stack(img_results[:7], dim=0)
        img_gts = torch.stack(img_results[7:], dim=0)

        if self.flip_sequence:  # flip the sequence: 7 frames to 14 frames
            img_lqs = torch.cat([img_lqs, img_lqs.flip(0)], dim=0)
            img_gts = torch.cat([img_gts, img_gts.flip(0)], dim=0)
        elif self.pad_sequence:  # pad the sequence: 7 frames to 8 frames
            img_lqs = torch.cat([img_lqs, img_lqs[-1:,...]], dim=0)
            img_gts = torch.cat([img_gts, img_gts[-1:,...]], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gt: (c, h, w)
        # key: str
        return {'L': img_lqs, 'H': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)