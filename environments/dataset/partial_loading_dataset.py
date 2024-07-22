import logging
from environments.dataset.base_dataset import TrajectoryDataset
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import h5py, cv2
import psutil


def img_file_key(p: Path):
    return int(p.name.partition(".")[0])


class Partial_Loading_Dataset(TrajectoryDataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        task_suite: str = "cupStacking",
        device="cpu",
        obs_dim: int = 20,
        action_dim: int = 2,
        max_len_data: int = 256,
        window_size: int = 1,
        if_sim: bool = False,
        cam_0_w=256,
        cam_0_h=256,
        cam_1_w=256,
        cam_1_h=256,
        cam_num=2,
        to_tensor=True,
    ):

        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size,
        )

        logging.info("Loading Real Robot Dataset")

        actions = []
        masks = []

        if task_suite == "cupStacking":
            data_dir = Path(data_directory + "/cupstacking")
        elif task_suite == "pickPlacing":
            data_dir = Path(data_directory + "/banana")
        elif task_suite == "insertion":
            data_dir = Path(data_directory + "/insertion")
        else:
            raise ValueError("Wrong name of task suite.")

        if not if_sim:
            load_img = -1
        else:
            load_img = 1

        self.cam_0_resize = (cam_0_w, cam_0_h)
        self.cam_1_resize = (cam_1_w, cam_1_h)
        self.cams_resize = [self.cam_0_resize, self.cam_1_resize]
        self.traj_dirs = sorted(list(data_dir.iterdir()))
        self.to_tensor = to_tensor

        self.loaded_traj_index = []
        self.traj_use_count = np.zeros(len(self.traj_dirs))

        self.imgs = []

        for i, traj_dir in enumerate(tqdm(self.traj_dirs)):
            zero_action = torch.zeros(
                (1, self.max_len_data, self.action_dim), dtype=torch.float32
            )
            zero_mask = torch.zeros((1, self.max_len_data), dtype=torch.float32)

            joint_pos = torch.load(traj_dir / "follower_joint_pos.pt")
            # joint_vel = torch.load(traj_dir / "joint_vel.pt")
            gripper_command = torch.load(traj_dir / "follower_gripper_state.pt")

            valid_len = len(joint_pos) - 1

            # zero_action[0, :valid_len, :] = torch.cat(
            #     [joint_pos, joint_vel, gripper_command[:, None]], dim=1
            # )

            zero_action[0, :valid_len, :] = torch.cat(
                [joint_pos[1:], gripper_command[1:, None]], dim=1
            )

            zero_mask[0, :valid_len] = 1
            actions.append(zero_action)
            masks.append(zero_mask)

        for i, traj_dir in enumerate(tqdm(self.traj_dirs)):
            image_path = traj_dir / "images"
            image_hdf5 = traj_dir / "imgs.hdf5"
            if Path(image_path).is_dir():
                pass
            elif Path(image_hdf5).exists():
                self.read_img_from_hdf5(
                    traj_index=i,
                    start=0,
                    end=-1,
                    cam_resizes=self.cams_resize,
                    device=self.device,
                    to_tensor=to_tensor,
                    preemptive=False,
                )

        # self.cams_imgs_index = cams_img_index
        self.actions = torch.cat(actions).to(device).float()
        self.masks = torch.cat(masks).to(device).float()

        self.num_data = len(self.actions) - 1

        self.slices = self.get_slices()

    def get_slices(self):
        slices = []

        min_seq_length = np.inf
        for i in range(self.num_data):
            T = self.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)

            if T - self.window_size < 0:
                print(
                    f"Ignored short sequence #{i}: len={T}, window={self.window_size}"
                )
            else:
                slices += [
                    (i, start, start + self.window_size)
                    for start in range(T - self.window_size + 1)
                ]  # slice indices follow convention [start, end)

        return slices

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):

        i, start, end = self.slices[idx]
        traj_dir = self.traj_dirs[i]

        act = self.actions[i, start:end]
        mask = self.masks[i, start:end]

        for list_index, traj_index in enumerate(self.loaded_traj_index):
            if traj_index == i:
                cam_0 = self.imgs[list_index][0][start:end]
                cam_1 = self.imgs[list_index][1][start:end]
                self.traj_use_count[i] += 1
                return cam_0, cam_1, act, mask

        cams_imgs = self.read_img_from_hdf5(
            traj_index=i,
            start=start,
            end=end,
            cam_resizes=self.cams_resize,
            device=self.device,
            to_tensor=self.to_tensor,
        )
        cam_0 = cams_imgs[0]
        cam_1 = cams_imgs[1]

        return cam_0, cam_1, act, mask

    def read_img_from_hdf5(
        self,
        traj_index,
        start,
        end,
        cam_resizes=[(256, 256), (256, 256)],
        device="cuda",
        to_tensor=True,
        preemptive=False,
    ):
        path = self.traj_dirs[traj_index]
        ava_mem = psutil.virtual_memory().available
        needed_mem = os.path.getsize(path)
        if ava_mem < needed_mem and not preemptive:
            print("not enough memory, only loading the index")
            return []
        elif ava_mem < needed_mem and preemptive:
            index_most_freq_used_traj = np.argmax(self.traj_use_count)
            del self.loaded_traj_index[index_most_freq_used_traj]
            del self.imgs[index_most_freq_used_traj]

        self.loaded_traj_index.append(traj_index)

        f = h5py.File(os.path.join(path, "imgs.hdf5"), "r")
        cams = []
        for i, cam in enumerate(list(f.keys())):
            arr = f[cam][start:end]
            imgs = []
            for img in arr:
                nparr = cv2.imdecode(img, 1)
                processed = self.preprocess_img_for_training(
                    img=nparr, resize=cam_resizes[i], device=device, to_tensor=to_tensor
                )

                imgs.append(processed)
            imgs = torch.concatenate(imgs, dim=0)
            cams.append(imgs)
        f.close()

        self.imgs.append(cams)
        return cams

    def preprocess_img_for_training(
        self, img, resize=(256, 256), device="cuda", to_tensor=True
    ):

        if not img.shape == resize:
            img = cv2.resize(img, resize)

        img = img.transpose((2, 0, 1)) / 255.0

        if to_tensor:
            img = torch.from_numpy(img).to(device).float().unsqueeze(0)

        return img
