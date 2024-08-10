import os
import logging

import hydra
import numpy as np

import wandb
from omegaconf import DictConfig, OmegaConf
import torch
from agents.utils.sim_path import sim_framework_path


log = logging.getLogger(__name__)


OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
torch.cuda.empty_cache()


from agents.utils.hdf5_to_img import read_img_from_hdf5
import torch
import matplotlib.pyplot as plt


def test_agent_on_train_data(path, agent):
    imgs = read_img_from_hdf5(path, 0, -1, to_tensor=False)
    joint_poses = torch.load(path + "/follower_joint_pos.pt")
    state_pairs = list(zip(imgs[0], imgs[1], joint_poses))

    num_action = len(joint_poses) - 1
    pred_joint_poses = np.zeros((num_action, 7))

    for i in range(num_action):
        state_pair = state_pairs[i]
        obs = (state_pair[0], state_pair[1])
        pred_action = agent.predict(obs, if_vision=True).squeeze()
        pred_joint_pos = pred_action[:7]
        pred_gripper_command = pred_action[-1]
        pred_joint_poses[i] = pred_joint_pos
    for i in range(7):
        fig, ax = plt.subplots()
        ax.plot(
            range(num_action), pred_joint_poses[::, i], label="prediction"
        )  # Plot some data on the Axes.
        ax.plot(range(num_action), joint_poses[:-1, i], label="truth")
        ax.legend()
        plt.show()
        plt.waitforbuttonpress()


@hydra.main(config_path="configs", config_name="real_robot_config.yaml")
def main(cfg: DictConfig) -> None:

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode="disabled",
        config=wandb.config,
    )

    cfg.task_suite = "cupStacking"
    cfg.if_sim = True
    agent = hydra.utils.instantiate(cfg.agents)
    agent.load_pretrained_model(
        "/home/alr_admin/david/praktikum/d3il_david/weights",
        sv_name="ddpm_100data_100epoch.pth",
    )

    env_sim = hydra.utils.instantiate(cfg.simulation)
    env_sim.test_agent(agent)

    path = "/media/alr_admin/ECB69036B69002EE/Data_less_obs_new_hdf5/cupStacking/2024_07_25-14_25_28"
    # test_agent_on_train_data(
    #     path,
    #     agent,
    # )

    log.info("done")

    wandb.finish()


if __name__ == "__main__":
    main()
