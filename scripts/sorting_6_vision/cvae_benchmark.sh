MUJOCO_GL=egl python run_vision.py --config-name=sorting_6_vision_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=cvae_vision_agent \
              agent_name=cvae_vision \
              window_size=1 \
              group=sorting_6_cvae_seeds \
              agents.model.model.encoder.latent_dim=32 \
              agents.kl_loss_factor=67.46378648811798