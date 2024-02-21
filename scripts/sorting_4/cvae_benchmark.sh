python run.py --config-name=sorting_4_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=cvae_agent \
              agent_name=cvae \
              window_size=1 \
              group=sorting_4_cvae_seeds \
              simulation.n_cores=60 \
              simulation.n_contexts=60 \
              simulation.n_trajectories_per_context=18 \
              agents.model.encoder.latent_dim=16 \
              agents.kl_loss_factor=67.46378648811798