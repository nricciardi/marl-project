from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback

from common.cli import CommonTrainingArgs

def initialize_base_tuner(args: CommonTrainingArgs, param_space: dict, algo_class: type | None) -> tune.Tuner:
    param_space["lr"] = tune.grid_search(args.lr)
    param_space["gamma"] = tune.grid_search(args.gamma)
    param_space["entropy_coeff"] = tune.grid_search(args.entropy_coeff)
    param_space["train_batch_size"] = tune.grid_search(args.training_batch_size)
    param_space["num_epochs"] = tune.grid_search(args.epochs)
    param_space["minibatch_size"] = tune.grid_search(args.minibatch_size)

    tuner = tune.Tuner(
        algo_class,
        param_space=param_space,
        run_config=tune.RunConfig(
            storage_path=args.checkpoint_dir,
            stop={
                "training_iteration": args.iters
            },
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=args.save_interval,
                checkpoint_at_end=True,
            ),
            callbacks=[
                # WandbLoggerCallback(
                #     project=str(algo_class),
                #     group=f"Scenario_{args.scenario}_{args.mode}",
                #     log_config=True,
                #     save_code=True,
                # )
            ]
        )

    )

    return tuner