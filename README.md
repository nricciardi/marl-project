# MARL Project

## Environments

Cooperative Pong: https://pettingzoo.farama.org/environments/butterfly/cooperative_pong/

MPE2: https://mpe2.farama.org/

VMAS: https://github.com/proroklab/VectorizedMultiAgentSimulator?tab=readme-ov-file#main-scenarios

Multiwalker: https://pettingzoo.farama.org/environments/sisl/multiwalker/



## WanDB

```
docker run --rm -v ./wandb:/vol -d -p 8080:8080 --name wandb-local wandb/local
```

```
pip install wandb
```

```
wandb login --host=http://localhost:8080
```

