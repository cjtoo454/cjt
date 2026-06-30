# LunarLander PPO 训练记录

训练日期：2026-06-25  
训练设备：CPU  
项目入口：按 `README.md` 使用 `train_base.py`、`train_real.py`、`evaluate_base.py`、`evaluate_real.py`

## 1. 训练前验证

先按 README 的短训练命令验证训练流程、Box2D 环境、模型保存路径是否正常。

```powershell
python train_base.py --timesteps 256 --n-envs 1 --device cpu
python train_real.py --timesteps 256 --n-envs 1 --device cpu
```

说明：由于 PPO 的 `n_steps=1024`，即使传入 `--timesteps 256`，Stable-Baselines3 也会至少收集一个 rollout，因此实际保存时显示 `total_timesteps=1024`。短训练只用于确认流程可运行，不作为模型效果依据。

## 2. 本次正式训练参数

本次正式训练没有直接修改源码配置文件，使用 README 的默认训练入口，仅显式指定 `--device cpu`。

默认 PPO 关键参数来自 `agent_ppo/conf/conf.py` 与 `agent_ppo/conf/ppo_lunarlander.yml`：

| 参数 | 值 | 思路 |
| --- | --- | --- |
| `n_timesteps` | `1_000_000` | README 默认正式训练步数，比短训练更能形成稳定策略 |
| `n_envs` | `16` | 并行采样，提高 CPU 下的数据收集效率 |
| `n_steps` | `1024` | 每次 rollout 的采样长度，兼顾稳定估计和训练速度 |
| `batch_size` | `64` | 保持 PPO 默认小批量更新，降低显存/内存压力 |
| `n_epochs` | `4` | 每批数据重复更新 4 次，避免过度拟合单批 rollout |
| `learning_rate` | `0.0003` | 采用 Stable-Baselines3/RL Zoo 常用 PPO 学习率 |
| `gamma` | `0.999` | LunarLander 回合较长，提高远期回报权重 |
| `gae_lambda` | `0.98` | 平衡优势估计的偏差和方差 |
| `clip_range` | `0.2` | PPO 标准裁剪范围，限制单次策略更新幅度 |
| `ent_coef` | `0.01` | 保留探索，避免过早收敛到单一动作模式 |
| 网络结构 | `pi=[64,64]`, `vf=[64,64]` | 任务状态维度较低，轻量 MLP 足够且训练快 |

训练命令：

```powershell
python train_base.py --device cpu
python train_real.py --device cpu
```

## 3. 训练结果

### Base 环境

保存路径：

```text
logs/ppo/LunarLander-v3_2/LunarLander-v3.zip
logs/ppo/LunarLander-v3_2/best_model.zip
```

独立评估命令：

```powershell
python evaluate_base.py --episodes 20 --device cpu
python evaluate_base.py --episodes 20 --best --device cpu
```

评估结果：

| 模型 | mean_reward | solved_rate | mean_length |
| --- | ---: | ---: | ---: |
| final | `256.213 +/- 23.401` | `1.000` | `361.4` |
| best | `257.602 +/- 23.889` | `1.000` | `354.4` |

结论：base 环境已达到 README 中 `episode_return >= 200` 的常用通过标准。

### Real 环境

保存路径：

```text
logs/ppo/LunarLander-v3-real_2/LunarLander-v3-real.zip
logs/ppo/LunarLander-v3-real_2/best_model.zip
```

独立评估命令：

```powershell
python evaluate_real.py --episodes 20 --device cpu
python evaluate_real.py --episodes 20 --best --device cpu
```

评估结果：

| 模型 | mean_reward | solved_rate | mean_final_score | completion_rate |
| --- | ---: | ---: | ---: | ---: |
| final | `144.060 +/- 107.852` | `0.350` | `34.207` | `0.400` |
| best | `192.979 +/- 104.413` | `0.700` | `63.406` | `0.800` |

结论：real 环境包含观测噪声、控制延迟和随机阵风，训练难度高于 base。本次训练应优先使用 `best_model.zip`，因为最终 checkpoint 在最后阶段出现回报回落，而 best checkpoint 的完成率和最终评分更高。

## 4. 参数修改思路

本次保留 README 默认参数，原因是 base 环境已经稳定过线，real 环境也能学到可用策略；后续若继续优化，建议按下面顺序小步调整。

1. 先延长 real 训练步数
   - 建议：`python train_real.py --timesteps 2000000 --device cpu`
   - 理由：real 环境随机性更强，100 万步后仍有较大方差，延长训练通常比先改复杂结构更稳。

2. 评估时优先比较 `best_model.zip`
   - 建议：固定使用 `python evaluate_real.py --episodes 50 --best --device cpu`
   - 理由：real 训练后期可能回落，使用最佳评估 checkpoint 更符合实际部署选择。

3. 如果 real 环境仍不稳定，再降低学习率
   - 可尝试：`learning_rate = 0.0002` 或 `0.0001`
   - 理由：噪声、延迟和阵风会增大更新方差，较小学习率能减少策略后期震荡，但训练会更慢。

4. 如果落地前动作抖动明显，可降低探索强度
   - 可尝试：`ent_coef = 0.005`
   - 理由：训练前期需要探索，后期过强探索可能导致着陆动作不够稳定。建议只在训练更长后再降低。

5. 如果 value loss 波动大，可增大网络容量
   - 可尝试：`pi=[128,128]`, `vf=[128,128]`
   - 理由：real 环境动态更复杂，稍大的 actor/critic 可能提升拟合能力。代价是训练速度下降，应在延长训练仍不足时再试。

6. 如需专门提升 final_score，可针对 real 评分项做奖励 shaping
   - 方向：提高姿态稳定、水平速度、落点精度相关奖励权重。
   - 理由：当前 real 的 `mean_weighted_score` 较高，但 `completion_rate` 决定最终分数上限，后续应优先提升稳定完成率。

## 5. 推荐使用的模型

base 环境：

```text
logs/ppo/LunarLander-v3_2/best_model.zip
```

real 环境：

```text
logs/ppo/LunarLander-v3-real_2/best_model.zip
```

