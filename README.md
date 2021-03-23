
# Efficient Robotic Object Search via HIEM: Hierarchical Policy Learning with Intrinsic-Extrinsic Modeling

<img src="https://github.com/Xin-Ye-1/HIEM/blob/main/overview.png" width = "400" height = "400" align=center />

This is the source code for our HIEM framework and the baseline methods we mentioned in the paper. 

[paper](https://arxiv.org/pdf/2010.08596.pdf) | [video](https://www.youtube.com/watch?v=rAHB3jIS3Wo)

## Requirements

Our code is developed and tested under the following dependencies:

- python==2.7.15
- scipy==1.2.0
- numpy==1.15.4
- tensorflow==1.6.0
- tf Slim 
- opencv==3.2.0-dev

Before running the code, please specify the path to the code directory in the `config.json`.

Download [our pre-processed data](https://drive.google.com/file/d/1enOKLbfm2cGWT8GOOW59Q8y_CB6L4nR0/view?usp=sharing) sourced from [House3D](https://github.com/facebookresearch/house3d) and extract here for our robotic object search task.

Download [our pre-trained models](https://drive.google.com/file/d/1uP-3JFy8fRnp9qGt6ZE5E-WUnCd79YpW/view?usp=sharing) and put them in the corresponding code directories for training and/or evaluating our method.

## Training

To train our model `HIEM` in the paper, run this command:

```bash
# Specify the parameters in HIEM/train.sh, 
# and from HIEM/
./train.sh
```
or

```bash
# From HIEM/
python train.py \
    --load_model=True \
    --default_scenes=<enviroments_to_train> \
    --default_targets=<target_objects_to_train> \
    --pretrained_model_path=${PATH_TO_PRETRAINED_MODEL} \
    --model_path=${PATH_TO_MODEL} 

```

where the `pretrained_model_path` is `../h-DQN/result*_mt_for_pretrain/model` and the environments and their target objects for training can be found in `readme`.

To train other baseline methods mentioned in the paper, run the same command from the corresponding directories.



## Evaluation and Results

To evaluate our method `HIEM` for the robotic object search task on House3D, 

- run the command,
```bash
# From HIEM/
CUDA_VISIBLE_DEVICES=-1 python evaluate.py \
  --max_episodes=1 \
  --load_model=True \
  --model_path="result1_mt_pretrain/model" \
  --evaluate_file='../random_method/1s6t_1.txt' \
  --default_scenes='5cf0e1e9493994e483e985c436b9d3bc' \
  --default_targets='music' \
  --default_targets='television' \
  --default_targets='table' \
  --default_targets='stand' \
  --default_targets='dressing_table' \
  --default_targets='heater' \ 
```
to reproduce the results of our method on the environment `1` as follows,

|  Method  |   SR   |     AS / MS     |   SPL  |   AR   |
| :-------:|:------:|:---------------:|:------:|:------:|
|   HIEM   |  1.00  |  41.18 / 25.63  |  0.72  |  0.70  |


To evaluate other environments and target objects, change `model_path`, `evaluate_file`, `default_scenes` and `default_targets` accordingly.

To evaluate other baseline methods, run the same command from the corresponding directories.