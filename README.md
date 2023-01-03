# PASTA - Planning with Spatial-Temporal Abstraction from Point Clouds for Deformable Object Manipulation

### Prerequsite
1. Install conda environments according to `environment.yml`, and then run `conda activate plb`.
2. Install [torch (1.9.0) with cudatoolkit (10.2)](https://pytorch.org/get-started/previous-versions/)
3. Install [pykeops (1.5)](https://www.kernel-operations.io/keops/python/installation.html); make sure version==1.5
3. Install [geomloss](https://www.kernel-operations.io/geomloss/api/install.html)
4. Install [PointFlow](https://github.com/stevenygd/PointFlow)
5. Run `./prepare.sh`
6. Download initial and target configurations of environments from [[Google Drive link for datasets (3G)]](https://drive.google.com/drive/folders/1ckOkxsuqK44Ay0e1I5EKmX3cOATB4Jam?usp=share_link)
7. (Optional) Download demonstration trajectories from [[Google Drive link for demonstration trajectories (16G)]](https://drive.google.com/drive/folders/1uzFKI5rehp2VMYc5MKyE-CPbSoEcKCup?usp=share_link)
8. (Optional) Download pretrained models from [[Google Drive link for pre-trained models (300M)]](https://drive.google.com/drive/folders/18tmH0stc1z_TzfAHbQDu5HASNkaWFKk_?usp=share_link)

### Environments
We currently include three tasks from the paper:
| LiftSpread  | CutRearrange | CutRearrangeSpread (CRS) |
| :---: | :---: | :---: |
| <img src="media/LiftSpread-v1_PASTA.gif" width="200">  | <img src="media/CutRearrange-v1_PASTA.gif" width="200">  | <img src="media/CutRearrangeSpread-v1_PASTA.gif" width="200">  |

* **LiftSpread**: The agent needs to first use a spatula (modeled as a thin surface) to lift a dough onto the cutting board and then adopt a rolling pin to roll over the dough to flatten it. The rolling pin is simulated as a 3-Dof capsule that can rotate along the long axis and the vertical axis and translate along the vertical axis to press the dough.
* **CutRearrange**:  This is a three-step task. Given an initial pile of dough, the agent needs to first cut the dough in half using a knife. Inspired by the recent cutting simulation (Heiden et al., 2021), we model the knife using a thin surface as the body and a prism as the blade. Next, the agent needs to use the gripper to transport each piece of the cut dough to target locations.
* **CutRearrangeSpread (CRS)** This task provides a number of demonstration trajectories performing one of the three skills: Cutting with a knife, pushing with a pusher, and spreading with a roller. The demonstration of each skill only shows a tool manipulating a single piece of dough.

### Training Procedures
The following procedures describe a pipeline for training PASTA. Note that many of the steps can be easily achieved by downloading datasets/models from the Google Drive links. If you want to train PASTA from scratch, you can follow the pipeline step-by-step.
1. For every environment, one needs to pre-generate initial and target configurations. An example run script is `run_scripts/generate_init_target.sh`. To download pre-generated configurations of the 3 environments above, please go to **step 6 of Prerequsite**.
2. To generate demonstration trajectories, one needs to run gradient-based trajectory optimization (GBTO). An example run script is `run_scripts/run_gbto.sh`. To download pre-generated demonstration trajectories of the 3 environments above, please go to **step 7 of Prerequsite**.
One also needs to post-process the demonstration data by running DBSCAN, if they are running GBTO from scratch.(see `run_scripts/run_dbscan.sh`) However, we already did this for you for the pre-generated demonstration data, so you don't need to do this again.
3. To actually train PASTA, one first needs to train a [PointFlow](https://github.com/stevenygd/PointFlow) VAE. Example run scripts are `PointFlow/scripts/set_<env_name>_gen_dist.sh`. To download pre-generated PointFlow models of the 3 environments above, please go to **step 8 of Prerequsite**.
4. Next, one needs to train PASTA's policy networks, which are parameterized by [PointNet++](https://github.com/pyg-team/pytorch_geometric). An example run script is `run_scripts/pasta_train_policy.sh` To download pre-generated policy models, please go to **step 8 of Prerequsite**.
5. Finally, one needs to train PASTA's abstraction modules (feasibility and cost predictors). Additionally, one needs to load the policy and VAE to evaluate the full model's performance. An example run script is `run_scripts/pasta_train_abstraction.sh`
6. To evaluate pretrained PASTA models, one can download the models from the Google Drive and run `run_scripts/pasta_plan.sh`


Bon Appetit!

<img src="media/pasta.jpeg" width="200">



## Cite
If you find this codebase useful in your research, please consider citing:
```
@inproceedings{lin2022planning,
title={Planning with Spatial-Temporal Abstraction from Point Clouds for Deformable Object Manipulation},
author={Xingyu Lin and Carl Qi and Yunchu Zhang and Zhiao Huang and Katerina Fragkiadaki and Yunzhu Li and Chuang Gan and David Held},
booktitle={6th Annual Conference on Robot Learning},
year={2022},
url={https://openreview.net/forum?id=tyxyBj2w4vw}
}
```
