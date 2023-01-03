from core.utils.visualization_utils import visualize_all_traj_his, visualize_all_traj, visualize_adam_pc, visualize_traj_actions
from core.eval.sampler import sample_traj_agent, sample_traj_setagent
from core.utils.diffskill_utils import visualize_trajs, aggregate_traj_info, make_grid, img_to_tensor, img_to_np
from core.utils.open3d_utils import visualize_point_cloud_plt
from core.utils.plb_utils import save_numpy_as_gif, save_rgb
from core.eval.hardcoded_eval_trajs import get_eval_skill_trajs, get_eval_traj
from core.utils import logger
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
import os
from core.traj_opt.env_spec import get_threshold
from core.utils.diffskill_utils import dict_add_prefix


def eval_skills(args, env, agent, epoch, tids=None):
    """ Run each skill on evaluation configurations; Save videos;
    Return raw trajs indexed by tid, time_step; Return aggregated info"""
    skill_traj, skill_info = [], {}
    tids = list(range(args.num_tools)) if tids is None else tids
    for tid in tids:
        trajs = []
        init_vs, target_vs = get_eval_skill_trajs(args.cached_state_path, tid)
        for init_v, target_v in tqdm(zip(init_vs, target_vs), desc="eval skills"):
            reset_key = {'init_v': init_v, 'target_v': target_v}
            reset_primitive = args.env_name != 'CutRearrangeSpread-v1'
            if args.train_set:
                traj = sample_traj_setagent(env, agent, reset_key, tid, log_succ_score=False, reset_primitive=reset_primitive)
            else:
                traj = sample_traj_agent(env, agent, reset_key, tid, log_succ_score=True, reset_primitive=reset_primitive)
            trajs.append(traj)
        visualize_traj_actions(trajs, save_name=osp.join(logger.get_dir(), f"eval_skill_actions_{epoch}_{tid}.png"))
        if args.train_modules == ['policy'] or args.train_set:
            keys = ['info_normalized_performance']
        else:
            keys = ['info_normalized_performance', 'succs', 'scores', 'score_error']

        fig, axes = plt.subplots(1, len(keys), figsize=(len(keys) * 5, 5))
        if len(keys) == 1:
            axes = [axes]
        for key_id, key in enumerate(keys):
            # Visualize traj
            if key == 'info_normalized_performance':
                visualize_trajs(trajs, key=key, ncol=10, save_name=osp.join(logger.get_dir(), f"eval_skill_inp_epoch_{epoch}_{tid}.gif"),
                                vis_target=True)
            elif key != 'score_error':
                visualize_trajs(trajs, key=key, ncol=10, save_name=osp.join(logger.get_dir(), f"eval_skill_{key}_epoch_{epoch}_{tid}.gif"),
                                vis_target=True)
            # Visualize stats
            for traj_id, traj in enumerate(trajs):
                if key == 'score_error':
                    vals = np.abs(traj['info_emds'] + traj['scores'][:len(traj['info_emds'])])
                else:
                    vals = traj[key]
                axes[key_id].plot(range(len(vals)), vals, label=f'traj_{traj_id}')
            axes[key_id].set_title(key)
        plt.legend()
        plt.tight_layout()
        plt.savefig(osp.join(logger.get_dir(), f"eval_skill_stats_epoch_{epoch}_{tid}.png"))

        info = aggregate_traj_info(trajs)
        skill_info.update(**dict_add_prefix(info, f'eval/skill_{tid}/'))
        skill_traj.append(trajs)

    return skill_traj, skill_info


def eval_vae(args, buffer, agent, epoch, all_obses=None, skill_traj=None):
    """ Either provide the skill_traj, or all_obses"""
    if all_obses is None:
        all_obses = []
        for tid in range(args.num_tools):
            for i in range(len(skill_traj[tid])):
                for j in range(len(skill_traj[tid][i]['obses'])):
                    all_obses.append(skill_traj[tid][i]['obses'][j])
        all_obses = np.array(all_obses)

    N = len(all_obses)
    sample_idx = np.random.randint(0, N, 8)
    obses = img_to_tensor(all_obses[sample_idx], mode=args.img_mode).to(agent.device)
    reconstr_obses, _, _ = agent.vae.reconstr(obses)
    reconstr_obses = img_to_np(reconstr_obses)
    imgs = np.concatenate([all_obses[sample_idx], reconstr_obses], axis=2)
    mse = np.mean(np.square(all_obses[sample_idx] - reconstr_obses))
    img = make_grid(imgs, ncol=4, padding=3)
    save_rgb(osp.join(logger.get_dir(), f'vae_reconstr_{epoch}.png'), img[:, :, :3] * 255.)

    # Plot VAE dist and visualization of the first two dimension
    N = buffer.cur_size
    obs_idx = np.random.choice(N, 1000).astype(np.int)
    obs = img_to_tensor(buffer.buffer['obses'][obs_idx], args.img_mode).cuda()
    z, z_mu, _ = agent.vae.encode(obs)
    z = z.detach().cpu().numpy()
    target_imgs = buffer.np_target_imgs
    goal_idx = np.array([0] * 500 + [1] * 500)
    goals = img_to_tensor(target_imgs[goal_idx], args.img_mode).cuda()
    z_goal, z_goal_mu, _ = agent.vae.encode(goals)
    z_goal = z_goal.detach().cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.scatter(z[:, 0], z[:, 1], label='obs')
    plt.scatter(z_goal[:500, 0], z_goal[:500, 1], label='goal_1')
    plt.scatter(z_goal[500:, 0], z_goal[500:, 1], label='goal_2')
    plt.legend()
    plt.savefig(osp.join(logger.get_dir(), f'vae_z_dim01_{epoch}.png'))

    plt.figure(figsize=(10, 10))
    norm_z = np.linalg.norm(z, axis=1)
    plt.hist(norm_z, 50, density=True, facecolor='g', alpha=0.75)
    plt.savefig(osp.join(logger.get_dir(), f'vae_norm_dist_{epoch}.png'))

    return {'eval/vae_reconstr_error': mse}


def eval_plan(args, env, agent, epoch, demo=False, profile=False):
    if args.train_set:
        from core.pasta.plan.compose_skills import execute
        if args.opt_mode =='adam':
            from core.pasta.plan.compose_skills import plan
        elif args.opt_mode =='rhp':
            from core.pasta.plan.compose_skills_rhp import plan
    else:
        from core.diffskill.plan.compose_skills import plan, execute
    from core.utils.visualization_utils import visualize_adam_info
    from core.utils.visualization_utils import visualize_mgoal
    init_vs, target_vs = get_eval_traj(args.cached_state_path, args.plan_step)

    save_dir = os.path.join(logger.get_dir(), f'plan_epoch_{epoch}')
    os.makedirs(save_dir, exist_ok=True)
    # Save the model used for planning
    if not args.run_plan:
        agent.save(os.path.join(save_dir, f'agent_{epoch}.ckpt'))

    normalized_scores, best_trajs, plan_times = [], [], []
    for i, (init_v, target_v) in enumerate(tqdm(zip(init_vs, target_vs), desc="plan")):
        plan_info = {'env': env, 'init_v': init_v, 'target_v': target_v}
        best_traj, all_traj, traj_info = plan(args, agent, plan_info, plan_step=args.plan_step, profile=profile)
        if profile:
            plan_times.append(traj_info['plan_time'])

        if not demo:
            img, sorted_idxes = visualize_all_traj(all_traj, overlay=False, color=(0, 0, 0), rgb=args.input_mode == 'rgbd', sort=args.opt_mode!='rhp')
            ncol = min(len(img), 5) if args.opt_mode !='rhp' else 1
            save_grid_img = make_grid(img[:10] / 255., ncol=ncol, padding=10, pad_value=0.)
            save_rgb(osp.join(save_dir, f'plan_traj_{i}.png'), save_grid_img[:, :, :3])
        else:
            img, sorted_idxes = visualize_all_traj(all_traj, overlay=False, demo=True)
            for ii, iimg in enumerate(img[:10]):
                save_rgb(osp.join(save_dir, f'plan_traj_{i}_sol_{ii}.png'), np.array(iimg[:, :, :3]).astype(np.float32))

        visualize_adam_info(all_traj, num_tools=args.num_tools, savename=osp.join(save_dir, f'adam_{i}.png'))
        if args.visualize_adam_pc:
            # visualize intermediate goals during the optimization process of planning.
            visualize_adam_pc(args, agent, epoch, all_traj, sorted_idxes, save_dir)

        execute_name = osp.join(save_dir, f'execute_{i}.gif')
        _, score = execute(env, agent, best_traj, save_name=execute_name, reset_primitive=True, demo=demo, save_dir=save_dir)
        normalized_scores.append(score)
        best_trajs.append(best_traj)
        if args.eval_single:
            break
    normalized_score = np.array(normalized_scores)
    thr = get_threshold(args.env_name)
    success = np.array(normalized_score > thr).astype(float)
    print('All normalized score:', normalized_scores)
    print('All success:', success)
    ret = {'plan_normalized_score': np.mean(normalized_score),
           'plan_success': np.mean(success),
           'best_trajs': best_trajs}
    if profile:
        ret['plan_time'] = np.array(plan_times).mean()
    return ret
