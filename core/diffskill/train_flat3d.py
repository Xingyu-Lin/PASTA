from tqdm import tqdm
import torch
import json
import os
from core.utils import logger
from core.utils.core_utils import set_ipdb_debugger, set_random_seed
from plb.envs.mp_wrapper import make_mp_envs
from core.diffskill.args import get_args
from core.utils.diffskill_utils import prepare_buffer

def run_task(arg_vv, log_dir, exp_name):  # Chester launch
    if 'debug' in exp_name:
        set_ipdb_debugger()
        import faulthandler
        faulthandler.enable()
    args = get_args(cmd=False)
    if args.run_plan:
        assert arg_vv['resume_path'] is not None
        variant_path = os.path.join(os.path.dirname(arg_vv['resume_path']), 'variant.json')
        with open(variant_path, 'r') as f:
            vv = json.load(f)
        args.__dict__.update(**vv)
    args.__dict__.update(**arg_vv)
    set_random_seed(args.seed)

    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    log_dir = logger.get_dir()
    assert log_dir is not None
    os.makedirs(log_dir, exist_ok=True)

    # Need to make the environment before moving tensor to torch
    env = make_mp_envs(args.env_name, args.num_env, args.seed)
    print('------------Env created!!--------')

    args.cached_state_path = env.getattr('cfg.cached_state_path', 0)
    action_dim = env.getattr('taichi_env.primitives.action_dim')[0]
    args.action_dim = action_dim

    # Dump parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    if args.use_wandb:
        import wandb
        wandb.init(project='DynAbs',
                   group=args.wandb_group,
                   name=exp_name,
                   resume='allow', id=None,
                   settings=wandb.Settings(start_method='thread'))
        wandb.config.update(args, allow_val_change=True)

    from core.diffskill.agent import Agent
    from core.utils.diffskill_utils import aggregate_traj_info, dict_add_prefix
    from core.eval.eval_helper import eval_skills, eval_plan
    device = 'cuda'
    if args.run_plan:
        assert args.resume_path is not None
        agent = Agent(args, None, num_tools=args.num_tools, device=device)
        print('Agent created')
        agent.load(args.resume_path, args.load_modules)
        plan_info = eval_plan(args, env, agent, 0, demo=False)
        if 'best_trajs' in plan_info:
            del plan_info['best_trajs']

        plan_info = dict_add_prefix(plan_info, 'plan/')
        for key, val in plan_info.items():
            logger.record_tabular(key, val)
        logger.dump_tabular()
        env.close()
        exit()
    
    buffer = prepare_buffer(args, device)

    # # ----------preparation done------------------
    agent = Agent(args, None, num_tools=args.num_tools, device=device)
    print('Agent created')
    if args.resume_path is not None:
        agent.load(args.resume_path, args.load_modules)

    agent.vae.generate_cached_buffer(buffer)

    for epoch in range(args.il_num_epoch):
        set_random_seed(
            (args.seed + 1) * (epoch + 1))  # Random generator may change since environment is not deterministic and may change during evaluation
        infos = {'train': [], 'eval': []}
        for mode in ['train', 'eval']:
            epoch_tool_idxes = [buffer.get_epoch_tool_idx(epoch, tid, mode) for tid in range(args.num_tools)]
            for batch_tools_idx in tqdm(zip(*epoch_tool_idxes), desc=mode):
                data_batch = buffer.sample_tool_transitions(batch_tools_idx, epoch, device)
                if mode == 'eval':
                    with torch.no_grad():
                        train_info = agent.train(data_batch, mode=mode, epoch=epoch)
                else:
                    train_info = agent.train(data_batch, mode=mode, epoch=epoch)

                infos[mode].append(train_info)
            infos[mode] = aggregate_traj_info(infos[mode], prefix=None)
            infos[mode] = dict_add_prefix(infos[mode], mode + '/')
            # Wandb logging after each epoch
            if args.use_wandb:
                wandb.log(infos[mode], step=epoch)

        agent.update_best_model(epoch, infos['eval'])
        if args.use_wandb:
            wandb.log(agent.best_dict, step=epoch)

        if epoch % args.il_eval_freq == 0:
            skill_info, vae_info, plan_info = {}, {}, {}

            # Evaluate skills
            if args.eval_skill:
                skill_traj, skill_info = eval_skills(args, env, agent, epoch)

            # Plan
            if args.eval_plan and epoch % args.il_eval_freq == 0:
                agent.load_best_model()  # Plan with the best model
                plan_info = eval_plan(args, env, agent, epoch, demo=False)
                if 'best_trajs' in plan_info:
                    del plan_info['best_trajs']
                agent.load_training_model()  # Return training

            # Logging
            logger.record_tabular('epoch', epoch)
            all_info = {}

            plan_info = dict_add_prefix(plan_info, 'plan/')

            # all_info.update(**train_info)
            all_info.update(**skill_info)
            all_info.update(**vae_info)
            all_info.update(**plan_info)
            [all_info.update(**infos[mode]) for mode in infos.keys()]
            for key, val in all_info.items():
                logger.record_tabular(key, val)
            if args.use_wandb:
                wandb.log(all_info, step=epoch)
            logger.dump_tabular()

            # Save model
            if epoch % args.il_eval_freq == 0:
                agent.save(os.path.join(logger.get_dir(), f'agent_{epoch}.ckpt'))
    env.close()


if __name__ == '__main__':
    args = get_args(cmd=True)
    arg_vv = vars(args)
    log_dir = 'data/flat3d'
    exp_name = '0001_test_flat3d'
    run_task(arg_vv, log_dir, exp_name)
