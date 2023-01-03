import numpy as np
from plb.envs.mp_wrapper import SubprocVecEnv


def to_action_mask(env, tool_mask):
    if isinstance(env, SubprocVecEnv):
        env_action_dims = env.getattr('taichi_env.primitives.action_dims', idx=0)
    else:
        env_action_dims = env.taichi_env.primitives.action_dims

    action_mask = np.zeros(env_action_dims[-1], dtype=np.float32)
    if isinstance(tool_mask, int):
        tid = tool_mask
        l, r = env_action_dims[tid:tid + 2]
        action_mask[l: r] = 1.
    else:
        for i in range(len(tool_mask)):
            l, r = env_action_dims[i:i + 2]
            action_mask[l: r] = tool_mask[i]
    return action_mask.flatten()


def get_generator_class(env_name):

    # generators = {'LiftSpread-v1': LiftspreadGenerator, 'GatherMove-v1': GathermoveGenerator, 'CutRearrange-v1': CutrearrangeGenerator}
    from core.traj_opt.gen_init_target.gathermove_generator import GathermoveGenerator
    from core.traj_opt.gen_init_target.crs_generator import CRSGenerator
    from core.traj_opt.gen_init_target.cutrearrange_generator import CutrearrangeGenerator
    generators = {'GatherMove-v1': GathermoveGenerator,
                  'CutRearrangeSpread-v1': CRSGenerator,
                  'CutRearrange-v1': CutrearrangeGenerator}

    return generators[env_name]


def get_threshold(env_name):
    threshold = {'LiftSpread-v1': 0.88, 'GatherMove-v1': 0.65, 'CutRearrange-v1': 0.8, 'CutRearrange-v2': 0.8, 'CutRearrangeSpread-v1': 0.83, 
    'CutRearrangeSpread-v2': 0.83, 'Lift-v1': 0.7, 'Spread-v1': 0.4}
    return threshold[env_name]


def get_num_traj(env_name):
    num_traj = {'LiftSpread-v1': 1000, 'GatherMove-v1': 1000, 'CutRearrange-v1': 1500, 'CutRearrange-v2': 1500, 'CutRearrangeSpread-v1': 600}
    return num_traj[env_name]


def get_tool_spec(env, env_name):
    if env_name == 'LiftSpread-v1':
        # [use tool1, ith position means loss for encouraging approaching actions for the ith tool],
        # [use tool2],
        # [use tool1 and 2]
        contact_loss_masks = [
            [1., 0., 0.],
            [0., 0., 0.],  # No loss for encouraging approaching actions
            [1., 0., 0.]
        ]
        # 0, 1 means just use the second tool's action space
        action_masks = [
            to_action_mask(env, [0, 1]),
            to_action_mask(env, [1, 0]),
            to_action_mask(env, [1, 1])
        ]
    elif env_name == 'GatherMove-v1' or env_name == 'Gather-v1' or env_name == 'Transport-v1':
        contact_loss_masks = [
            [1., 0., 0.],  # Gripper action
            [0., 0., 0.],  # No loss for encouraging approaching actions
            [1., 0., 0.]
        ]
        action_masks = [
            to_action_mask(env, [1, 0]),
            to_action_mask(env, [0, 1]),
            to_action_mask(env, [1, 1])
        ]
    elif env_name == 'CutRearrange-v1' or env_name == 'Cut-v1' or env_name == 'CutRearrange-v2':
        contact_loss_masks = [
            [1., 0.],  # Kinfe
            [0., 1.],
            [1., 1.]
        ]
        action_masks = [
            to_action_mask(env, 0),
            to_action_mask(env, 1),
            to_action_mask(env, [1, 1])
        ]
    elif env_name == 'CutRearrangeSpread-v1':
        contact_loss_masks = [
            [1., 0., 0.],  # Kinfe
            [0., 1., 0.],  # Mover
            [0., 0., 1.],  # Roller
            [1., 1., 1.]
        ]
        action_masks = [
            to_action_mask(env, [1, 0, 0]),
            to_action_mask(env, [0, 1, 0]),
            to_action_mask(env, [0, 0, 1]),
            to_action_mask(env, [1, 1, 1])
        ]
    elif env_name == 'Multicut-v1':
        contact_loss_masks = [
            [1.],  # Kinfe
        ]
        action_masks = [
            to_action_mask(env, 0),
        ]
    else:
        raise NotImplementedError

    return {'contact_loss_masks': contact_loss_masks,
            'action_masks': action_masks}


def set_render_mode(env, env_name, mode='mesh'):
    import taichi as ti
    import tina
    import os
    # cwd = os.getcwd()
    cwd = os.path.dirname(os.path.abspath(__file__))
    asset_path = os.path.join(cwd, '../..', 'assets')

    env.taichi_env.renderer.verbose = False
    if mode == 'mesh':
        # Add table
        model = tina.MeshModel(os.path.join(asset_path, 'table/Table_Coffee_RiceChest.obj'), scale=0.02)
        material = tina.Diffuse(tina.Texture(ti.imread(os.path.join(asset_path, "table/Table_Coffee_RiceChest/_Wood_Cherry_.jpg"))))
        env.taichi_env.renderer.bind(-1, model, material, init_trans=env.taichi_env.renderer.state2mat([-0.06, -0.38, 0.8, 1., 0., 0., 0.]))  # -0.38

        # Add cutting board
        if env_name == 'LiftSpread-v1' or env_name == 'Lift-v1' or env_name == 'Spread-v1':
            # Cutting board
            board_model = tina.MeshModel(os.path.join(asset_path, 'cuttingboard/Cutting_Board.obj'), scale=(0.02, 0.1, 0.02))
            board_material = tina.Diffuse(tina.Texture(ti.imread(os.path.join(asset_path, 'cuttingboard/textures/Cutting_Board_Diffuse.png'))))
            s = env.taichi_env.primitives[2].get_state(0)
            initial_mat = env.taichi_env.renderer.state2mat(s)
            target_mat = env.taichi_env.renderer.state2mat(
                [s[0] - 0.064182 + 0.05, s[1] + 0.02 - 0.06974, s[2], 0.707, 0., 0.707, 0.])  # TODO Does this need to be changed?
            env.taichi_env.renderer.bind(2, board_model, board_material,
                                         init_trans=np.linalg.pinv(initial_mat) @ target_mat)  # object pose @ init_pose ..
        elif env_name == 'GatherMove-v1':
            # Cutting board
            board_model = tina.MeshModel(os.path.join(asset_path, 'cuttingboard/Cutting_Board.obj'), scale=(0.02, 0.04, 0.02))
            board_material = tina.Diffuse(tina.Texture(ti.imread(os.path.join(asset_path, 'cuttingboard/textures/Cutting_Board_Diffuse.png'))))
            s = env.taichi_env.primitives[2].get_state(0)
            initial_mat = env.taichi_env.renderer.state2mat(s)
            target_mat = env.taichi_env.renderer.state2mat(
                [s[0] - 0.064182 + 0.05, s[1] - 0.007896, s[2], 0.707, 0., 0.707, 0.])  # TODO Does this need to be changed?
            env.taichi_env.renderer.bind(2, board_model, board_material,
                                         init_trans=np.linalg.pinv(initial_mat) @ target_mat)  # object pose @ init_pose ..
    elif mode == 'primitive':
        env.taichi_env.renderer.unbind_all()
    else:
        raise NotImplementedError


def get_reset_tool_state(env_name, dpc, tid, back=False):
    # Set the tool be on top of the component of interest. 
    # For pusher, set the tool to be on right side
    if 'CutRearrangeSpread'in env_name:
        if tid == 0:
            back_state = [0.8, 0.3, 0.1, 1.0, 0.0, 0.0, 0.0]
        elif tid == 1:
            back_state = [0.5, 0.25, 0.1, 1., 0.0, 0.0, 0.0]
        else:
            back_state = [0.25, 0.25, 0.1, 0.707, 0.707, 0., 0.]
        if back:
            return back_state
        com = np.mean(dpc[0], axis=0)
        if tid == 0 or tid == 2:  # knife or Rolling pin
            back_state[0] = com[0]
            back_state[2] = com[2]
        elif tid == 1:  # pusher
            back_state[0] = np.max(dpc[0, :, 0]) + 0.02
            back_state[1] = 0.05
            back_state[2] = com[2]
    elif env_name == 'CutRearrange-v1':
        if tid == 0:
            back_state = [0.5, 0.3, 0.5, 1.0, 0.0, 0.0, 0.0]
        else:
            back_state = [0.5, 0.10, 0.5, 0.707, 0.0, 0.707, 0.0, 0.18]
            # if not back: # gripper
            #     com = np.mean(dpc[0], axis=0)
            #     back_state[0] = com[0]
            #     back_state[1] = 0.05
            #     back_state[2] = com[2]
    else:
        raise NotImplementedError
    return back_state

def check_correct_tid(env_name, init_v, target_v, tid):
    if env_name =='CutRearrange-v1' or env_name == 'CutRearrange-v2':
        correct_tid = 0 if init_v % 3 == 0 else 1
        return tid == correct_tid
    elif env_name =='CutRearrangeSpread-v1':
        if init_v < 200:
            correct_tid = 0
        elif init_v < 400:
            correct_tid = 1
        elif init_v < 600 or init_v >= 620:
            correct_tid = 2
        else:
            return True
        return tid == correct_tid
    else:
        return True