import os
import cv2
import pdb
import torch
import numpy as np
from core.utils.plb_utils import save_numpy_as_gif

# TODO: handle case where multiple manipulators are present.

KEY_TO_STATUS = {
    **dict.fromkeys([ord("a"), ord("A")], (0, -1)),  # left
    **dict.fromkeys([ord("d"), ord("D")], (0, 1)),  # right
    **dict.fromkeys([ord("j"), ord("J")], (1, -1)),  # up
    **dict.fromkeys([ord("k"), ord("K")], (1, 1)),  # down
    **dict.fromkeys([ord("w"), ord("W")], (3, -1)),  # forward
    **dict.fromkeys([ord("s"), ord("S")], (3, 1)),  # backward
    **dict.fromkeys([ord("q"), ord("Q")], (4, -1)),  # forward
    **dict.fromkeys([ord("e"), ord("E")], (4, 1)),  # backward
    **dict.fromkeys([ord("u"), ord("U")], (5, -1)),  # forward
    **dict.fromkeys([ord("i"), ord("I")], (5, 1)),  # backward
    **dict.fromkeys([ord(","), ord("<")], (7, -1)),  # forward
    **dict.fromkeys([ord("."), ord(">")], (7, 1)),  # backward
    **dict.fromkeys([ord("v"), ord("V")], (9, -1)),  # forward
    **dict.fromkeys([ord("b"), ord("B")], (9, 1)),  # backward
    **dict.fromkeys([ord("r"), ord("R")], "reset"),
    **dict.fromkeys([ord("c"), ord("C")], "quit"),
    **dict.fromkeys([ord("x"), ord("X")], "save"),
    **dict.fromkeys([ord("p"), ord("P")], "switch"), # Switch tools
    # **dict.fromkeys([ord("p"), ord("P")], "swap"),
}
VELOCITY = 0.5
MAX_HORIZON = float('inf')
DATA_DIR = "./data"


def run_sandbox():
    tool_id = 0
    env_name = 'CutRearrangeSpread'
    env_save_dir = os.path.join(DATA_DIR, env_name, 'data')
    os.makedirs(env_save_dir, exist_ok=True)

    from plb.envs import make
    from core.diffskill.args import get_args
    from plb.engine.taichi_env import TaichiEnv
    from plb.optimizer.solver import Solver
    from plb.algorithms.logger import Logger

    device = 'cuda'

    log_dir = './data/connect'
    args = get_args("")

    obs_channel = len(args.img_mode)
    img_obs_shape = (args.image_dim, args.image_dim, obs_channel)

    from plb.envs.multitask_env import MultitaskPlasticineEnv
    env = MultitaskPlasticineEnv(cfg_path=f'cut_rearrange_spread.yml', generating_cached_state=True)
    taichi_env = env.taichi_env

    action_dim = env.taichi_env.primitives[0].action_dim
    print(action_dim)

    env.reset(init_v=0, target_v=0)
    cv2.namedWindow(env_name, cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow(env_name, 1000, 500)
    # cv2.namedWindow(env_name + '_depth', cv2.WINDOW_GUI_NORMAL)

    traj_cnt = 0
    trajectory = []
    frames = []
    horizon_count = 0
    while True:
        action = np.zeros(env.taichi_env.primitives.action_dim)
        frame = taichi_env.render(mode='rgb', img_size=256) * 255.  # show target
        frame = frame[:, :, :3].copy()

        frame = np.uint8(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.putText(frame, str(horizon_count), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        frames.append(frame[:, :, [2,1, 0]].copy())
        cv2.imshow(env_name, frame)
        key = cv2.waitKey(10)
        try:
            status = KEY_TO_STATUS[key]
            if status == "quit":
                print("Quitting...")
                break
            elif status == "reset":
                env.reset()
                trajectory = []
                frames = []
                horizon_count = 0
                tool_id = 0
            elif status == "save":
                obs = frame
                trajectory.append((obs, None))
                print('2:', frame.shape)
                index = len(os.listdir(env_save_dir))
                save_dir = os.path.join(env_save_dir, str(index).zfill(4))
                os.makedirs(save_dir, exist_ok=False)

                torch.save(trajectory, os.path.join(save_dir, 'trajectory.pth'))
                np.save(os.path.join(save_dir, f'target_grid_mass.npy'), env.taichi_env.simulator.grid_m.to_numpy())
                np.save(os.path.join(save_dir, f'target_positions.npy'), env.taichi_env.simulator.get_x(0))
                save_numpy_as_gif(np.array(frames)[:, :, :, :3], os.path.join(save_dir, f'trajectory.gif'))
                print(f'[Horizon={len(trajectory) - 1}] Successfully saved to {save_dir}')
                # reset params
                env.reset()
                trajectory = []
                frames = []
                horizon_count = 0
                traj_cnt += 1
                tool_id=0
            elif status == 'switch':
                tool_id = 1 - tool_id
                print(f'Swithcing to tool {tool_id}')
            else:
                new_cnt = (horizon_count + 1)
                if new_cnt <= MAX_HORIZON:
                    if tool_id ==0:
                        action[status[0]] = status[1] * VELOCITY
                    else:
                        action[status[0]+7] = status[1] * VELOCITY
                    obs = frame
                    print('3:', frame.shape)
                    trajectory.append((obs, action))
                    horizon_count = new_cnt
                else:
                    print("Horizon is maxed out! Please consider saving the environment.")
        except KeyError:
            pass
        env.taichi_env.step(action)
    cv2.destroyWindow(env_name)


if __name__ == '__main__':
    run_sandbox()