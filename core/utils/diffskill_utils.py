import torch
from torch import nn
import numpy as np
import os
import cv2 as cv
from core.traj_opt.env_spec import get_threshold
from core.utils.pc_utils import resample_pc
from core.utils.plb_utils import save_numpy_as_gif, make_grid, save_rgb
from plb.envs.mp_wrapper import SubprocVecEnv
import torch.nn.functional as F
from tqdm import tqdm


def softclipping(x, l, r):
    x = r - F.softplus(r - x)
    x = l + F.softplus(x - l)
    x = torch.clamp(x, max=r)
    return x


def sample_n_ball(B, N, ball_norm, device='cuda'):
    """ Return B uniform samples from N-ball, specifying the norm of the ball using normalized Gaussian"""
    x = torch.normal(mean=torch.zeros(B, N, device=device))
    norm = torch.norm(x, dim=1)[:, None]
    r = torch.sqrt(torch.rand(size=norm.shape, device=device))  # [0, 1]
    return x / norm * r * ball_norm


def get_grad_norm(loss, network):
    """ Get the norm of the gradient w.r.t. the loss. Gradient is cleared before and after.
    Graph is retained so that backward can still be called for the same forward function"""
    network.zero_grad()
    loss.backward(retain_graph=True)
    ele = []
    for param in network.parameters():
        if param.grad is not None:
            ele.append(torch.norm(param.grad.data))
    grad = torch.stack(ele, dim=0).mean()
    network.zero_grad()
    return grad


def dict_add_prefix(d, prefix, skip_substr=None):
    new_d = {}
    for key, val in d.items():
        if skip_substr is None or skip_substr not in key:
            new_d[prefix + key] = val
        else:
            new_d[key] = val
    return new_d


def batch_pred_n(func, N, batch_size=2048, collate_fn=None):
    # For network of unconditioned generation where the only input variable is the batch size
    # N: Number of total samples needed
    if N <= batch_size:
        return func(N)
    else:
        all_pred = []
        gen = range(0, N, batch_size) if N // batch_size < 50 else tqdm(range(0, N, batch_size), desc=f'Batch prediction')
        for i in gen:
            all_pred.append(func(min(i + batch_size, N) - i))
        if collate_fn is None:
            return torch.cat(all_pred, dim=0)
        else:
            return collate_fn(all_pred)


def batch_pred(func, kwargs, batch_size=2048, collate_fn=None):
    rand_key = list(kwargs.keys())[0]
    N = len(kwargs[rand_key])
    if N <= batch_size:
        return func(**kwargs)
    else:
        all_pred = []
        gen = range(0, N, batch_size) if N // batch_size < 20 else tqdm(range(0, N, batch_size), desc=f'Batch prediction')
        for i in gen:
            new_kwargs = {}
            for key, val in kwargs.items():
                if isinstance(val, torch.Tensor) or isinstance(val, np.ndarray):
                    new_kwargs[key] = val[i:min(i + batch_size, N)]
                else:
                    new_kwargs[key] = val
            pred = func(**new_kwargs)
            all_pred.append(pred)
        if collate_fn is None:
            return torch.cat(all_pred, dim=0)
        else:
            return collate_fn(all_pred)


def batch_rand_int(low, high, size):
    # Generate random int from [low, high)
    return np.floor(np.random.random(size) * (high - low) + low).astype(np.int)


def batch_rand_float(low, high, size):
    return torch.rand(size, dtype=torch.float) * (high - low) + low


env_action_dims = None


def to_action_mask(env, tool_mask):
    global env_action_dims
    if env_action_dims is None:
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


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def gaussian_logl(z):
    """ Loglikelihood of gaussian dist. z: [shape] x z_dim"""
    return -0.5 * torch.sum(z ** 2, dim=-1)


def traj_gaussian_logl(z):
    """z in the shape of num_traj x traj_step x z_dim """
    assert len(z.shape) == 3
    n, num_step, zdim = z.shape
    z_all = gaussian_logl(z.view(n * num_step, zdim)).view(n, num_step)
    return z_all.sum(dim=1)


def aggregate_traj_info(trajs, prefix='info_'):
    infos = {}
    for key in trajs[0].keys():
        if prefix is None or key.startswith(prefix):
            if prefix is None:
                s = key
            else:
                s = key[len(prefix):]
            vals = np.concatenate([np.array(traj[key]).flatten() for traj in trajs])
            infos[f"{s}_mean"] = np.mean(vals)
            # infos[f"{s}_median"] = np.median(vals)
            # infos[f"{s}_std"] = np.std(vals)
    return infos


def cv_render(img, name='GoalEnvExt', scale=5):
    '''Take an image in ndarray format and show it with opencv. '''
    img = img[:, :, :3]
    new_img = img[:, :, (2, 1, 0)]
    h, w = new_img.shape[:2]
    new_img = cv.resize(new_img, (w * scale, h * scale))
    cv.imshow(name, new_img)
    cv.waitKey(20)


def debug_show_img(img):
    img = img_to_np(img)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(img[:, :, :3])
    plt.show()


def img_to_tensor(imgs, mode):  # BNMC to BCNM
    imgs = torch.FloatTensor(imgs).permute([0, 3, 1, 2]).contiguous()
    B, C, N, M = imgs.shape
    if mode == 'rgb':
        if C == 3:
            return imgs
        elif C == 4:
            return imgs.view(B, C // 4, 4, N, M)[:, :, :3, :, :].reshape(B, C // 4 * 3, N, M)
    elif mode == 'rgbd':
        return imgs
    elif mode == 'd':
        return imgs.view(B, C // 4, 4, N, M)[:, :, 3, :, :].reshape(B, C // 4, N, M)


def img_to_np(imgs):  # BCNM to BNMC
    if len(imgs.shape) == 4:
        return imgs.detach().cpu().permute([0, 2, 3, 1]).numpy()
    elif len(imgs.shape) == 3:
        return imgs[None].detach().cpu().permute([0, 2, 3, 1]).numpy()[0]


def get_iou(a, b, mode='normalized_soft_iou'):
    assert a.shape == b.shape, "shape a: {}, shape  b:{}".format(a.shape, b.shape)
    assert len(a.shape) == 4
    if mode == 'l2':  # Within (0, 1)
        a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
        a = a / np.linalg.norm(a, axis=-1, keepdims=True) ** 2 * np.sum(a, axis=-1)
        b = b / np.linalg.norm(b, axis=-1, keepdims=True) ** 2 * np.sum(b, axis=-1)
        I = np.sum(a * b, axis=-1)
        U = np.sum(a + b, axis=-1) - I
        return I / U
    elif mode == 'soft_iou':
        a = a / np.max(a, axis=(1, 2, 3), keepdims=True)
        b = b / np.max(b, axis=(1, 2, 3), keepdims=True)
        I = np.sum(a * b, axis=(1, 2, 3))
        U = np.sum(a + b, axis=(1, 2, 3)) - I
        return I / U
    elif mode == 'normalized_soft_iou':
        a = a / np.max(a, axis=(1, 2, 3), keepdims=True)
        b = b / np.max(b, axis=(1, 2, 3), keepdims=True)
        s = b.reshape(b.shape[0], -1)
        K = (2 * np.sum(s, axis=-1) - np.sum(s * s, axis=-1)) / (np.sum(s * s, axis=-1))  # normalization
        I = np.sum(a * b, axis=(1, 2, 3))
        U = np.sum(a + b, axis=(1, 2, 3)) - I
        return I / U * K


def compute_pairwise_iou(mass_grid_list):
    m = np.array(mass_grid_list)
    T = m.shape[0]
    m1 = np.tile(m[None, :, :, :, :], [T, 1, 1, 1, 1]).reshape(T * T, 64, 64, 64)
    m2 = np.tile(m[:, None, :, :, :], [1, T, 1, 1, 1]).reshape(T * T, 64, 64, 64)
    ious = get_iou(m1, m2)
    return ious


def load_target_imgs(cached_state_path, mode=None, ret_tensor=True):
    np_target_imgs = np.load(os.path.join(cached_state_path, 'target/target_imgs.npy'))
    if ret_tensor:
        return np_target_imgs, img_to_tensor(np_target_imgs, mode)
    else:
        return np_target_imgs


def load_target_pcs(cached_state_path):
    target_mass_grids = []
    import glob
    from natsort import natsorted
    target_paths = natsorted(glob.glob(os.path.join(cached_state_path, 'target/target_[0-9]*.npy')))
    for path in target_paths:
        target_mass_grid = np.load(path)
        target_mass_grids.append(target_mass_grid[:1000])
    return np.array(target_mass_grids)


def load_target_dbscan(args, cached_state_path, np_target_pcs):
    dbscan_name = f'{args.dbscan_eps}-{args.dbscan_min_samples}-{args.dbscan_min_points}'
    target_dbscan_path = os.path.join(cached_state_path, 'target/target_dbscan-{}.npy'.format(dbscan_name))
    if os.path.exists(target_dbscan_path):
        np_target_dbscan = np.load(target_dbscan_path)
        if len(np_target_dbscan) == len(np_target_pcs):
            return np_target_dbscan
    from core.pasta.generate_dbscan_label import dbscan_cluster
    np_target_dbscan = dbscan_cluster(np_target_pcs, args=args)
    np.save(target_dbscan_path, np_target_dbscan)
    print('Saving dbscan to ', target_dbscan_path)
    return np_target_dbscan


def load_target_info(args, device, load_set=True):
    np_target_imgs, target_imgs = load_target_imgs(args.cached_state_path, args.img_mode)
    np_target_pcs = load_target_pcs(args.cached_state_path)
    target_imgs = target_imgs.to(device)
    target_info = {
        'np_target_imgs': np_target_imgs,
        'target_imgs': target_imgs,
        'np_target_pc': np_target_pcs}
    if load_set:
        np_target_dbscan = load_target_dbscan(args, args.cached_state_path, np_target_pcs)
        target_info['np_target_dbscan'] = np_target_dbscan
    return target_info


def visualize_trajs(trajs, ncol, key, save_name, vis_target=False, demo_obses=None, clean=False):
    """vis_target: Whether to overlay the target images. demo_obses: whether show original demonstration on the side """
    horizon = max(len(trajs[i]['obses']) for i in range(len(trajs))) + 10  # Add frames after finishing

    all_imgs = []
    for i in range(len(trajs)):
        imgs = []
        for j in range(horizon):
            if j < len(trajs[i]['obses']):
                img = trajs[i]['obses'][j, :, :, :3].copy()  # Do not change the input images
                if vis_target:
                    img[:, :, :3] = img[:, :, :3] * 0.7 + trajs[i]['target_img'][:, :, :3] * 0.3
                if key is not None:
                    if not clean and j < len(trajs[i][key]):
                        if img.shape[0] == 256:
                            write_number(img, float(trajs[i][key][j]), fontScale=1.5, lineType=3, loc=(80, 250))
                        else:
                            write_number(img, float(trajs[i][key][j]))
            else:
                if not clean:
                    # Set the boundary to green
                    margin = 2
                    img[:margin, :] = img[-margin:, :] = img[:, :margin] = img[:, -margin:] = [0., 1., 0.]
            if demo_obses is not None:
                combined_img = np.hstack([img, demo_obses[i, min(j, len(demo_obses[i]) - 1)]])
                imgs.append(combined_img)
            else:
                imgs.append(img)
        all_imgs.append(imgs)
    a = np.array(all_imgs).swapaxes(0, 1)
    all_frames = []
    for f in range(a.shape[0]):
        padding = 0 if clean else 3
        frame = make_grid(a[f] * 255, ncol=ncol, padding=padding)
        all_frames.append(frame)

    save_numpy_as_gif(np.array(all_frames), save_name)


def write_number(img, number, color=(1., 1., 1.), loc=(20, 60), fontScale=0.4, lineType=2):  # Inplace modification
    if not isinstance(number, str) and np.isnan(number):
        return
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = loc
    fontScale = fontScale
    fontColor = color
    lineType = lineType
    if isinstance(number, str):
        cv2.putText(img, '{}'.format(number),
                    bottomLeftCornerOfText, font,
                    fontScale, fontColor, lineType)
    elif isinstance(number, int):
        cv2.putText(img, str(number),
                    bottomLeftCornerOfText, font,
                    fontScale, fontColor, lineType)
    else:
        cv2.putText(img, '{:.2f}'.format(number),
                    bottomLeftCornerOfText, font,
                    fontScale, fontColor, lineType)
    return


def write_number_tensors(tensor_imgs, numbers, color=(1., 1., 1.), loc=(20, 60)):
    device = tensor_imgs.device
    imgs = img_to_np(tensor_imgs)
    new_imgs = []
    for img, number in zip(imgs, numbers):
        tmp_img = img.copy()
        write_number(tmp_img, number, color, loc)
        new_imgs.append(tmp_img)
    new_imgs = np.array(new_imgs)
    return img_to_tensor(new_imgs, mode='rgbd').to(device)

def get_img(args, xs, ys):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2 as cv
    plt.rcParams.update({
        "figure.facecolor": (1.0, 1.0, 1.0, 0.),
        "axes.facecolor": (0.0, 0.0, 0.0, 1),
        "ytick.labelsize": 25,
        "ytick.color": (1., 1., 1., 1.),
    })

    # make a Figure and attach it to a canvas.
    fig = Figure(figsize=(8, 8), dpi=32)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting here
    ax = fig.add_axes([0.55, 0.02, 0.4, 0.2])
    ax.tick_params('both', length=0, width=0, which='both')
    thr = get_threshold(args.env_name)
    ax.plot(np.linspace(-1, 2, 100), np.ones(100) * thr, linewidth=3, color=(0., 1., 0., 0.7))
    ax.set_yticks([0., thr])

    ax.grid(True, color=(0.5, 0.5, 0.5, 1))
    ax.plot(xs, ys, color=(1., 0, 0, 1), linewidth=5)
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(-0.05, thr + 0.3)
    ax.set_xticks([])

    # Retrieve a view on the renderer buffer
    canvas.draw()
    buf = canvas.buffer_rgba()
    # convert to a NumPy array
    X = np.asarray(buf).astype(np.float32) / 255.
    # X = cv.resize(X, (args.img_size, args.img_size), interpolation=cv.INTER_AREA)
    return X


def visualize_dataset(demo_path, cached_state_path, save_name, overlay_target=None, visualize_reset=False, num_moves=1):
    from core.diffskill.buffer import ReplayBuffer
    buffer = ReplayBuffer(None)
    buffer.load(demo_path)
    max_step = 50
    horizon = max_step * num_moves
    N = buffer.cur_size // horizon
    all_imgs = []

    for i in range(N):
        img_idx = 0
        # if i % 8 != 0:
        imgs = []
        if overlay_target:
            # init_v = buffer.buffer['init_v'][i * horizon]
            target_v = buffer.buffer['target_v'][i * horizon]
            print('loading img:', os.path.join(cached_state_path, 'target/target_{}.png'.format(target_v)))
            target_img = cv.imread(os.path.join(cached_state_path, 'target/target_{}.png'.format(target_v)))
            target_img = cv.cvtColor(target_img, cv.COLOR_BGR2RGB) / 255.
        # emds = buffer.buffer['info_emds'][i * horizon: (i + 1) * horizon]
        for move in range(num_moves):
            for j in range(max_step):
                img = buffer.buffer['obses'][i * horizon + max_step * move + j].copy()
                num = float(buffer.buffer['info_emds'][i * horizon + max_step * move + j])
                if overlay_target:
                    img = img[:, :, :3] * 0.8 + target_img * 0.2
                    write_number(img, num)
                    img_idx += 1
                imgs.append(img)
            if visualize_reset:
                for j in range(buffer.buffer['reset_motion_obses'].shape[1]):
                    if j < buffer.buffer['reset_motion_lens'][i]:  # If no reset, img will be the last img above
                        img = buffer.buffer['reset_motion_obses'][i][j].copy()
                        img = img[:, :, :3] * 0.8 + target_img * 0.2
                        write_number(img, buffer.buffer['reset_info_emds'][i][j])
                    imgs.append(img)
        all_imgs.append(imgs)
    a = np.array(all_imgs).swapaxes(0, 1)
    all_frames = []
    for f in range(a.shape[0]):
        col = min(N, 10)
        frame = make_grid(a[f] * 255, ncol=col)
        all_frames.append(frame)

    save_numpy_as_gif(np.array(all_frames), save_name)


def visualize_agent_dataset(buffer, cached_state_path, save_name, overlay_target=None):
    horizon = 50
    N = buffer.cur_size // horizon
    print('visualize_dataset, N: ', N)
    all_imgs = []

    for i in range(N):
        # i = N-1
        imgs = []
        if overlay_target:
            # init_v = buffer.buffer['init_v'][i * horizon]
            target_v = buffer.buffer['target_v'][i * horizon]
            target_img = cv.imread(os.path.join(cached_state_path, 'target/target_{}.png'.format(target_v)))
            target_img = cv.cvtColor(target_img, cv.COLOR_BGR2RGB) / 255.
        target_ious = buffer.buffer['target_ious'][i * horizon: (i + 1) * horizon]
        print(buffer.buffer['target_ious'].shape)
        # print('----------------------')
        for j in range(horizon):
            img = buffer.buffer['obses'][i * horizon + j]
            if overlay_target:
                img = img[:, :, :3] * 0.7 + target_img * 0.3
                write_number(img, float(target_ious[j]))
            imgs.append(img)
        all_imgs.append(imgs)
    a = np.array(all_imgs).swapaxes(0, 1)
    all_frames = []
    for f in range(a.shape[0]):
        frame = make_grid(a[f] * 255, ncol=10)
        all_frames.append(frame)

    save_numpy_as_gif(np.array(all_frames), save_name)


from sklearn.neighbors import NearestNeighbors


def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist


def calculate_performance(buffer_path, max_step=50, num_moves=1):
    from core.diffskill.buffer import ReplayBuffer
    buffer = ReplayBuffer(args=None)
    buffer.load(buffer_path)
    horizon = max_step * num_moves
    final_emds = buffer.buffer['info_normalized_performance'][:buffer.cur_size].reshape(-1, horizon)[:, -1]
    print("final_emd_shape:", final_emds.shape)
    return np.mean(final_emds)


def calculate_performance_buffer(buffer, max_step=50, num_moves=1):
    horizon = max_step * num_moves
    final_emds = buffer.buffer['info_normalized_performance'][:buffer.cur_size].reshape(-1, horizon)[:, -1]
    print("final_emd_shape:", final_emds.shape)
    return np.mean(final_emds)


def get_component_masks(pcl1, pcl2, dbscan, thres=1e-2):
    """Takes in two pointcloud and return their masks"""
    dbscan.fit(pcl1)
    label1 = dbscan.labels_
    dbscan.fit(pcl2)
    label2 = dbscan.labels_
    mask1, mask2 = np.ones((len(pcl1)), dtype=bool), np.ones((len(pcl2)), dtype=bool)
    for i in range(np.max(label1) + 1):
        for j in range(np.max(label2) + 1):
            idx1, idx2 = np.where(i == label1)[0], np.where(j == label2)[0]
            loss = chamfer_distance(pcl1[idx1], pcl2[idx2])
            print(i, j, loss)
            if loss < thres:
                mask1[idx1] = 0
                mask2[idx2] = 0
    # mask out noise
    mask1[np.where(-1==label1)[0]] = 0
    mask2[np.where(-1==label2)[0]] = 0
    return mask1, mask2

def load_tool_points(args, action_primitives, buffer, force=False):
    tool_particles = []
    N = buffer.cur_size
    # first check whether state already has the tool particles
    if not force and buffer.buffer['states'].shape[1] > 3300:
        print("buffer already has tool particles. Stop modifying the state.")
        return
    for i in range(N):
        all_tool_points = []
        for tid, tool in enumerate(action_primitives):
            tid_state_idx = 3000 + tid * args.dimtool
            tool_points = tool.get_surface_points(state=buffer.buffer['states'][i, tid_state_idx:tid_state_idx+8])
            all_tool_points.append(tool_points)
        tool_particles.append(np.vstack(all_tool_points).flatten())
    tool_particles = np.vstack(tool_particles)
    new_states = np.empty((buffer.buffer['states'].shape[0], 3000+len(action_primitives)*args.dimtool+tool_particles.shape[1]), dtype=np.float)
    new_states[:N] = np.concatenate([buffer.buffer['states'][:N, :3000+len(action_primitives)*args.dimtool], tool_particles], axis=-1)
    buffer.buffer['states'] = new_states
    print("New buffer state dimension: ", buffer.buffer['states'].shape)

def prepare_buffer(args, device):
    from core.diffskill.imitation_buffer import ImitationReplayBuffer, filter_buffer_nan
    # Load buffer
    buffer = ImitationReplayBuffer(args)
    buffer.load(args.dataset_path)
    filter_buffer_nan(buffer)
    buffer.generate_train_eval_split(filter=args.filter_buffer)
    target_info = load_target_info(args, device, load_set=args.train_set)
    buffer.__dict__.update(**target_info)
    return buffer