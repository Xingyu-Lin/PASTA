import scipy.spatial.distance
from tqdm import tqdm
import numpy as np
from sklearn.cluster import DBSCAN
import json
import os
from core.pasta.args import get_args


def make_plot(all_x, all_labels):
    n = 100
    all_x, all_labels = all_x[:n], all_labels[:n]
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    matplotlib.rcParams['figure.dpi'] = 100
    fig = plt.figure(figsize=(4, 4))
    canvas = FigureCanvas(fig)

    images = []
    width, height = fig.get_size_inches() * fig.get_dpi()
    width = int(width)
    height = int(height)

    cmap = plt.get_cmap("tab10")
    ax1 = fig.add_subplot(111, projection='3d')

    # ax1.view_init(elev=130, azim=270)
    #
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    # plt.tight_layout()

    for x, label in zip(all_x, all_labels):
        q = ax1.scatter(x[:, 0], x[:, 1], x[:, 2], marker='o', alpha=0.5, s=5, c=label, cmap=cmap)
        canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        images.append(image.copy() / 255.)
        q.remove()
    plt.close()

    from core.utils.visualization_utils import make_grid
    img = make_grid(np.array(images)[:n], ncol=10)
    return img


def dbscan_cluster(xs, eps=0.03, min_samples=6, min_points=50, args=None):
    if args is not None:
        eps = args.dbscan_eps
        min_samples = args.dbscan_min_samples
        min_points = args.dbscan_min_points

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    def filter(x, label):
        x, label = x.copy(), label.copy()
        n = np.max(label)
        keep_idx, filter_idx = [], []
        for i in range(-1, n + 1):
            idx = np.where(label == i)
            if i != -1 and len(x[idx]) > min_points:
                label[idx] = (np.zeros(len(x[idx])) + len(keep_idx))
                keep_idx.append(idx[0].reshape(-1, 1))
            elif len(x[idx]) > 0:
                filter_idx.append(idx[0].reshape(-1, 1))
        if len(filter_idx) == 0:
            return label
        keep_idx, filter_idx = np.vstack(keep_idx)[:, 0], np.vstack(filter_idx)[:, 0]
        # For known x, assign it to the nearest label
        dist = scipy.spatial.distance.cdist(x[filter_idx], x[keep_idx])
        label[filter_idx] = label[keep_idx[np.argmin(dist, axis=1)]]
        return label

    if len(xs.shape) == 2:
        xs = xs[None]
    assert xs.shape[1] >= 1000  # At least 1000 points
    all_labels = []
    for x in xs if len(xs) < 500 else tqdm(xs, desc=f'Generating dbscan label'):
        dbscan.fit(x)
        label = dbscan.labels_
        label = filter(x, label)
        all_labels.append(label)
    return np.stack(all_labels, axis=0)


def generate_dbscan_label(buffer, args):
    """ For each trajectory in the buffer, generate the trajectory for reset motion.
        If reset motion length > max_reset_length, then just subsample them
    """
    dpc, _ = buffer.get_state(range(len(buffer)))
    np.random.seed(0)
    all_labels = dbscan_cluster(dpc, args=args)
    buffer.buffer['dbscan_labels'] = all_labels
    from core.utils.plb_utils import save_rgb
    save_dir = os.path.join(args.dataset_dir, 'vis_dbscan')
    os.makedirs(save_dir, exist_ok=True)

    for traj_id in np.random.choice(len(dpc) // 50, 10):
        idx = range(traj_id * 50, traj_id * 50 + 50)
        img = make_plot(dpc[idx], all_labels[idx])
        save_rgb(os.path.join(save_dir, f'cluster_traj_{traj_id}.png'), img)


def run_task(arg_vv, log_dir, exp_name):  # Chester launch
    from core.pasta.args import get_args
    import os
    args = get_args()
    args.__dict__.update(**arg_vv)
    args.dataset_dir = os.path.dirname(args.dataset_path)
    from core.diffskill.imitation_buffer import ImitationReplayBuffer
    import os

    default_args = get_args()
    buffer = ImitationReplayBuffer(args=default_args)
    buffer.load(args.dataset_path)
    generate_dbscan_label(buffer, args)
    buffer.save(os.path.join(args.dataset_dir, 'dataset.gz'))
    with open(os.path.join(args.dataset_dir, 'dbscan_vv.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    args = get_args(cmd=True)
    arg_vv = vars(args)
    log_dir = 'data/generate_dbscan'
    exp_name = ''
    run_task(arg_vv, log_dir, exp_name)
