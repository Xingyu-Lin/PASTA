import torch
import numpy as np
import os.path as osp
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import cm
from core.utils.pc_utils import batch_resample_pc
from core.utils.plb_utils import make_grid

from core.utils.diffskill_utils import write_number
from core.utils.plb_utils import save_numpy_as_gif, save_numpy_as_video
from core.utils.open3d_utils import visualize_point_cloud_batch, visualize_point_cloud_plt

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


def visualize_traj_actions(trajs, save_name):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for i in range(len(trajs)):
        actions = trajs[i]['actions']
        for j in range(actions.shape[1]):
            if np.any(actions[:, j] > 0):
                plt.plot(list(range(actions.shape[0])), actions[:, j], label=j, linewidth=5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(-0.35, -0.1, 1, 1.2), ncol=4, prop={'size': 20})
    plt.tight_layout()
    plt.savefig(save_name)


def visualize_mgoal(goal_his, savename):
    # goal his shape: list of size niter,  num_sample x num_step x 64 x 64 x 4
    imgs = []
    for i in range(0, goal_his.shape[0], 5):
        mgoal = goal_his[i].transpose(1, 0, 2, 3, 4)
        mgoal = np.concatenate(list(mgoal), axis=2)
        grid_img = make_grid(mgoal, ncol=5, padding=5, pad_value=0.5)
        write_number(grid_img, i)
        imgs.append(grid_img)
    for j in range(20):
        imgs.append(grid_img)

    save_numpy_as_gif(np.array(imgs) * 255, savename)


def visualize_adam_info(all_traj, savename, num_tools=2, topk=None):
    """If topk is not None, only plot the loss and zlogl for the topk trajectories"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'silver']
    for i, traj in enumerate(all_traj):
        num_iter = len(traj['his_loss'])
        name = 'tool'
        cid = 0
        for t in traj['tool']:
            name = name + f'_{t}'
            cid = cid * num_tools + t
        # Two tools
        c = colors[cid % len(colors)]
        ax1.plot(range(num_iter), np.array(traj['his_loss']), label=name, color=c)
        ax2.plot(range(num_iter), np.array(traj['his_traj_score']), label=name, color=c)
        # ax3.plot(range(num_iter), np.array(traj['his_traj_zlogl']), label=name, color=c)

    ax1.set_title('total_loss')
    ax2.set_title('traj_score')
    ax3.set_title('traj_zlogl')
    legend_without_duplicate_labels(ax2)
    plt.tight_layout()
    plt.savefig(savename)


def visualize_all_traj_his(all_traj, all_imgs, img_idx, his_idx, overlay=False, demo=False, color=(1, 1, 1)):
    # all_imgs shape: len(all_traj), his_length, plan_step, *image_shape
    all_traj_imgs = []
    all_score = []
    fontScale = 2.5
    loc = (80, 700)
    for idx, traj in enumerate(all_traj):
        imgs = all_imgs[idx, img_idx]
        traj_imgs = []
        for i in range(len(imgs) - 1):
            if overlay:
                overlay_img = (imgs[i] * 0.7 + imgs[i + 1] * 0.3).copy()
            else:
                overlay_img = imgs[i].copy()
            if not demo:
                if i == len(imgs) - 2:
                    write_number(overlay_img, traj['his_traj_pred_r'][his_idx], color=color, fontScale=fontScale, loc=loc)
                else:
                    write_number(overlay_img, traj['his_traj_succ'][his_idx][i], color=color, fontScale=fontScale, loc=loc)
            traj_imgs.append(overlay_img)
        last_goal = imgs[-1].copy()
        if not demo:
            write_number(last_goal, 'Trajectory score: ' + str(round(traj['his_traj_score'][his_idx], 2)), color=color, fontScale=fontScale, loc=loc, lineType=5)
            name = ''
            for t in traj['tool']:
                name = name + str(t)
            write_number(traj_imgs[0], name, color=color, fontScale=fontScale, loc=(80, 270))
        traj_imgs.append(last_goal)

        traj_imgs = np.hstack(traj_imgs)
        all_traj_imgs.append(traj_imgs)
        all_score.append(traj['traj_score'])
    idx = np.argsort(np.array(all_score))[::-1]  # Sort by the scores
    all_traj_imgs = np.array(all_traj_imgs)[idx]
    return all_traj_imgs, idx


def visualize_all_traj(all_traj, overlay=False, demo=False, color=(1, 1, 1), rgb=False, sort=True):
    all_traj_imgs = []
    all_score = []
    if rgb:
        fontScale = 0.4
        loc = (20, 60)
        color = (255. *0.8, 255.*0.8,255.*0.8)
    else:
        fontScale = 2
        loc = (20, 80)
    for traj in all_traj:
        imgs = traj['traj_img']
        traj_imgs = []
        for i in range(len(imgs) - 1):
            if overlay:
                overlay_img = (imgs[i] * 0.7 + imgs[i + 1] * 0.3).copy()
            else:
                overlay_img = imgs[i].copy()
            if not demo:
                if i == len(imgs) - 2:
                    write_number(overlay_img, traj['pred_r'], color=color, fontScale=fontScale, loc=loc)
                else:
                    write_number(overlay_img, traj['traj_succ'][i], color=color, fontScale=fontScale, loc=loc)
            traj_imgs.append(overlay_img)
        last_goal = imgs[-1].copy()
        if not demo:
            write_number(last_goal, traj['traj_score'], color=color, fontScale=fontScale, loc=loc)
            name = ''
            for t in traj['tool']:
                name = name + str(t)
            write_number(traj_imgs[0], name, color=color, fontScale=fontScale, loc=(80, 270))
        traj_imgs.append(last_goal)

        traj_imgs = np.hstack(traj_imgs)
        all_traj_imgs.append(traj_imgs)
        all_score.append(traj['traj_score'])
    if sort:
        idx = np.argsort(np.array(all_score))[::-1]  # Sort by the scores
    else:
        idx = np.arange(len(all_score))  # Sort by the scores
    all_traj_imgs = np.array(all_traj_imgs)[idx]
    return all_traj_imgs, idx


def visualize_traj_neighborhood(agent, traj, all_z=None, t_axes=(0, 2)):
    # all_z: (z_train, z_eval), z_train.shape: (n_train, 5) 
    u = np.vstack([traj['u_obs'], traj['zs']])
    u = torch.FloatTensor(u).cuda()
    img = visualize_neighborhood(agent, u[:-1, :], u[1:], traj['tool'], all_z=all_z, t_axes=t_axes)
    return img


def visualize_adam_pc(args, agent, epoch, all_traj, sorted_idxes, save_dir, view_init=(140, -90), use_mitsuba=False):
    sorted_all_traj = [all_traj[i] for i in sorted_idxes[:5]]
    his_us_np = np.vstack([np.expand_dims(traj['his_traj_u'], 0) for traj in sorted_all_traj])
    n, his_length, plan_step, dimu = his_us_np.shape
    if not use_mitsuba:
        dpc_img, goal_dpc_img = visualize_point_cloud_plt(sorted_all_traj[0]['dpc'], view_init=view_init), visualize_point_cloud_plt(
        sorted_all_traj[0]['goal_dpc'], view_init=view_init)
    else:
        from mitsuba_renderer.renderer import render_pc, render_pcs
        size = 1024
        clip_size = size // 13
        dpc_img, _ = render_pc(sorted_all_traj[0]['dpc'], None, clip_size=clip_size, height=size, width=size)
        goal_dpc_img, _ = render_pc(sorted_all_traj[0]['goal_dpc'], None, clip_size=clip_size, height=size, width=size)
    img_shape = dpc_img.shape
    his_img_plan = []
    for traj_i in range(len(sorted_all_traj)):
        his_us = torch.FloatTensor(his_us_np[traj_i]).cuda().view(-1, args.dimu)
        his_pc_mgoals = agent.vae.decode(his_us, num_points=1000).detach().cpu().numpy()
        if not use_mitsuba:
            his_imgs_mgoals = visualize_point_cloud_batch(his_pc_mgoals, view_init=view_init)
        else:
            his_imgs_mgoals, _ = render_pcs(his_pc_mgoals, size=size, ncol=plan_step, return_list=True)
        for l in range(his_length):
            his_img_plan.append(dpc_img)
            for j in range(plan_step):
                his_img_plan.append(his_imgs_mgoals[l * plan_step + j])
            his_img_plan.append(goal_dpc_img)
    his_img_plan = np.array(his_img_plan).reshape(len(sorted_all_traj), his_length, plan_step + 2, *img_shape)
    all_his_img = []
    if use_mitsuba:
        his_img_plan = his_img_plan * 255.
    if args.adam_iter < 50:
        save_idx = np.arange(0, args.adam_iter)
    else:
        save_idx = np.arange(0, args.adam_iter, args.adam_iter // 50)
    for i, idx in enumerate(save_idx):
        his_img, _ = visualize_all_traj_his(sorted_all_traj, his_img_plan, i, idx, overlay=False, color=(0, 0, 0), demo=True)
        save_grid_his_img = make_grid(his_img[:len(sorted_all_traj)] / 255., ncol=1, padding=0, pad_value=0.)
        all_his_img.append(save_grid_his_img)
    save_numpy_as_video(np.array(all_his_img), osp.join(save_dir, f'plan_traj_hist{epoch}.mp4'), scale=0.4)


def visualize_adam_pc_set(args, agent, epoch, all_traj, sorted_idxes, save_dir, view_init=(140, -90), use_mitsuba=False):
    from tqdm import tqdm
    sorted_all_traj = [all_traj[i] for i in sorted_idxes[:5]]
    all_his_us = [traj['struct_u_his'] for traj in sorted_all_traj]
    n = len(all_his_us)
    plan_step = len(all_his_us[0])
    his_length, _, D = all_his_us[0][0].shape
    if not use_mitsuba:
        dpc_img, goal_dpc_img = visualize_point_cloud_plt(sorted_all_traj[0]['dpc'], view_init=view_init), visualize_point_cloud_plt(
        sorted_all_traj[0]['goal_dpc'], view_init=view_init)
    else:
        from mitsuba_renderer.renderer import render_pc, render_pcs
        size = 1024
        clip_size = size // 13
        dpc_img, _ = render_pc(sorted_all_traj[0]['dpc'], None, clip_size=clip_size, height=size, width=size)
        goal_dpc_img, _ = render_pc(sorted_all_traj[0]['goal_dpc'], None, clip_size=clip_size, height=size, width=size)
    img_shape = dpc_img.shape
    his_img_plan = []
    for traj_i in tqdm(range(n)):
        list_his_us = all_his_us[traj_i]
        for l in tqdm(range(his_length)):
            his_img_plan.append(dpc_img)
            if l == 0:
                img_mgoals = []
                for j in range(plan_step):
                    his_us = torch.FloatTensor(list_his_us[j].reshape(-1, D)).cuda()
                    his_pc_mgoals = agent.vae.decode(his_us, num_points=1000).view(his_length, -1, 3).detach().cpu().numpy()
                    his_pc_mgoals = batch_resample_pc(his_pc_mgoals, 1000)
                    if not use_mitsuba:
                        his_imgs_mgoals = visualize_point_cloud_batch(his_pc_mgoals, view_init=view_init)
                    else:
                        his_imgs_mgoals, _ = render_pcs(his_pc_mgoals, size=size, ncol=plan_step, return_list=True, save_name='test.png')
                    img_mgoals.append(his_imgs_mgoals)
            for j in range(plan_step):
                his_img_plan.append(img_mgoals[j][l])
            his_img_plan.append(goal_dpc_img)
    his_img_plan = np.array(his_img_plan).reshape(len(sorted_all_traj), his_length, plan_step + 2, *img_shape)
    if use_mitsuba:
        his_img_plan = his_img_plan * 255.
    all_his_img = []
    if args.adam_iter < 50:
        save_idx = np.arange(0, args.adam_iter)
    else:
        save_idx = np.arange(0, args.adam_iter, args.adam_iter // 50)
    for i, idx in enumerate(save_idx):

        his_img, _ = visualize_all_traj_his(sorted_all_traj, his_img_plan, i, idx, overlay=False, color=(0, 0, 0), demo=True)
        save_grid_his_img = make_grid(his_img[:len(sorted_all_traj)] / 255., ncol=1, padding=0, pad_value=0.)
        all_his_img.append(save_grid_his_img)
    save_numpy_as_video(np.array(all_his_img), osp.join(save_dir, f'plan_traj_hist{epoch}.mp4'), scale=0.4)

def visualize_neighborhood(agent, u1s, u2s, tids, dpi=200, all_z=None, t_axes=(0, 2)):
    """ u1, u2: B x (dimz + 3), torch"""
    matplotlib.rcParams['figure.dpi'] = dpi
    fig = plt.figure(figsize=(3, 3))
    ax1 = fig.add_subplot(111)
    canvas = FigureCanvas(fig)
    width, height = fig.get_size_inches() * fig.get_dpi()
    width = int(width)
    height = int(height)

    def generate_grid(low_x, max_x, low_y, max_y, n):
        xs = torch.linspace(low_x, max_x, steps=n)
        ys = torch.linspace(low_y, max_y, steps=n)
        xx, yy = torch.meshgrid(xs, ys)
        grid_coor = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)  # (n^2) x 2
        return grid_coor

    n = 200
    z_coor = generate_grid(-6, 6, -6, 6, n)
    if t_axes[1] == 2:
        t_coor = generate_grid(0.2, 0.8, 0.2, 0.8, n)
        t_x, t_y = t_coor[:, [0]], t_coor[:, [1]]
        t_coor = torch.cat([t_x, torch.ones_like(t_x) * 0.0676, t_y], dim=1)
    else:
        t_coor = generate_grid(0.2, 0.8, 0., 0.6, n)
        t_x, t_y = t_coor[:, [0]], t_coor[:, [1]]
        t_coor = torch.cat([t_x, t_y, torch.ones_like(t_x) * 0.5], dim=1)
    np_z_coor, np_t_coor = z_coor.numpy(), t_coor.numpy()
    z_coor, t_coor = z_coor.cuda(), t_coor.cuda()
    tx_idx = t_axes[0] - 3
    ty_idx = t_axes[1] - 3
    t_fixed_axs = 3 - t_axes[0] - t_axes[1]
    t_fixed_idx = t_fixed_axs - 3

    def visualize_z_spaces(z_train, z_eval, titles):
        cmap = plt.get_cmap("tab10")
        images = []
        for idx, title in enumerate(titles):
            if 'z' in title:
                x = np_z_coor[:, 0].reshape(n, n)
                y = np_z_coor[:, 1].reshape(n, n)
            elif 't' in title:
                x = np_t_coor[:, tx_idx].reshape(n, n)
                y = np_t_coor[:, ty_idx].reshape(n, n)

            ax1.axis([x.min(), x.max(), y.max(), y.min()])
            ax1.set_title(title)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            if 'z1' in title or 't1' in title:
                if 'z' in title:
                    q1 = ax1.scatter(z_train[::1, 0], z_train[::1, 1], alpha=0.5, s=0.1, label='train', color=cmap(0))
                    q2 = ax1.scatter(z_eval[::1, 0], z_eval[::1, 1], alpha=0.5, s=0.1, label='eval', color=cmap(1))
                else:
                    q1.remove()
                    q2.remove()
                    q1 = ax1.scatter(z_train[::1, tx_idx], z_train[::1, ty_idx], alpha=0.5, s=0.1, label='train', color=cmap(0))
                    q2 = ax1.scatter(z_eval[::1, tx_idx], z_eval[::1, ty_idx], alpha=0.5, s=0.1, label='eval', color=cmap(1))
            plt.tight_layout()
            # leg = plt.legend(loc='upper left')
            canvas.draw()  # draw the canvas, cache the renderer
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
            images.append(image)
            # leg.remove()
        q1.remove()
        q2.remove()
        return images

    def get_img(feas, selected, title=''):
        """Idx is the current point"""
        feas = feas.detach().cpu().numpy().reshape(n, n)
        best_idx = np.argmax(feas.flatten())

        if 'z' in title:
            x = np_z_coor[:, 0].reshape(n, n)
            y = np_z_coor[:, 1].reshape(n, n)
        elif 't' in title:
            x = np_t_coor[:, tx_idx].reshape(n, n)
            y = np_t_coor[:, ty_idx].reshape(n, n)
        best_x, best_y = x.flatten()[best_idx], y.flatten()[best_idx]
        q1 = ax1.pcolormesh(x, y, feas, cmap=cm.get_cmap('viridis'), vmin=feas.min(), vmax=feas.max())
        ax1.axis([x.min(), x.max(), y.max(), y.min()])

        ax1.set_title(title)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        q2 = ax1.scatter([best_x], [best_y], marker="*", color="r", s=30)
        q3 = ax1.scatter([selected[0].item()], [selected[1].item()], color="green", marker='v', s=20,
                         alpha=0.8)  # idx reversed since fea is reversed.
        plt.tight_layout()

        canvas.draw()  # draw the canvas, cache the renderer

        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        q1.remove()
        q2.remove()
        q3.remove()
        return image

    # Predict each one is just easier
    imgs = []
    with torch.no_grad():
        for (u1, u2, tid) in zip(u1s, u2s, tids):
            tiled_u1, tiled_u2 = u1[None].repeat(len(z_coor), 1), u2[None].repeat(len(z_coor), 1)

            # Change z1
            new_u1 = tiled_u1.clone()
            new_u1[:, :-3] = z_coor
            fea_z1 = agent.feas[tid](new_u1, tiled_u2, eval=True)
            img1 = get_img(fea_z1, u1[:-3], title='z1 landscape')

            # Change z2
            new_u2 = tiled_u2.clone()
            new_u2[:, :-3] = z_coor
            fea_z2 = agent.feas[tid](tiled_u1, new_u2, eval=True)
            img2 = get_img(fea_z2, u2[:-3], title='z2 landscape')

            # Change t1
            new_u1 = tiled_u1.clone()
            new_u1[:, -3:] = t_coor
            # new_u1[:, t_fixed_idx] = tiled_u1[:, t_fixed_idx]
            fea_t1 = agent.feas[tid](new_u1, tiled_u2, eval=True)
            img3 = get_img(fea_t1, [u1[tx_idx], u1[ty_idx]], title='t1 landscape')

            # Change t2
            new_u2 = tiled_u2.clone()
            new_u2[:, -3:] = t_coor
            # new_u2[:, t_fixed_idx] = tiled_u2[:, t_fixed_idx]
            fea_t2 = agent.feas[tid](tiled_u1, new_u2, eval=True)
            img4 = get_img(fea_t2, [u2[tx_idx], u2[ty_idx]], title='t2 landscape')

            if all_z is not None:
                img_overlays = visualize_z_spaces(all_z[0], all_z[1],
                                                  titles=['z1 landscape', 'z2 landscape', 't1 landscape', 't2 landscape'])
                img1 = 0.5 * img1 + 0.5 * img_overlays[0]
                img2 = 0.5 * img2 + 0.5 * img_overlays[1]
                img3 = 0.5 * img3 + 0.5 * img_overlays[2]
                img4 = 0.5 * img4 + 0.5 * img_overlays[3]
            imgs.extend([img1, img2, img3, img4])

    all_imgs = make_grid(np.array(imgs) / 255., ncol=4, padding=5, pad_value=0.5)
    plt.close()
    return all_imgs


def visualize_set_neighborhood(agent, u1s, u2s, tids, dpi=200, all_z=None, t_axes=(0, 2)):
    """ u1, u2: B x (dimz + 3), torch"""
    matplotlib.rcParams['figure.dpi'] = dpi
    fig = plt.figure(figsize=(3, 3))
    ax1 = fig.add_subplot(111)
    canvas = FigureCanvas(fig)
    width, height = fig.get_size_inches() * fig.get_dpi()
    width = int(width)
    height = int(height)

    def generate_grid(low_x, max_x, low_y, max_y, n):
        xs = torch.linspace(low_x, max_x, steps=n)
        ys = torch.linspace(low_y, max_y, steps=n)
        xx, yy = torch.meshgrid(xs, ys)
        grid_coor = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)  # (n^2) x 2
        return grid_coor
    n = 200
    z_coor = generate_grid(-6, 6, -6, 6, n)
    if t_axes[1] == 2:
        t_coor = generate_grid(0.2, 0.8, 0.2, 0.8, n)
        t_x, t_y = t_coor[:, [0]], t_coor[:, [1]]
        t_coor = torch.cat([t_x, torch.ones_like(t_x) * 0.0676, t_y], dim=1)
    else:
        t_coor = generate_grid(0.2, 0.8, 0., 0.6, n)
        t_x, t_y = t_coor[:, [0]], t_coor[:, [1]]
        t_coor = torch.cat([t_x, t_y, torch.ones_like(t_x) * 0.5], dim=1)
    np_z_coor, np_t_coor = z_coor.numpy(), t_coor.numpy()
    z_coor, t_coor = z_coor.cuda(), t_coor.cuda()
    tx_idx = t_axes[0] - 3
    ty_idx = t_axes[1] - 3
    t_fixed_axs = 3 - t_axes[0] - t_axes[1]
    t_fixed_idx = t_fixed_axs - 3

    def visualize_z_spaces(z_train, titles):
        cmap = plt.get_cmap("tab10")
        images = []
        for idx, title in enumerate(titles):
            if 'z' in title:
                x = np_z_coor[:, 0].reshape(n, n)
                y = np_z_coor[:, 1].reshape(n, n)
            elif 't' in title:
                x = np_t_coor[:, tx_idx].reshape(n, n)
                y = np_t_coor[:, ty_idx].reshape(n, n)

            ax1.axis([x.min(), x.max(), y.max(), y.min()])
            ax1.set_title(title)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            if 'z1' in title or 't1' in title:
                if 'z' in title:
                    q1 = ax1.scatter(z_train[::1, 0], z_train[::1, 1], alpha=0.5, s=0.1, label='train', color=cmap(0))
                else:
                    q1.remove()
                    q1 = ax1.scatter(z_train[::1, tx_idx], z_train[::1, ty_idx], alpha=0.5, s=0.1, label='train', color=cmap(0))
            plt.tight_layout()
            # leg = plt.legend(loc='upper left')
            canvas.draw()  # draw the canvas, cache the renderer
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
            images.append(image)
            # leg.remove()
        q1.remove()
        return images

    def get_img(feas, selected, title=''):
        """Idx is the current point"""
        feas = feas.detach().cpu().numpy().reshape(n, n)
        best_idx = np.argmax(feas.flatten())

        if 'z' in title:
            x = np_z_coor[:, 0].reshape(n, n)
            y = np_z_coor[:, 1].reshape(n, n)
        elif 't' in title:
            x = np_t_coor[:, tx_idx].reshape(n, n)
            y = np_t_coor[:, ty_idx].reshape(n, n)
        best_x, best_y = x.flatten()[best_idx], y.flatten()[best_idx]
        q1 = ax1.pcolormesh(x, y, feas, cmap=cm.get_cmap('viridis'), vmin=0., vmax=1.)
        ax1.axis([x.min(), x.max(), y.max(), y.min()])

        ax1.set_title(title)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        q2 = ax1.scatter([best_x], [best_y], marker="*", color="r", s=30)
        q3 = ax1.scatter([selected[0].item()], [selected[1].item()], color="green", marker='v', s=20,
                         alpha=0.8)  # idx reversed since fea is reversed.
        plt.tight_layout()

        canvas.draw()  # draw the canvas, cache the renderer

        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        q1.remove()
        q2.remove()
        q3.remove()
        return image

    # Predict each one is just easier
    imgs = []
    dimz = 2
    with torch.no_grad():
        for (u1, u2, tid) in zip(u1s, u2s, tids):
            u1, u2 = u1.flatten(), u2.flatten()
            tiled_u1, tiled_u2 = u1[None].repeat(len(z_coor), 1), u2[None].repeat(len(z_coor), 1)
            # print(tiled_u1.shape, tiled_u2.shape)

            # Change z1
            new_u1 = tiled_u1.clone()
            new_u1[:, :dimz] = z_coor
            fea_z1 = agent.feas[tid](torch.hstack([new_u1, tiled_u2]), eval=True)
            img1 = get_img(fea_z1, u1[:-3], title='z1 landscape')

            # Change z2
            new_u2 = tiled_u2.clone()
            new_u2[:, :dimz] = z_coor
            fea_z2 = agent.feas[tid](torch.hstack([tiled_u1, new_u2]), eval=True)
            img2 = get_img(fea_z2, u2[:-3], title='z2 landscape')

            # Change t1
            new_u1 = tiled_u1.clone()
            new_u1[:, dimz:dimz + 3] = t_coor
            # new_u1[:, t_fixed_idx] = tiled_u1[:, t_fixed_idx]
            fea_t1 = agent.feas[tid](torch.hstack([new_u1, tiled_u2]), eval=True)
            img3 = get_img(fea_t1, [u1[tx_idx], u1[ty_idx]], title='t1 landscape')

            # Change t2
            new_u2 = tiled_u2.clone()
            new_u2[:, dimz:dimz + 3] = t_coor
            # new_u2[:, t_fixed_idx] = tiled_u2[:, t_fixed_idx]
            fea_t2 = agent.feas[tid](torch.hstack([tiled_u1, new_u2]), eval=True)
            img4 = get_img(fea_t2, [u2[tx_idx + 5], u2[ty_idx + 5]], title='t2 landscape')

            if all_z is not None:
                img_overlays = visualize_z_spaces(all_z,
                                                  titles=['z1 landscape', 'z2 landscape', 't1 landscape', 't2 landscape'])
                img1 = 0.5 * img1 + 0.5 * img_overlays[0]
                img2 = 0.5 * img2 + 0.5 * img_overlays[1]
                img3 = 0.5 * img3 + 0.5 * img_overlays[2]
                img4 = 0.5 * img4 + 0.5 * img_overlays[3]
            imgs.extend([img1, img2, img3, img4])

    all_imgs = make_grid(np.array(imgs) / 255., ncol=4, padding=5, pad_value=0.5)
    plt.close()
    return all_imgs

def visualize_set_reward_neighborhood(agent, u1s, u2s, tids, dpi=200, all_z=None, t_axes=(0, 2)): # Debug
    """ u1, u2: B x (dimz + 3), torch"""
    matplotlib.rcParams['figure.dpi'] = dpi
    fig = plt.figure(figsize=(3, 3))
    ax1 = fig.add_subplot(111)
    canvas = FigureCanvas(fig)
    width, height = fig.get_size_inches() * fig.get_dpi()
    width = int(width)
    height = int(height)

    def generate_grid(low_x, max_x, low_y, max_y, n):
        xs = torch.linspace(low_x, max_x, steps=n)
        ys = torch.linspace(low_y, max_y, steps=n)
        xx, yy = torch.meshgrid(xs, ys)
        grid_coor = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)  # (n^2) x 2
        return grid_coor
    n = 200
    z_coor = generate_grid(-6, 6, -6, 6, n)
    if t_axes[1] == 2:
        t_coor = generate_grid(0.2, 0.8, 0.2, 0.8, n)
        t_x, t_y = t_coor[:, [0]], t_coor[:, [1]]
        t_coor = torch.cat([t_x, torch.ones_like(t_x) * 0.0676, t_y], dim=1)
    else:
        t_coor = generate_grid(0.2, 0.8, 0., 0.6, n)
        t_x, t_y = t_coor[:, [0]], t_coor[:, [1]]
        t_coor = torch.cat([t_x, t_y, torch.ones_like(t_x) * 0.5], dim=1)
    np_z_coor, np_t_coor = z_coor.numpy(), t_coor.numpy()
    z_coor, t_coor = z_coor.cuda(), t_coor.cuda()
    tx_idx = t_axes[0] - 3
    ty_idx = t_axes[1] - 3
    t_fixed_axs = 3 - t_axes[0] - t_axes[1]
    t_fixed_idx = t_fixed_axs - 3

    def visualize_z_spaces(z_train, titles):
        cmap = plt.get_cmap("tab10")
        images = []
        for idx, title in enumerate(titles):
            if 'z' in title:
                x = np_z_coor[:, 0].reshape(n, n)
                y = np_z_coor[:, 1].reshape(n, n)
            elif 't' in title:
                x = np_t_coor[:, tx_idx].reshape(n, n)
                y = np_t_coor[:, ty_idx].reshape(n, n)

            ax1.axis([x.min(), x.max(), y.max(), y.min()])
            ax1.set_title(title)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            if 'z1' in title or 't1' in title:
                if 'z' in title:
                    q1 = ax1.scatter(z_train[::1, 0], z_train[::1, 1], alpha=0.5, s=0.1, label='train', color=cmap(0))
                else:
                    q1.remove()
                    q1 = ax1.scatter(z_train[::1, tx_idx], z_train[::1, ty_idx], alpha=0.5, s=0.1, label='train', color=cmap(0))
            plt.tight_layout()
            # leg = plt.legend(loc='upper left')
            canvas.draw()  # draw the canvas, cache the renderer
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
            images.append(image)
            # leg.remove()
        q1.remove()
        return images

    def get_img(feas, selected, title=''):
        """Idx is the current point"""
        feas = feas.detach().cpu().numpy().reshape(n, n)
        best_idx = np.argmin(feas.flatten())

        if 'z' in title:
            x = np_z_coor[:, 0].reshape(n, n)
            y = np_z_coor[:, 1].reshape(n, n)
        elif 't' in title:
            x = np_t_coor[:, tx_idx].reshape(n, n)
            y = np_t_coor[:, ty_idx].reshape(n, n)
        best_x, best_y = x.flatten()[best_idx], y.flatten()[best_idx]
        q1 = ax1.pcolormesh(x, y, feas, cmap=cm.get_cmap('viridis'), vmin=0., vmax=0.01)
        ax1.axis([x.min(), x.max(), y.max(), y.min()])

        ax1.set_title(title)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        q2 = ax1.scatter([best_x], [best_y], marker="*", color="r", s=30)
        q3 = ax1.scatter([selected[0].item()], [selected[1].item()], color="green", marker='v', s=20,
                         alpha=0.8)  # idx reversed since fea is reversed.
        plt.tight_layout()

        canvas.draw()  # draw the canvas, cache the renderer

        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        q1.remove()
        q2.remove()
        q3.remove()
        return image

    # Predict each one is just easier
    imgs = []
    dimz = 2
    with torch.no_grad():
        for (u1, u2, tid) in zip(u1s, u2s, tids):
            u1, u2 = u1.flatten(), u2.flatten()
            tiled_u1, tiled_u2 = u1[None].repeat(len(z_coor), 1), u2[None].repeat(len(z_coor), 1)
            B, D = len(z_coor), 5
            # Change z1
            new_u1 = tiled_u1.clone()
            new_u1[:, :dimz] = z_coor
            fea_z1 = agent.reward_predictor.predict(new_u1.view(B, -1, D), tiled_u2.view(B, -1, D)).sum(dim=-1)
            img1 = get_img(fea_z1, u1[:-3], title='z1 landscape')

            # Change z2
            new_u2 = tiled_u2.clone()
            new_u2[:, :dimz] = z_coor
            fea_z2 = agent.reward_predictor.predict(tiled_u1.view(B, -1, D), new_u2.view(B, -1, D)).sum(dim=-1)
            img2 = get_img(fea_z2, u2[:-3], title='z2 landscape')

            # Change t1
            new_u1 = tiled_u1.clone()
            new_u1[:, dimz:dimz + 3] = t_coor
            # new_u1[:, t_fixed_idx] = tiled_u1[:, t_fixed_idx]
            fea_t1 = agent.reward_predictor.predict(new_u1.view(B, -1, D), tiled_u2.view(B, -1, D)).sum(dim=-1)
            img3 = get_img(fea_t1, [u1[tx_idx], u1[ty_idx]], title='t1 landscape')

            # Change t2
            new_u2 = tiled_u2.clone()
            new_u2[:, dimz:dimz + 3] = t_coor
            # new_u2[:, t_fixed_idx] = tiled_u2[:, t_fixed_idx]
            fea_t2 = agent.reward_predictor.predict(tiled_u1.view(B, -1, D), new_u2.view(B, -1, D)).sum(dim=-1)
            img4 = get_img(fea_t2, [u2[tx_idx + 5], u2[ty_idx + 5]], title='t2 landscape')

            if all_z is not None:
                img_overlays = visualize_z_spaces(all_z,
                                                  titles=['z1 landscape', 'z2 landscape', 't1 landscape', 't2 landscape'])
                img1 = 0.5 * img1 + 0.5 * img_overlays[0]
                img2 = 0.5 * img2 + 0.5 * img_overlays[1]
                img3 = 0.5 * img3 + 0.5 * img_overlays[2]
                img4 = 0.5 * img4 + 0.5 * img_overlays[3]
            imgs.extend([img1, img2, img3, img4])

    all_imgs = make_grid(np.array(imgs) / 255., ncol=4, padding=5, pad_value=0.5)
    plt.close()
    return all_imgs
