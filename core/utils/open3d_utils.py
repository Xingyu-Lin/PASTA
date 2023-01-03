import socket

if 'autobot' not in socket.gethostname() and 'seuss' not in socket.gethostname() and 'compute' not in socket.gethostname():
    import open3d as o3d

import matplotlib
import matplotlib.cm
import numpy as np

cmap = matplotlib.cm.get_cmap('Spectral')
# Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
cam_center = [0.5, 0.3, 0.6]  # look_at target
cam_eye = [0.5, 0.6, 1.0]  # camera position
cam_up = [0, -0.5, 0]  # camera orientation


def draw_unit_box():
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[0, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def draw_frame():
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])


def set_camera(vis, camera_path):
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters(camera_path)
    ctr.convert_from_pinhole_camera_parameters(parameters)


def visualize_point_cloud(pcls):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(draw_frame())
    vis.add_geometry(draw_unit_box())
    # if len(pcls.shape) == 2:
    #     pcls = np.expand_dims(pcls, 0)
    pcds = []
    for idx, pcl in enumerate(pcls):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
        color = cmap(idx / len(pcls))[:3]
        pcd.colors = o3d.utility.Vector3dVector(np.ones_like(pcl) * color)
        pcds.append(pcd)
        vis.add_geometry(pcd)
    set_camera(vis, 'core/utils/camera_info.json')
    vis.run()


def visualize_point_cloud_plt(pcl, view_init=(140, -90)):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    matplotlib.rcParams['figure.dpi'] = 100
    fig = plt.figure(figsize=(3, 3))
    canvas = FigureCanvas(fig)
    cmap = plt.get_cmap("tab10")
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], s=5, color=cmap(0))
    # ax1.view_init(elev=130, azim=270)
    #
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)
    ax1.view_init(*view_init)
    # ax1.set_xlim3d(0.2, 0.8)
    # ax1.set_ylim3d(0, 0.6)
    # ax1.set_zlim3d(0.2, 0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    plt.tight_layout()
    canvas.draw()  # draw the canvas, cache the renderer
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.close()
    return image


def visualize_point_cloud_batch(lst_pcl, dpi=100, view_init=(140, -90)):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    matplotlib.rcParams['figure.dpi'] = dpi
    fig = plt.figure(figsize=(3, 3))
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
    ax1.view_init(*(view_init))
    # ax1.set_xlim3d(0.2, 0.8)
    # ax1.set_ylim3d(0, 0.6)
    # ax1.set_zlim3d(0.2, 0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    plt.tight_layout()

    for pcl in lst_pcl:
        q = ax1.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], s=5, color=cmap(0))
        canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        images.append(image)
        q.remove()
    plt.close()
    return images

def visualize_pcl_policy_input(pcl, tool_pcl, goal_pcl, view_init=(140, -90)):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    matplotlib.rcParams['figure.dpi'] = 100
    fig = plt.figure(figsize=(3, 3))
    canvas = FigureCanvas(fig)
    cmap = plt.get_cmap("tab10")
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], s=5, color=cmap(0))
    if tool_pcl is not None:
        ax1.scatter(tool_pcl[:, 0], tool_pcl[:, 1], tool_pcl[:, 2], s=5, color=cmap(1))
    ax1.scatter(goal_pcl[:, 0], goal_pcl[:, 1], goal_pcl[:, 2], s=5, color=cmap(2))
    # ax1.view_init(elev=130, azim=270)
    #
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)
    ax1.view_init(*view_init)
    # ax1.set_xlim3d(0.2, 0.8)
    # ax1.set_ylim3d(0, 0.6)
    # ax1.set_zlim3d(0.2, 0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    plt.tight_layout()
    canvas.draw()  # draw the canvas, cache the renderer
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.close()
    return image