import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import imageio


class Visualizer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.vis_dir = os.path.join(base_path, 'visualizations')
        self.video_dir = os.path.join(base_path, 'videos')
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        self._frames = []  # temporary list of frames for current episode

    def _draw_topdown(self, scan, odom_x, odom_y, yaw, goal_x, goal_y, max_range=3.5):
        # scan: full 360-array or sampled; assume it's a 1D numpy array of ranges
        scan = np.array(scan)
        angles = np.linspace(-math.pi, math.pi, len(scan), endpoint=False)
        xs = scan * np.cos(angles) + odom_x
        ys = scan * np.sin(angles) + odom_y

        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlim(odom_x - max_range - 0.5, odom_x + max_range + 0.5)
        ax.set_ylim(odom_y - max_range - 0.5, odom_y + max_range + 0.5)
        ax.set_aspect('equal')
        ax.scatter(xs, ys, s=5, c='r', alpha=0.6, label='LIDAR')

        # Robot
        ax.plot(odom_x, odom_y, marker=(3, 0, math.degrees(yaw)-90), markersize=15, color='blue')
        ax.scatter([odom_x], [odom_y], c='blue')

        # Goal
        ax.scatter([goal_x], [goal_y], c='green', marker='*', s=120, label='Goal')

        ax.legend(loc='upper right')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Top-down (LIDAR + Goal)')
        ax.grid(True, alpha=0.3)
        return fig

    def save_frame(self, episode, step, scan, odom_x, odom_y, yaw, goal_x, goal_y):
        fig = self._draw_topdown(scan, odom_x, odom_y, yaw, goal_x, goal_y)
        fname = os.path.join(self.vis_dir, f'ep{episode:04d}_step{step:04d}.png')
        fig.savefig(fname, dpi=100)
        plt.close(fig)
        self._frames.append(fname)
        return fname

    def finalize_episode(self, episode, fps=10):
        if not self._frames:
            return None
        video_path = os.path.join(self.video_dir, f'episode_{episode:04d}.mp4')
        with imageio.get_writer(video_path, fps=fps) as writer:
            for f in self._frames:
                img = imageio.imread(f)
                writer.append_data(img)
        # clear frames list and optionally remove frame files
        self._frames = []
        return video_path

    def clear(self):
        self._frames = []
