import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 示例数据（替换为你的实际数据）
# pred = np.array(...)  # 形状 (n, 3)
# true = np.array(...)        # 形状 (n, 3)

# 生成示例数据（正弦曲线）
def data_plot(pred, true, save_path):
    t = np.linspace(0, 4*np.pi, 100)
    # preds_ = np.column_stack([np.sin(t), np.cos(t), t/4])
    # true = np.column_stack([np.sin(t + 0.5), np.cos(t + 0.3), t/4 + 0.2])

    # 创建3D图像
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制预测轨迹（红色）
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], 
            'r-', linewidth=2, label='Predicted Trajectory')

    # 绘制真实轨迹（蓝色）
    ax.plot(true[:, 0], true[:, 1], true[:, 2], 
            'b--', linewidth=2, label='True Trajectory')

    # 添加标签和图例
    ax.set_xlabel('X Axis', fontsize=12)
    ax.set_ylabel('Y Axis', fontsize=12)
    ax.set_zlabel('Z Axis', fontsize=12)
    ax.set_title('3D Trajectory Comparison', fontsize=14)
    ax.legend(fontsize=10)

    # 设置视角（可选）
    ax.view_init(elev=20, azim=-35)  # 仰角20度，方位角-35度

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')