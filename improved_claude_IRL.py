import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import csv
import os  # 用于构建输出文件夹并保存图像


warnings.filterwarnings('ignore')


class MaxEntIRL:
    """最大熵逆强化学习for驾驶风格识别 - 改进版本"""

    def __init__(self, feature_dim=4, learning_rate=0.05, lam=0.01):
        """
        初始化IRL模型
        """
        self.feature_dim = feature_dim
        self.lr = learning_rate
        self.lam = lam  # 正则化参数
        
        # 初始化θ为小的随机值（而不是均匀分布）
        self.theta = np.random.normal(0, 0.05, size=feature_dim)

        # Adam优化器参数
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.pm = None  # momentum
        self.pv = None  # velocity

        # 记录训练过程
        self.theta_history = []
        self.feature_diff_history = []
        self.position_diff_history = []
        self.loss_history = []
        self.grad_history = []
        self.likelihood_history = []

        # 车辆数据统计
        self.vehicle_v_max = None
        self.vehicle_v_min = None
        self.vehicle_a_max = None
        self.vehicle_a_min = None
        
        # 特征归一化参数
        self.max_features = None
        self.min_features = None

    def load_vehicle_data(self, filepath):
        """
        加载CSV数据
        """
        print("正在加载数据...")
        df = pd.read_csv(filepath)

        print(f"\n数据概览:")
        print(f"  帧数: {len(df)}")
        print(f"  车辆ID: {df['id'].iloc[0]}")

        # 计算速度和加速度的实际范围
        self.vehicle_v_max = df['xVelocity'].max()
        self.vehicle_v_min = df['xVelocity'].min()
        self.vehicle_a_max = df['xAcceleration'].max()
        self.vehicle_a_min = df['xAcceleration'].min()

        print(f"\n车辆实际运动参数:")
        print(f"  速度范围: [{self.vehicle_v_min:.2f}, {self.vehicle_v_max:.2f}] m/s")
        print(f"  加速度范围: [{self.vehicle_a_min:.2f}, {self.vehicle_a_max:.2f}] m/s²")

        return df

    def preprocess_vehicle_data(self, vehicle_data):
        """
        预处理数据
        """
        # 计算时间（100帧=4秒，25Hz）
        vehicle_data['time'] = vehicle_data.index * 0.04

        # 计算jerk
        dt = 0.04
        vehicle_data['jerk'] = vehicle_data['xAcceleration'].diff() / dt
        vehicle_data['jerk'] = vehicle_data['jerk'].fillna(0)

        # 处理dhw
        vehicle_data['dhw'] = vehicle_data['dhw'].fillna(100)

        return vehicle_data

    def create_trajectory_from_data(self, vehicle_data):
        """
        创建轨迹
        """
        trajectory = {
            'x': vehicle_data['x'].values,
            'v': vehicle_data['xVelocity'].values,
            'acc': vehicle_data['xAcceleration'].values,
            'jerk': vehicle_data['jerk'].values,
            't': vehicle_data['time'].values,
            'dhw': vehicle_data['dhw'].values
        }

        return trajectory

    def extract_features(self, trajectory, normalize=True):
        """
        提取轨迹特征
        """
        # f1: 平均速度（使用绝对值）
        f1 = np.mean(np.abs(trajectory['v']))

        # f2: 平均加速度绝对值
        f2 = np.mean(np.abs(trajectory['acc']))

        # f3: 平均jerk绝对值
        f3 = np.mean(np.abs(trajectory['jerk']))

        # f4: 最小车距
        f4 = np.min(trajectory['dhw']) if len(trajectory['dhw']) > 0 else 100

        features = np.array([f1, f2, f3, f4])
        
        if normalize and self.max_features is not None:
            # 使用全局归一化参数
            for i in range(len(features)):
                if self.max_features[i] > 0:
                    features[i] /= self.max_features[i]
                    
        return features

    def compute_feature_statistics(self, all_trajectories):
        """
        计算所有轨迹的特征统计信息用于归一化
        """
        all_features = []
        for traj in all_trajectories:
            features = self.extract_features(traj, normalize=False)
            all_features.append(features)
        
        all_features = np.array(all_features)
        self.max_features = np.max(all_features, axis=0)
        self.min_features = np.min(all_features, axis=0)
        
        # 确保max_features中没有零值
        self.max_features = np.maximum(self.max_features, 1e-6)
        
        print(f"特征统计:")
        print(f"  最大值: {self.max_features}")
        print(f"  最小值: {self.min_features}")

    def generate_candidate_trajectories_grid(self, initial_state, T=4.0, dt=0.04):
        """
        网格方式生成候选轨迹
        """
        x0, v0, a0 = initial_state
        t = np.arange(0, T, dt)

        # 速度网格：0.5 m/s间隔
        v_end_candidates = []
        v_current = v0
        while v_current <= self.vehicle_v_max:
            v_end_candidates.append(v_current)
            v_current += 0.5

        v_current = v0 - 0.5
        while v_current >= self.vehicle_v_min:
            v_end_candidates.insert(0, v_current)
            v_current -= 0.5

        v_end_grid = sorted(list(set(v_end_candidates)))

        # 加速度网格：0.2 m/s²间隔
        a_end_candidates = []
        a_current = a0
        while a_current <= self.vehicle_a_max:
            a_end_candidates.append(a_current)
            a_current += 0.2

        a_current = a0 - 0.2
        while a_current >= self.vehicle_a_min:
            a_end_candidates.insert(0, a_current)
            a_current -= 0.2

        a_end_grid = sorted(list(set(a_end_candidates)))

        trajectories = []

        for v_end in v_end_grid:
            for a_end in a_end_grid:
                coeffs = self._solve_polynomial_coefficients(x0, v0, a0, v_end, a_end, T)

                if coeffs is None:
                    continue

                x = np.zeros(len(t))
                v = np.zeros(len(t))
                acc = np.zeros(len(t))
                jerk = np.zeros(len(t))

                for i, ti in enumerate(t):
                    x[i] = coeffs[0] + coeffs[1] * ti + coeffs[2] * ti ** 2 + coeffs[3] * ti ** 3 + coeffs[4] * ti ** 4
                    v[i] = coeffs[1] + 2 * coeffs[2] * ti + 3 * coeffs[3] * ti ** 2 + 4 * coeffs[4] * ti ** 3
                    acc[i] = 2 * coeffs[2] + 6 * coeffs[3] * ti + 12 * coeffs[4] * ti ** 2
                    jerk[i] = 6 * coeffs[3] + 24 * coeffs[4] * ti

                dhw = np.full_like(t, 30)

                trajectories.append({
                    'x': x, 'v': v, 'acc': acc, 'jerk': jerk,
                    't': t, 'dhw': dhw
                })

        return trajectories

    def _solve_polynomial_coefficients(self, x0, v0, a0, vf, af, T):
        """
        求解四次多项式系数
        """
        coeff_a0 = x0
        coeff_a1 = v0
        coeff_a2 = a0 / 2

        A = np.array([[3 * T ** 2, 4 * T ** 3], [6 * T, 12 * T ** 2]])
        b = np.array([vf - v0 - a0 * T, af - a0])

        try:
            solution = np.linalg.solve(A, b)
            coeff_a3 = solution[0]
            coeff_a4 = solution[1]
            return [coeff_a0, coeff_a1, coeff_a2, coeff_a3, coeff_a4]
        except:
            return None

    def compute_trajectory_reward(self, features):
        """计算奖励"""
        return np.dot(self.theta, features)

    def compute_trajectory_probabilities(self, trajectories):
        """
        计算所有轨迹的概率分布 - 改进版本
        """
        rewards = []
        features_list = []
        
        for traj in trajectories:
            features = self.extract_features(traj)
            reward = self.compute_trajectory_reward(features)
            rewards.append(reward)
            features_list.append(features)
        
        # 数值稳定性处理
        rewards = np.array(rewards)
        rewards = np.clip(rewards, -10, 10)  # 防止数值溢出
        
        # softmax概率计算
        exp_rewards = np.exp(rewards - np.max(rewards))  # 减去最大值提高数值稳定性
        probabilities = exp_rewards / np.sum(exp_rewards)
        
        return probabilities, features_list, rewards

    def train_step_improved(self, expert_trajectory, candidate_trajectories):
        """
        改进的单步训练 - 参考general_IRL.py
        """
        if len(candidate_trajectories) == 0:
            return 0, 0, 0, 0

        # 计算专家轨迹特征
        expert_features = self.extract_features(expert_trajectory)
        
        # 计算所有候选轨迹的概率和特征
        probabilities, features_list, rewards = self.compute_trajectory_probabilities(candidate_trajectories)
        
        # 计算特征期望 E[f(ζ)] = Σ P(ζi|θ) * f(ζi)
        feature_expectation = np.zeros(self.feature_dim)
        for i, features in enumerate(features_list):
            feature_expectation += probabilities[i] * features
        
        # 计算梯度 (包括正则化项)
        gradient = expert_features - feature_expectation - 2 * self.lam * self.theta
        
        # Adam优化器更新
        if self.pm is None:
            self.pm = np.zeros_like(gradient)
            self.pv = np.zeros_like(gradient)
        
        # 更新动量
        self.pm = self.beta1 * self.pm + (1 - self.beta1) * gradient
        self.pv = self.beta2 * self.pv + (1 - self.beta2) * (gradient * gradient)
        
        # 偏差校正
        iteration = len(self.theta_history) + 1
        mhat = self.pm / (1 - self.beta1 ** iteration)
        vhat = self.pv / (1 - self.beta2 ** iteration)
        
        # 更新参数
        update_vec = mhat / (np.sqrt(vhat) + self.eps)
        self.theta += self.lr * update_vec
        
        # 计算指标
        feature_diff = np.linalg.norm(gradient)
        
        # 找到最优轨迹（概率最高的）
        best_idx = np.argmax(probabilities)
        best_traj = candidate_trajectories[best_idx]
        position_diff = abs(expert_trajectory['x'][-1] - best_traj['x'][-1])
        
        # 计算似然
        likelihood = -np.log(probabilities[best_idx] + 1e-10)
        
        # 计算损失（负对数似然 + 正则化）
        loss = likelihood + self.lam * np.dot(self.theta, self.theta)
        
        return feature_diff, position_diff, likelihood, loss

    def train_improved(self, expert_trajectories, n_iterations=2000):
        """
        改进的训练函数 - 参考general_IRL.py的训练策略
        """
        print(f"\n开始改进训练 (共{n_iterations}轮):")
        print(f"初始θ: [{', '.join([f'{t:.4f}' for t in self.theta])}]")
        print(f"学习率: {self.lr}, 正则化参数: {self.lam}")
        
        # 预先生成所有候选轨迹并计算特征统计
        print("预处理候选轨迹...")
        all_candidate_trajectories = []
        for expert_traj in expert_trajectories:
            initial_state = [
                expert_traj['x'][0],
                expert_traj['v'][0],
                expert_traj['acc'][0]
            ]
            candidates = self.generate_candidate_trajectories_grid(initial_state, T=4.0, dt=0.04)
            all_candidate_trajectories.extend(candidates)
        
        # 计算特征统计用于归一化
        all_trajectories = expert_trajectories + all_candidate_trajectories
        self.compute_feature_statistics(all_trajectories)
        
        # 创建训练日志
        with open(r'D:\桌面\rzwj.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['iteration', 'theta_1', 'theta_2', 'theta_3', 'theta_4', 
                               'feature_diff', 'position_diff', 'likelihood', 'loss', 'gradient_norm'])
        
        # 训练循环
        for iteration in range(n_iterations):
            total_feature_diff = 0
            total_position_diff = 0
            total_likelihood = 0
            total_loss = 0
            total_gradient_norm = 0
            valid_count = 0

            for expert_traj in expert_trajectories:
                initial_state = [
                    expert_traj['x'][0],
                    expert_traj['v'][0],
                    expert_traj['acc'][0]
                ]
                
                candidate_trajectories = self.generate_candidate_trajectories_grid(
                    initial_state, T=4.0, dt=0.04
                )
                
                if len(candidate_trajectories) == 0:
                    continue
                
                f_diff, p_diff, likelihood, loss = self.train_step_improved(
                    expert_traj, candidate_trajectories
                )
                
                if f_diff > 0:
                    total_feature_diff += f_diff
                    total_position_diff += p_diff
                    total_likelihood += likelihood
                    total_loss += loss
                    total_gradient_norm += f_diff
                    valid_count += 1

            if valid_count > 0:
                avg_feature_diff = total_feature_diff / valid_count
                avg_position_diff = total_position_diff / valid_count
                avg_likelihood = total_likelihood / valid_count
                avg_loss = total_loss / valid_count
                avg_gradient_norm = total_gradient_norm / valid_count

                # 记录历史
                self.theta_history.append(self.theta.copy())
                self.feature_diff_history.append(avg_feature_diff)
                self.position_diff_history.append(avg_position_diff)
                self.loss_history.append(avg_loss)
                self.grad_history.append(avg_gradient_norm)
                self.likelihood_history.append(avg_likelihood)

                # 写入训练日志
                with open(r'D:\桌面\rzwj.csv', 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([iteration + 1, self.theta[0], self.theta[1], self.theta[2], self.theta[3],
                                       avg_feature_diff, avg_position_diff, avg_likelihood, avg_loss, avg_gradient_norm])

                # 每10轮显示一次详细信息
                if iteration % 10 == 0:
                    print(f"\n轮次 {iteration:3d}:")
                    print(f"  θ = [{', '.join([f'{t:.4f}' for t in self.theta])}]")
                    print(f"  Loss = {avg_loss:.6f}")
                    print(f"  似然 = {avg_likelihood:.6f}")
                    print(f"  特征差异 = {avg_feature_diff:.6f}")
                    print(f"  梯度范数 = {avg_gradient_norm:.6f}")
                
                # 每50轮显示一次进度
                elif iteration % 5 == 0:
                    print(f"轮次 {iteration:3d}: Loss={avg_loss:.6f}, 梯度={avg_gradient_norm:.6f}")

        print("\n训练完成!")

    def normalize_theta(self):
        """归一化θ"""
        theta_sum = np.sum(np.abs(self.theta))
        if theta_sum > 0:
            self.theta = np.abs(self.theta) / theta_sum
        return self.theta

    def plot_convergence_improved(self):
        """
        绘制改进的收敛曲线
        """
        if len(self.theta_history) == 0:
            print("没有训练数据")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # θ收敛曲线
        ax1 = axes[0, 0]
        theta_array = np.array(self.theta_history)
        iterations = range(len(theta_array))

        labels = ['θ₁ (速度)', 'θ₂ (舒适性)', 'θ₃ (平顺性)', 'θ₄ (安全性)']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i in range(self.feature_dim):
            ax1.plot(iterations, theta_array[:, i], label=labels[i],
                     linewidth=2, color=colors[i])

        ax1.set_xlabel('迭代次数', fontsize=12)
        ax1.set_ylabel('θ值', fontsize=12)
        ax1.set_title('θ参数收敛曲线', fontsize=14)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 损失曲线
        ax2 = axes[0, 1]
        ax2.plot(self.loss_history, 'b-', linewidth=2)
        ax2.set_xlabel('迭代次数', fontsize=12)
        ax2.set_ylabel('总损失', fontsize=12)
        ax2.set_title('训练损失曲线', fontsize=14)
        ax2.grid(True, alpha=0.3)

        # 似然曲线
        ax3 = axes[1, 0]
        ax3.plot(self.likelihood_history, 'g-', linewidth=2)
        ax3.set_xlabel('迭代次数', fontsize=12)
        ax3.set_ylabel('负对数似然', fontsize=12)
        ax3.set_title('似然收敛曲线', fontsize=14)
        ax3.grid(True, alpha=0.3)

        # 梯度范数曲线
        ax4 = axes[1, 1]
        ax4.plot(self.grad_history, 'r-', linewidth=2)
        ax4.set_xlabel('迭代次数', fontsize=12)
        ax4.set_ylabel('梯度范数', fontsize=12)
        ax4.set_title('梯度收敛曲线', fontsize=14)
        ax4.grid(True, alpha=0.3)

        # 添加收敛信息
        if len(self.loss_history) > 0:
            ax2.text(0.02, 0.98, f'初始Loss: {self.loss_history[0]:.6f}\n最终Loss: {self.loss_history[-1]:.6f}',
                     transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        # 在当前脚本所在目录下创建 outputs 文件夹
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'training_convergence.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    # 使用改进的参数
    irl = MaxEntIRL(feature_dim=4, learning_rate=0.05, lam=0.01)

    csv_file = input("输入CSV路径 (按Enter使用默认): ").strip()
    if not csv_file:
        csv_file = "irlcs.csv"

    try:
        print("=" * 50)
        vehicle_data = irl.load_vehicle_data(csv_file)
        vehicle_data = irl.preprocess_vehicle_data(vehicle_data)

        expert_trajectory = irl.create_trajectory_from_data(vehicle_data)
        expert_trajectories = [expert_trajectory]

        print("\n专家轨迹统计:")
        expert_features_raw = irl.extract_features(expert_trajectory, normalize=False)
        print(f"  原始特征: {expert_features_raw}")

        print("\n" + "=" * 50)
        # 使用改进的训练方法
        irl.train_improved(expert_trajectories, n_iterations=2000)

        # 绘制收敛曲线
        print("\n绘制收敛曲线...")
        irl.plot_convergence_improved()

        # 归一化
        theta_normalized = irl.normalize_theta()

        print("\n" + "=" * 50)
        print("改进训练完成！")
        print(f"\n最终θ (未归一化):")
        theta_final = irl.theta_history[-1] if irl.theta_history else irl.theta
        print(f"  [{', '.join([f'{t:.4f}' for t in theta_final])}]")

        print(f"\n归一化后的θ:")
        print(f"  θ₁ (速度): {theta_normalized[0]:.4f} ({theta_normalized[0] * 100:.1f}%)")
        print(f"  θ₂ (舒适): {theta_normalized[1]:.4f} ({theta_normalized[1] * 100:.1f}%)")
        print(f"  θ₃ (平顺): {theta_normalized[2]:.4f} ({theta_normalized[2] * 100:.1f}%)")
        print(f"  θ₄ (安全): {theta_normalized[3]:.4f} ({theta_normalized[3] * 100:.1f}%)")

        # 输出训练统计
        if irl.loss_history:
            print(f"\n训练统计:")
            print(f"  初始损失: {irl.loss_history[0]:.6f}")
            print(f"  最终损失: {irl.loss_history[-1]:.6f}")
            print(f"  损失减少: {((irl.loss_history[0] - irl.loss_history[-1]) / irl.loss_history[0] * 100):.1f}%")

        print("=" * 50)
        print(f"训练日志已保存到: claude_irl_training_log.csv")
        print(f"收敛图表已保存到: training_convergence.png")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
