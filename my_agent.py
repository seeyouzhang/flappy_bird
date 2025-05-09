import numpy as np
import pygame
from pytorch_mlp import MLPRegression
import argparse
from console import FlappyBirdEnv
from collections import deque
import random

STUDENT_ID = 'a1234567'
DEGREE = 'UG'  # or 'PG'


class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        # do not modify these
        self.show_screen = show_screen
        if mode is None:
            self.mode = 'train'  # mode is either 'train' or 'eval', we will set the mode of your agent to eval mode
        else:
            self.mode = mode

        # 状态空间维度：鸟的y位置、管道顶部和底部的位置
        self.state_dim = 3
        # 动作空间维度：跳或不跳
        self.action_dim = 2
        
        # 经验回放缓冲区
        self.storage = deque(maxlen=10000)
        
        # Q网络
        self.network = MLPRegression(input_dim=self.state_dim, output_dim=self.action_dim, learning_rate=0.001)
        # 目标网络
        self.network2 = MLPRegression(input_dim=self.state_dim, output_dim=self.action_dim, learning_rate=0.001)
        # 初始化目标网络参数
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

        # DQN超参数
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.batch_size = 32  # 批量大小
        self.discount_factor = 0.99  # 折扣因子
        self.target_update_frequency = 10  # 目标网络更新频率
        self.episode_count = 0  # 用于跟踪训练轮数

        # do not modify this
        if load_model_path:
            self.load_model(load_model_path)

    def choose_action(self, state: dict, action_table: dict) -> int:
        """
        This function should be called when the agent action is requested.
        Args:
            state: input state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            action: the action code as specified by the action_table
        """
        # 将状态转换为网络输入格式
        state_tensor = self._preprocess_state(state)
        
        if self.mode == 'train':
            # ε-贪婪策略
            if random.random() < self.epsilon:
                # 随机探索
                action = random.choice(list(action_table.values()))
            else:
                # 利用当前策略
                q_values = self.network.predict(state_tensor)
                action = list(action_table.values())[np.argmax(q_values)]
            
            # 衰减探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
            # 评估模式下直接选择最优动作
            q_values = self.network.predict(state_tensor)
            action = list(action_table.values())[np.argmax(q_values)]
            
        return action

    def _preprocess_state(self, state: dict) -> np.ndarray:
        """
        将游戏状态转换为网络输入格式
        """
        # 提取关键状态信息：鸟的y位置、最近的管道顶部和底部位置
        bird_y = state['bird']['y']
        pipe_top = state['pipes'][0]['top'] if state['pipes'] else 0
        pipe_bottom = state['pipes'][0]['bottom'] if state['pipes'] else 0
        
        return np.array([bird_y, pipe_top, pipe_bottom], dtype=np.float32)

    def receive_after_action_observation(self, state: dict, action_table: dict) -> None:
        """
        This function should be called to notify the agent of the post-action observation.
        Args:
            state: post-action state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            None
        """
        if self.mode != 'train':
            return

        # 计算奖励
        reward = self._calculate_reward(state)
        
        # 存储经验
        current_state = self._preprocess_state(state)
        self.storage.append((current_state, reward))
        
        # 如果经验回放缓冲区足够大，进行训练
        if len(self.storage) >= self.batch_size:
            self._train_network()
            
        # 定期更新目标网络
        self.episode_count += 1
        if self.episode_count % self.target_update_frequency == 0:
            MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

    def _calculate_reward(self, state: dict) -> float:
        """
        计算奖励值
        """
        # 基础奖励：存活奖励
        reward = 0.1
        
        # 如果游戏结束，给予负奖励
        if state.get('game_over', False):
            reward = -1.0
            return reward
            
        # 如果通过管道，给予正奖励
        if state.get('score_increased', False):
            reward = 1.0
            
        return reward

    def _train_network(self):
        """
        训练Q网络
        """
        # 从经验回放缓冲区中随机采样
        batch = random.sample(self.storage, self.batch_size)
        states = np.array([exp[0] for exp in batch])
        rewards = np.array([exp[1] for exp in batch])
        
        # 计算目标Q值
        target_q_values = self.network2.predict(states)
        current_q_values = self.network.predict(states)
        
        # 更新Q值
        for i in range(self.batch_size):
            if rewards[i] == -1.0:  # 游戏结束
                target_q_values[i] = rewards[i]
            else:
                target_q_values[i] = rewards[i] + self.discount_factor * np.max(current_q_values[i])
        
        # 训练网络
        self.network.train(states, target_q_values)

    def save_model(self, path: str = 'my_model.ckpt'):
        """
        Save the MLP model. Unless you decide to implement the MLP model yourself, do not modify this function.

        Args:
            path: the full path to save the model weights, ending with the file name and extension

        Returns:

        """
        self.network.save_model(path=path)

    def load_model(self, path: str = 'my_model.ckpt'):
        """
        Load the MLP model weights.  Unless you decide to implement the MLP model yourself, do not modify this function.
        Args:
            path: the full path to load the model weights, ending with the file name and extension

        Returns:

        """
        self.network.load_model(path=path)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression, net_as_source: MLPRegression):
        """
        Update one MLP model's model parameter by the parameter of another MLP model.
        Args:
            net_to_update: the MLP to be updated
            net_as_source: the MLP to supply the model parameters

        Returns:
            None
        """
        net_to_update.load_state_dict(net_as_source.state_dict())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1)
    args = parser.parse_args()

    # 训练环境
    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=True, level=args.level, game_length=10)
    agent = MyAgent(show_screen=True)
    
    # 训练参数
    episodes = 10000
    best_score = 0
    
    # 训练循环
    for episode in range(episodes):
        env.play(player=agent)
        
        # 打印训练进度
        if episode % 100 == 0:
            print(f"Episode: {episode}, Score: {env.score}, Mileage: {env.mileage}, Epsilon: {agent.epsilon:.3f}")
        
        # 保存最佳模型
        if env.score > best_score:
            best_score = env.score
            agent.save_model(path='best_model.ckpt')
            print(f"New best model saved! Score: {best_score}")
        
        # 每100个episode保存一次检查点
        if episode % 100 == 0:
            agent.save_model(path=f'checkpoint_{episode}.ckpt')
        
        # 每1000个episode清空经验回放缓冲区
        if episode % 1000 == 0:
            agent.storage.clear()
        
        # 每100个episode更新目标网络
        if episode % 100 == 0:
            MyAgent.update_network_model(net_to_update=agent.network2, net_as_source=agent.network)

    # 评估模式
    print("\n开始评估...")
    env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level)
    agent2 = MyAgent(show_screen=False, load_model_path='best_model.ckpt', mode='eval')

    eval_episodes = 10
    scores = []
    for episode in range(eval_episodes):
        env2.play(player=agent2)
        scores.append(env2.score)
        print(f"评估 Episode {episode + 1}, Score: {env2.score}")

    print(f"\n评估结果:")
    print(f"最高分: {np.max(scores)}")
    print(f"平均分: {np.mean(scores):.2f}")
    print(f"标准差: {np.std(scores):.2f}")
