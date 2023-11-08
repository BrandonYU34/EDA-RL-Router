import torch
import torch.nn as nn
from torch.nn import functional as F


class PPO:
    def __init__(self, ag_id, model, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        # AC網路
        self.ACNet = model
        # 策略網路優化器
        self.actor_optimizer = torch.optim.Adam(self.ACNet.policy_network.parameters(), lr=actor_lr)
        # 價值網路優化器
        self.critic_optimizer = torch.optim.Adam(self.ACNet.value_network.parameters(), lr=critic_lr)

        # 學習率調整器，衰減率為次方根倒數
        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_optimizer,
                                                                 lr_lambda=lambda epoch: 1 / ((epoch + 1) ** 0.5))
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic_optimizer,
                                                                  lr_lambda=lambda epoch: 1 / ((epoch + 1) ** 0.5))
        self.ag_id = ag_id
        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE優勢函數縮放系數
        self.epochs = epochs  # 每個batch的訓練次數
        self.eps = eps  # PPO剪裁範圍
        self.device = device

    # 訓練用隨機性動作選擇
    def stochastic_action(self, state):
        obs_tensor = torch.tensor(state.getObservation(self.ag_id), dtype=torch.float).unsqueeze(0).to(self.device)
        dis_tensor = torch.tensor(state.getDistance(self.ag_id), dtype=torch.float).unsqueeze(0).to(self.device)
        pos_tensor = torch.tensor(state.getPos(self.ag_id), dtype=torch.float).unsqueeze(0).to(self.device)
        # softmax輸出各個動作概率[1,n_states]
        probs = self.ACNet.forward_policy(obs_tensor, dis_tensor, pos_tensor)
        # probs的概率分布
        action_list = torch.distributions.Categorical(probs)
        # 訓練用，使用隨機性動作探索
        action = action_list.sample().item()
        return action

    # 測試用確定性動作選擇
    def deterministic_action(self, state):
        obs_tensor = torch.tensor(state.getObservation(self.ag_id), dtype=torch.float).unsqueeze(0).to(self.device)
        dis_tensor = torch.tensor(state.getDistance(self.ag_id), dtype=torch.float).unsqueeze(0).to(self.device)
        pos_tensor = torch.tensor(state.getPos(self.ag_id), dtype=torch.float).unsqueeze(0).to(self.device)
        # softmax輸出各個動作概率[1,n_states]
        probs = self.ACNet.forward_policy(obs_tensor, dis_tensor, pos_tensor)
        # probs的概率分布
        #print(probs)

        action = torch.argmax(probs).item()
        return action

    # 训练
    def learn(self, transition_dict):
        # 從字典中提取數據集
        observations = torch.tensor(transition_dict['observations'], dtype=torch.float).to(self.device)
        distances = torch.tensor(transition_dict['distances'], dtype=torch.float).to(self.device)
        positions = torch.tensor(transition_dict['positions'], dtype=torch.float).to(self.device)
        actions = torch.LongTensor(transition_dict['actions']).unsqueeze(1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).unsqueeze(1).to(self.device)
        next_observations = torch.tensor(transition_dict['next_observations'], dtype=torch.float).to(self.device)
        next_distances = torch.tensor(transition_dict['next_distances'], dtype=torch.float).to(self.device)
        next_positions = torch.tensor(transition_dict['next_positions'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).unsqueeze(1).to(self.device)

        # 目標值，下一個狀態的state_value
        next_q_target = self.ACNet.forward_value(next_observations, next_distances, next_positions)
        # 目標值，下一個狀態的state_value
        td_target = (self.gamma * next_q_target) * (1 - dones) + rewards
        # 預測值，當前的state_value
        td_value = self.ACNet.forward_value(observations, distances, positions)
        # 目標值和預測值state_value差
        td_delta = td_target - td_value

        # 時序差分值 tensor->numpy
        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0
        advantage_list = []

        # GAE優勢函數計算
        for delta in td_delta[::-1]:  # 逆時序差分值 axis=1上倒著取 [], [], []
            # GAE的公式
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        # 正序
        advantage_list.reverse()
        # numpy --> tensor [b,1]
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)

        # 策略網路輸出每个動作的概率，舊策略的概率對數
        old_log_probs = torch.log(self.ACNet.forward_policy(observations, distances, positions).gather(1, actions)).detach()

        actor_loss, critic_loss = None, None
        # 一批次訓練 epochs 輪
        for _ in range(self.epochs):
            probs = self.ACNet.forward_policy(observations, distances, positions).gather(1, actions)
            log_probs = torch.log(probs)
            # 新舊策略間的比例
            ratio = torch.exp(log_probs - old_log_probs)
            # 近端策略優化裁剪目標函式左側項
            surr1 = ratio * advantage
            # 右側項，ratio限制於 1-self.eps 到 1+self.eps 之間
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # 策略網路損失函數
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            # 價值網路的損失函數，當前與下一個state_value的策略差
            critic_loss = torch.mean(F.mse_loss(self.ACNet.forward_value(observations, distances, positions), td_target.detach()))

            # 更新策略網路
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 更新價值網路
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # 學習率衰減
        print(actor_loss, critic_loss)
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def imitate(self, transition_dict):
        # 從字典中提取數據集
        observations = torch.tensor(transition_dict['observations'], dtype=torch.float).to(self.device)
        distances = torch.tensor(transition_dict['distances'], dtype=torch.float).to(self.device)
        positions = torch.tensor(transition_dict['positions'], dtype=torch.float).to(self.device)
        actions = torch.LongTensor(transition_dict['actions']).to(self.device)
        x = 0
        # 一批次訓練 epochs 輪

        loss = None
        while x < len(actions):
            predicted_action_probs = self.ACNet.forward_policy(observations[x:x+3], distances[x:x+3], positions[x:x+3])
            # CrossEntropyLoss需要的是動作的索引，而不是one-hot編碼
            criterion = nn.CrossEntropyLoss()
            loss = criterion(predicted_action_probs, actions[x:x+3])
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            #self.actor_scheduler.step()
            x += 3
        print(loss)



