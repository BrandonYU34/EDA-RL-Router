import numpy as np
import torch
from Net import ACNetwork
from routing_gym import RoutingEnv
from IL_expert_alternate import Expert, generate_coordinates, switcher
from PPO_structure import PPO


if __name__ == "__main__":
    expert = Expert()
    num_agents = 5

    start_pos, end_pos = generate_coordinates()
    env = RoutingEnv(start_pos, end_pos)
    max_step = env.wsize ** 2

    device = torch.device('cuda')
    PPO_model = ACNetwork().to(device)
    #PPO_model.load_state_dict(torch.load('IL_agent.pt'))

    # ----------------------------------------- #
    # 參數設置
    # ----------------------------------------- #

    num_episodes = 100000  # 迭代次數
    actor_lr = 2e-5  # 策略網路學習率
    critic_lr = 2e-5  # 價值網路學習率
    lmbda = 0.95  # 優勢函數系數
    epochs = 10  # 一個批次訓練的次數
    eps = 0.2  # PPO中限制更新範圍參數
    gamma = 0.95  # 折扣因子

    # ----------------------------------------- #
    # model
    # ----------------------------------------- #

    centralized_agent = PPO(0, PPO_model, actor_lr, critic_lr, lmbda, epochs, eps, gamma,
                            device)
    PPO_agent = [PPO(i + 1, PPO_model, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device) for i in
                 range(num_agents)]
    # ----------------------------------------- #
    # 訓練--回合更新 on_policy
    # ----------------------------------------- #

    start_pos, end_pos = generate_coordinates()
    start_pos, end_pos, shorts = expert.instruct(start_pos, end_pos)
    while shorts:
        start_pos, end_pos = generate_coordinates()
        start_pos, end_pos, shorts = expert.instruct(start_pos, end_pos)

    print(start_pos, end_pos)

    for i in range(num_episodes):
        start_pos, end_pos = generate_coordinates()
        start_pos, end_pos, shorts = expert.instruct(start_pos, end_pos)
        while shorts:
            start_pos, end_pos = generate_coordinates()
            start_pos, end_pos, shorts = expert.instruct(start_pos, end_pos)
        state = env.reset(start_pos, end_pos)  # 環境重製
        done = False
        episode_return, executing_id, num_step = 0, 0, 0  # 總reward, 正在執行動作的agent ID
        # 儲存每個episode的數據
        transition_dict = {
            'observations': [],
            'distances': [],
            'actions': [],
            'positions': [],
        }
        while not done and num_step < max_step:
            executing_id = switcher(executing_id, num_agents, state)
            #action_p = PPO_agent[executing_id-1].deterministic_action(state)
            action = expert.policy[num_step]
            next_state, reward, done, _ = env.step(executing_id, action)  # 环境更新
            transition_dict['observations'].append(state.getObservation(executing_id))
            transition_dict['distances'].append(state.getDistance(executing_id))
            transition_dict['positions'].append(state.getPos(executing_id))
            transition_dict['actions'].append(action)
            # 更新狀態
            state = next_state
            num_step += 1
            # 累績獎勵

        # 模型訓練
        for key in transition_dict:
            transition_dict[key] = np.array(transition_dict[key])

        print('Episode', i, ':', end='')
        centralized_agent.imitate(transition_dict)
        if i % 100 == 0:
            torch.save(PPO_model.state_dict(), 'IL_agent.pt')
