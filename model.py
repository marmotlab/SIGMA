import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.cuda.amp import autocast
import configs
import wandb

class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.block1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.block2 = nn.Conv2d(channel, channel, 3, 1, 1)

    def forward(self, x):
        identity = x

        x = self.block1(x)
        x = F.relu(x)

        x = self.block2(x)

        x += identity

        x = F.relu(x)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W_Q = nn.Linear(input_dim, output_dim * num_heads)
        self.W_K = nn.Linear(input_dim, output_dim * num_heads)
        self.W_V = nn.Linear(input_dim, output_dim * num_heads)
        self.W_O = nn.Linear(output_dim * num_heads, output_dim, bias=False)

    def forward(self, input, attn_mask):
        # input: [batch_size x num_agents x input_dim]
        batch_size, num_agents, input_dim = input.size()
        assert input_dim == self.input_dim
        
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,2)  # q_s: [batch_size x num_heads x num_agents x output_dim]
        k_s = self.W_K(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,2)  # k_s: [batch_size x num_heads x num_agents x output_dim]
        v_s = self.W_V(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,2)  # v_s: [batch_size x num_heads x num_agents x output_dim]

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        assert attn_mask.size(0) == batch_size, 'mask dim {} while batch size {}'.format(attn_mask.size(0), batch_size)

        attn_mask = attn_mask.unsqueeze(1).repeat_interleave(self.num_heads, 1) # attn_mask : [batch_size x num_heads x num_agents x num_agents]
        assert attn_mask.size() == (batch_size, self.num_heads, num_agents, num_agents)

        # context: [batch_size x num_heads x num_agents x output_dim]
        with autocast(enabled=False):
            scores = torch.matmul(q_s.float(), k_s.float().transpose(-1, -2)) / (self.output_dim**0.5) # scores : [batch_size x n_heads x num_agents x num_agents]
            scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
            attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_agents, self.num_heads*self.output_dim) # context: [batch_size x len_q x n_heads * d_v]
        output = self.W_O(context)

        return output # output: [batch_size x num_agents x output_dim]

class CommBlock(nn.Module):
    def __init__(self, input_dim, output_dim=64, num_heads=configs.num_comm_heads, num_layers=configs.num_comm_layers):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.self_attn = MultiHeadAttention(input_dim, output_dim, num_heads)

        self.update_cell = nn.GRUCell(output_dim, input_dim)


    def forward(self, latent, comm_mask):
        '''
        latent shape: batch_size x num_agents x latent_dim
        
        '''
        num_agents = latent.size(1)

        # agent indices of agent that use communication
        update_mask = comm_mask.sum(dim=-1) > 1
        comm_idx = update_mask.nonzero(as_tuple=True)

        # no agent use communication, return
        if len(comm_idx[0]) == 0:
            return latent

        if len(comm_idx)>1:
            update_mask = update_mask.unsqueeze(2)

        attn_mask = comm_mask==False

        for _ in range(self.num_layers):

            info = self.self_attn(latent, attn_mask=attn_mask)
            # latent = attn_layer(latent, attn_mask=attn_mask)
            if len(comm_idx)==1:

                batch_idx = torch.zeros(len(comm_idx[0]), dtype=torch.long)
                latent[batch_idx, comm_idx[0]] = self.update_cell(info[batch_idx, comm_idx[0]], latent[batch_idx, comm_idx[0]])
            else:
                update_info = self.update_cell(info.view(-1, self.output_dim), latent.view(-1, self.input_dim)).view(configs.batch_size, num_agents, self.input_dim)
                # update_mask = update_mask.unsqueeze(2)
                latent = torch.where(update_mask, update_info, latent)
                # latent[comm_idx] = self.update_cell(info[comm_idx], latent[comm_idx])
                # latent = self.update_cell(info, latent)

        return latent

class Network(nn.Module):
    def __init__(self, input_shape=configs.obs_shape, cnn_channel=configs.cnn_channel, hidden_dim=configs.hidden_dim,
                max_comm_agents=configs.max_comm_agents):

        super().__init__()

        self.input_shape = input_shape
        self.latent_dim = 16*7*7
        self.hidden_dim = hidden_dim
        self.max_comm_agents = max_comm_agents

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(self.input_shape[0], cnn_channel, 3, 1),
            nn.ReLU(True),

            ResBlock(cnn_channel),

            ResBlock(cnn_channel),

            ResBlock(cnn_channel),

            nn.Conv2d(cnn_channel, 16, 1, 1),
            nn.ReLU(True),

            nn.Flatten(),
        )

        self.recurrent = nn.GRUCell(self.latent_dim, self.hidden_dim)

        self.comm = CommBlock(hidden_dim)

        self.mlp_dim = 1
        self.lambdas = configs.lambdas
        # MLP for calculating output
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.mlp_dim),
            nn.Sigmoid()
        )

        # dueling q structure
        if configs.Advantage_all:
            self.adv = nn.Linear(hidden_dim + self.mlp_dim, 5)
        else:
            self.adv = nn.Linear(hidden_dim, 5)
        self.state = nn.Linear(hidden_dim, 1)

        self.hidden = None

        for _, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def step(self, obs, pos):
        num_agents = obs.size(0)

        latent = self.obs_encoder(obs)

        # MLP 输出
        mlp_output = self.mlp(latent)

        if self.hidden is None:
            self.hidden = self.recurrent(latent)
        else:
            self.hidden = self.recurrent(latent, self.hidden)

        # from num_agents x hidden_dim to 1 x num_agents x hidden_dim
        self.hidden = self.hidden.unsqueeze(0)

        # masks for communication block
        agents_pos = pos
        pos_mat = (agents_pos.unsqueeze(1)-agents_pos.unsqueeze(0)).abs()
        dist_mat = (pos_mat[:,:,0]**2+pos_mat[:,:,1]**2).sqrt()
        # mask out agents that out of range of FOV
        in_obs_mask = (pos_mat<=configs.obs_radius).all(2)
        # mask out agents that are far away
        _, ranking = dist_mat.topk(min(self.max_comm_agents, num_agents), dim=1, largest=False)
        dist_mask = torch.zeros((num_agents, num_agents), dtype=torch.bool)
        dist_mask.scatter_(1, ranking, True)

        comm_mask = torch.bitwise_and(in_obs_mask, dist_mask)

        self.hidden = self.comm(self.hidden, comm_mask)
        self.hidden = self.hidden.squeeze(0)

        # 将 MLP 输出与隐藏状态拼接，传递给动作生成网络
        combined_input = torch.cat((self.hidden, mlp_output), dim=1)

        # 计算损失项
        sheaf_section_loss = torch.zeros(num_agents, dtype=torch.float32)

        for i in range(num_agents):
            # 获取第 i 个智能体的邻域智能体索引
            neighbors = comm_mask[i].nonzero(as_tuple=True)[0]

            # 如果没有邻域智能体，则跳过
            if len(neighbors) == 0:
                continue

            # 第 i 个智能体的 mlp_output
            output_i = mlp_output[i]

            # 初始化当前智能体的 L2 损失值
            loss = 0.0

            # 计算第 i 个智能体与其邻域智能体的 L2 损失值
            for j in neighbors:
                output_j = mlp_output[j]
                loss += torch.nn.functional.mse_loss(output_i, output_j, reduction='sum')

            # 计算平均损失
            sheaf_section_loss[i] = loss / len(neighbors)

        if configs.Advantage_all:
            adv_val = self.adv(combined_input)
        else:
            adv_val = self.adv(self.hidden)
        state_val = self.state(self.hidden)
        # adv_val = self.adv(self.hidden)
        # state_val = self.state(self.hidden)
        sheaf_section_loss = sheaf_section_loss.unsqueeze(1)
        if configs.Sec_cons:
            q_val = state_val - self.lambdas * sheaf_section_loss + adv_val - adv_val.mean(1, keepdim=True)
        else:
            q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        actions = torch.argmax(q_val, 1).tolist()

        return actions, q_val.numpy(), self.hidden.numpy(), comm_mask.numpy()

    @torch.no_grad()
    def step_test(self, obs, pos):
        num_agents = obs.size(0)

        latent = self.obs_encoder(obs)

        # MLP 输出
        mlp_output = self.mlp(latent)

        if self.hidden is None:
            self.hidden = self.recurrent(latent)
        else:
            self.hidden = self.recurrent(latent, self.hidden)

        # from num_agents x hidden_dim to 1 x num_agents x hidden_dim
        self.hidden = self.hidden.unsqueeze(0)

        # masks for communication block
        agents_pos = pos
        pos_mat = (agents_pos.unsqueeze(1)-agents_pos.unsqueeze(0)).abs()
        dist_mat = (pos_mat[:,:,0]**2+pos_mat[:,:,1]**2).sqrt()
        # mask out agents that out of range of FOV
        in_obs_mask = (pos_mat<=configs.obs_radius).all(2)
        # mask out agents that are far away
        _, ranking = dist_mat.topk(min(self.max_comm_agents, num_agents), dim=1, largest=False)
        dist_mask = torch.zeros((num_agents, num_agents), dtype=torch.bool)
        dist_mask.scatter_(1, ranking, True)

        comm_mask = torch.bitwise_and(in_obs_mask, dist_mask)

        self.hidden = self.comm(self.hidden, comm_mask)
        self.hidden = self.hidden.squeeze(0)

        # 将 MLP 输出与隐藏状态拼接，传递给动作生成网络
        combined_input = torch.cat((self.hidden, mlp_output), dim=1)

        # 计算损失项
        sheaf_section_loss = torch.zeros(num_agents, dtype=torch.float32)

        for i in range(num_agents):
            # 获取第 i 个智能体的邻域智能体索引
            neighbors = comm_mask[i].nonzero(as_tuple=True)[0]

            # 如果没有邻域智能体，则跳过
            if len(neighbors) == 0:
                continue

            # 第 i 个智能体的 mlp_output
            output_i = mlp_output[i]

            # 初始化当前智能体的 L2 损失值
            loss = 0.0

            # 计算第 i 个智能体与其邻域智能体的 L2 损失值
            for j in neighbors:
                output_j = mlp_output[j]
                loss += torch.nn.functional.mse_loss(output_i, output_j, reduction='sum')

            # 计算平均损失
            sheaf_section_loss[i] = loss / len(neighbors)

        if configs.Advantage_all:
            adv_val = self.adv(combined_input)
        else:
            adv_val = self.adv(self.hidden)
        state_val = self.state(self.hidden)
        # adv_val = self.adv(self.hidden)
        # state_val = self.state(self.hidden)
        sheaf_section_loss = sheaf_section_loss.unsqueeze(1)
        if configs.Sec_cons:
            q_val = state_val - self.lambdas * sheaf_section_loss + adv_val - adv_val.mean(1, keepdim=True)
        else:
            q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        actions = torch.argmax(q_val, 1).tolist()

        return actions, q_val.numpy(), self.hidden.numpy(), comm_mask.numpy(), sheaf_section_loss.mean().numpy()

    def reset(self):
        self.hidden = None

    @autocast()
    def forward(self, obs, steps, hidden, comm_mask):
        # comm_mask shape: batch_size x seq_len x max_num_agents x max_num_agents
        max_steps = obs.size(1)
        num_agents = comm_mask.size(2)

        assert comm_mask.size(2) == configs.max_num_agents

        obs = obs.transpose(1, 2)

        obs = obs.contiguous().view(-1, *self.input_shape)

        latent = self.obs_encoder(obs)

        # MLP 输出
        mlp_output = self.mlp(latent) # [17280, 2]

        latent = latent.view(configs.batch_size*num_agents, max_steps, self.latent_dim).transpose(0, 1) # [18, 960, 784] latent i shape: torch.Size([960, 784])
        mlp_output = mlp_output.view(configs.batch_size * num_agents, max_steps, self.mlp_dim).transpose(0, 1) # [18, 960, 2]

        hidden_buffer = []
        mlp_output_buffer = []
        for i in range(max_steps):
            # hidden size: batch_size*num_agents x self.hidden_dim
            # [960, 256]
            hidden = self.recurrent(latent[i], hidden)
            hidden = hidden.view(configs.batch_size, num_agents, self.hidden_dim) # [192, 5, 256]
            mlp_tmp = mlp_output[i].view(configs.batch_size, num_agents, self.mlp_dim) # [192, 5, 2]
            hidden = self.comm(hidden, comm_mask[:, i])
            # only hidden from agent 0
            hidden_buffer.append(hidden[:, 0]) # hidden[:, 0] [192, 256]
            mlp_output_buffer.append(mlp_tmp[:, 0])
            hidden = hidden.view(configs.batch_size*num_agents, self.hidden_dim)

        # hidden buffer size: batch_size x seq_len x self.hidden_dim
        hidden_buffer = torch.stack(hidden_buffer).transpose(0, 1) # hidden_buffer shape: torch.Size([192, 18, 256])

        # hidden size: batch_size x self.hidden_dim
        hidden = hidden_buffer[torch.arange(configs.batch_size), steps-1] # hidden shape: torch.Size([192, 256])

        # 重新调整 mlp_output 的形状以匹配 hidden
        mlp_output_buffer= torch.stack(mlp_output_buffer).transpose(0, 1)
        mlp_output_q = mlp_output_buffer[torch.arange(configs.batch_size), steps-1] # mlp_output shape: torch.Size([192, 2])

        # 将 MLP 输出与隐藏状态拼接，传递给动作生成网络
        combined_input = torch.cat((hidden, mlp_output_q), dim=1)

        # 计算损失项
        sheaf_section_loss = torch.zeros(configs.batch_size, num_agents, dtype=torch.float32)

        mlp_q = mlp_output.view(max_steps, configs.batch_size, num_agents, self.mlp_dim)[steps - 1, torch.arange(configs.batch_size)] # torch.Size([192, 5, 2])
        # comm_mask shape:torch.Size([192, 18, 5, 5])
        comm_q = comm_mask[torch.arange(configs.batch_size), steps - 1] # [192, 5, 5]
        for b in range(configs.batch_size):
            for i in range(num_agents):
                # 获取第 i 个智能体的邻域智能体索引
                neighbors = comm_q[b, i].nonzero(as_tuple=True)[0] # comm_tmpbishape:torch.Size([5])s

                # 如果没有邻域智能体，则跳过
                if len(neighbors) == 0:
                    continue

                # 第 i 个智能体的 mlp_output
                output_i = mlp_q[b, i]

                # 初始化当前智能体的 L2 损失值
                loss = 0.0

                # 计算第 i 个智能体与其邻域智能体的 L2 损失值
                for j in neighbors:
                    output_j = mlp_q[b, j]
                    loss += torch.nn.functional.mse_loss(output_i, output_j, reduction='sum')

                # 计算平均损失
                sheaf_section_loss[b, i] = loss / len(neighbors)

        if configs.Advantage_all:
            adv_val = self.adv(combined_input)
        else:
            adv_val = self.adv(hidden)
        state_val = self.state(hidden)

        sheaf_section_loss = sheaf_section_loss[:, 0]
        sheaf_section_loss = sheaf_section_loss.unsqueeze(1).to(torch.device("cuda"))
        # wandb.log({"sheaf_section_loss": sheaf_section_loss})
        if configs.Sec_cons:
            q_val = state_val - self.lambdas * sheaf_section_loss + adv_val - adv_val.mean(1, keepdim=True)
        else:
            q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        return q_val


