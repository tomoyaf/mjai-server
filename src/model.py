import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from transformers import BertLayer, BertConfig
from glob import glob
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup


def get_model(use_pretrained_weight):
    config = BertConfig()
    config.vocab_size = 37 + 5
    config.hidden_size = 120
    return ActorCritic(config)


class Embedding(object):
    def __init__(self, config, n_token_type=13):
        super().__init__()
        self.config = config
        self.tile_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        emb = self.tile_embeddings(x)
        emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class Actor(nn.Module):
    def __init__(self, config, bert):
        super().__init__()
        self.config = config
        self.output_dim = 37
        self.bert = bert
        self.head = nn.Linear(self.config.hidden_size, self.output_dim)

    def forward(self,x):
        bert_outputs = self.bert(x)
        last_hidden_state = bert_outputs[0]
        logits = self.head(last_hidden_state)
        return logits


class Critic(nn.Module):
    def __init__(self, config, bert):
        super().__init__()
        self.config = config
        self.bert = bert

    def forward(self,x,):



class ActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = Embedding(config)
        self.actor = Actor(config, self.embeddings)
        self.critic = Critic(config, self.embeddings)
        self.l2 = nn.MSELoss()
        self.alpha = 0.0
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.actor_scheduler = get_linear_schedule_with_warmup(
            self.actor_optimizer,
            num_warmup_steps=config.actor_n_warmup_steps,
            num_training_steps=config.n_training_steps
        )
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.critic_scheduler = get_linear_schedule_with_warmup(
            self.critic_optimizer,
            num_warmup_steps=config.critic_n_warmup_steps,
            num_training_steps=config.n_training_steps
        )

    def train(self, x):
        action_prob_explores = [self.actor(state) for [state, _, _] in x]
        empirical_entropy = 0.0

        for [state, next_state, reward, action_prob_explore] in zip(*x, action_prob_explores):
            # Actor
            self.actor.train()
            self.critic.eval()
            advantage = reward + self.config.gamma * self.critic(next_state) - self.critic(state)
            action_prob = self.actor(x)
            self.alpha = self.alpha + self.config.beta * (self.config.entropy_target - empirical_entropy)
            entropy = -self.alpha * action_prob * torch.log(action_prob)
            empirical_entropy = torch.mean(entropy)

            actor_loss = (action_prob / action_prob_explore) * torch.log(action_prob) * advantage + entropy

            actor_loss.backward()
            self.actor_optimizer.zero_grad()
            self.actor_optimizer.step()

            # Critic
            self.actor.eval()
            self.critic.train()
            critic_loss = self.l2(reward + self.config.gamma * self.critic(next_state) - self.critic(state))

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def forward(self, x):
        self.actor.eval()
        action_prob = self.actor(x)
