"""Simulation logic for the fake news spread model."""

import numpy as np
import networkx as nx
import random
import pandas as pd
from config import (THETA_B, THETA_S, A, BASE_GAMMA, LAMBDA_DECAY, 
                   BETA1, BETA2, BETA3, BETA4, CATEGORY_TRAIT)

class FakeNewsSimulator:
    def __init__(self, num_agents, agent_data=None):
        """Initialize the simulator with given number of agents and optional agent data."""
        self.num_agents = num_agents
        self.G = nx.erdos_renyi_graph(num_agents, 0.3)
        self.agent_states = {}
        self._graph_pos = nx.spring_layout(self.G, seed=42)
        
        if agent_data is None:
            agent_data = self._generate_random_agents(num_agents)
        self._initialize_agents(agent_data)

    def _generate_random_agents(self, n):
        """Generate random agent profiles."""
        data = {
            'confirmation_bias': np.random.uniform(0, 1, n),
            'emotional_susceptibility': np.random.uniform(0, 1, n),
            'trust_level': np.random.uniform(0, 1, n),
            'critical_thinking': np.random.uniform(0, 1, n),
            'fact_check_signal': np.random.uniform(0, 1, n),
            'risk_perception': np.random.uniform(0, 1, n)
        }
        return pd.DataFrame(data)

    def _initialize_agents(self, agent_df):
        """Initialize agent states from provided data."""
        for i, row in agent_df.iterrows():
            risk_perception = row['risk_perception'] if 'risk_perception' in row else 0.5
            self.agent_states[i] = {
                'belief': 0.0,
                'shared': False,
                'confirmation_bias': row['confirmation_bias'],
                'emotional_susceptibility': row['emotional_susceptibility'],
                'trust_level': row['trust_level'],
                'critical_thinking': row['critical_thinking'],
                'fact_check_signal': row['fact_check_signal'],
                'risk_perception': risk_perception,
                'scammed': False
            }

    def seed_initial_state(self, is_scam=False):
        """Set initial believers/scam victims."""
        for node in random.sample(list(self.agent_states.keys()), 2):
            if is_scam:
                self.agent_states[node]['scammed'] = True
            else:
                self.agent_states[node]['belief'] = 1.0
                self.agent_states[node]['shared'] = True

    def simulate_scam_round(self):
        """Simulate one round of scam spread."""
        nodes = list(self.G.nodes())
        scammed = {i: self.agent_states[i]['scammed'] for i in nodes}
        new_scammed = {}
        
        for i in nodes:
            if scammed[i]:
                new_scammed[i] = True
                continue
                
            neighbors = list(self.G.neighbors(i))
            exposures = sum(scammed[j] for j in neighbors)
            
            if exposures == 0:
                new_scammed[i] = False
            else:
                a = self.agent_states[i]
                P_scam = 0.5 * a['trust_level'] + 0.4 * (1 - a['risk_perception']) - 0.3 * a['critical_thinking']
                P_scam = max(0, min(1, P_scam))
                new_scammed[i] = P_scam > 0.5
                
        for i in nodes:
            self.agent_states[i]['scammed'] = new_scammed[i]
            
        return [self.agent_states[i]['scammed'] for i in nodes]

    def simulate_fake_news_round(self, juice_factor, topic_weight, topic_category,
                               intervention=False, extra_rounds=False):
        """Simulate one round of fake news spread."""
        nodes = list(self.G.nodes())
        gamma = BASE_GAMMA * (2 if intervention else 1)
        shared = {i: self.agent_states[i]['shared'] for i in nodes}
        beliefs = {i: self.agent_states[i]['belief'] for i in nodes}
        new_beliefs = {}

        # Get thresholds and parameters based on conditions
        theta_b, theta_s, lambda_decay, exposure_alpha = self._get_dynamic_parameters(
            juice_factor, intervention, extra_rounds
        )

        # Calculate weights based on topic category
        weights = self._calculate_trait_weights(topic_category)

        # Update beliefs for each agent
        for i in nodes:
            neighbors = list(self.G.neighbors(i))
            exposures = sum(shared[j] for j in neighbors)
            a = self.agent_states[i]
            
            if extra_rounds or exposures == 0:
                new_beliefs[i] = beliefs[i] * np.exp(-lambda_decay)
            else:
                P_star = self._calculate_belief_probability(
                    a, weights, topic_weight, juice_factor
                )
                exposure_factor = (1 / (1 + np.exp(-exposure_alpha * exposures))) * (1 + 1.2 * juice_factor)
                P_believe = exposure_factor * P_star - gamma * a['fact_check_signal']
                new_beliefs[i] = 1.0 if P_believe > theta_b else beliefs[i] * np.exp(-lambda_decay)

        # Update agent states
        for i in nodes:
            self.agent_states[i]['belief'] = new_beliefs[i]
            self.agent_states[i]['shared'] = new_beliefs[i] > theta_s

        return [self.agent_states[i]['shared'] for i in nodes]

    def _get_dynamic_parameters(self, juice_factor, intervention, extra_rounds):
        """Get dynamic parameters based on current conditions."""
        if juice_factor >= 0.95:
            theta_b, theta_s = 0.35, 0.55
            lambda_decay, exposure_alpha = 0.03, 1.2
        elif juice_factor >= 0.8:
            theta_b, theta_s = 0.42, 0.62
            lambda_decay, exposure_alpha = 0.06, 1.0
        else:
            theta_b, theta_s = THETA_B, THETA_S
            lambda_decay, exposure_alpha = LAMBDA_DECAY, A

        if intervention:
            theta_s = 0.92
            lambda_decay = 0.25

        if extra_rounds:
            lambda_decay = 0.5

        return theta_b, theta_s, lambda_decay, exposure_alpha

    def _calculate_trait_weights(self, topic_category):
        """Calculate trait weights based on topic category."""
        weights = {
            'confirmation_bias': BETA1,
            'emotional_susceptibility': BETA2,
            'trust_level': BETA3,
            'critical_thinking': BETA4
        }
        
        if topic_category and topic_category in CATEGORY_TRAIT:
            main_trait = CATEGORY_TRAIT[topic_category]
            for k in weights:
                weights[k] *= 1.7 if k == main_trait else 0.8
                
        return weights

    def _calculate_belief_probability(self, agent, weights, topic_weight, juice_factor):
        """Calculate the probability of an agent believing the fake news."""
        return (
            weights['confirmation_bias'] * agent['confirmation_bias'] +
            weights['emotional_susceptibility'] * agent['emotional_susceptibility'] * (topic_weight + 0.2 * juice_factor) +
            weights['trust_level'] * agent['trust_level'] +
            0.15 * juice_factor -
            weights['critical_thinking'] * agent['critical_thinking']
        )

    def get_graph_layout(self):
        """Get the precomputed graph layout."""
        return self._graph_pos

    def get_node_colors(self, is_scam=False):
        """Get colors for visualization."""
        if is_scam:
            return ['red' if self.agent_states[n]['scammed'] else 'blue' 
                   for n in self.G.nodes()]
        return ['red' if self.agent_states[n]['shared'] else 'blue' 
                for n in self.G.nodes()]
