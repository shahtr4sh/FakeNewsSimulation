"""Population-based modeling for fake news spread simulation."""

import numpy as np
from typing import Dict, List, Tuple

class PopulationSimulator:
    def __init__(self, initial_population: int):
        """Initialize the population-based simulator.
        
        Args:
            initial_population: Total population size
        """
        self.total_population = initial_population
        
        # Population compartments
        self.susceptible = initial_population - 2  # Start with 2 initial spreaders
        self.believers = 2
        self.immune = 0
        
        # History tracking
        self.susceptible_history = [self.susceptible]
        self.believers_history = [self.believers]
        self.immune_history = [self.immune]
        
        # Default parameters
        self.contact_rate = 0.3
        self.belief_rate = 0.4
        self.recovery_rate = 0.1
        
    def adjust_rates(self, topic_weight: float, juice_factor: float, 
                    intervention: bool = False) -> None:
        """Adjust transmission rates based on news properties."""
        # Base contact rate modified by topic weight
        self.contact_rate = 0.3 * topic_weight
        
        # Belief rate affected by juiciness
        self.belief_rate = 0.4 * (1 + juice_factor)
        
        # Recovery (immunization) rate increased by intervention
        self.recovery_rate = 0.1 * (2 if intervention else 1)
        
        # Ensure rates are within valid range
        self.belief_rate = min(0.95, self.belief_rate)
        self.contact_rate = min(0.95, self.contact_rate)
        
    def simulate_step(self) -> Tuple[int, int, int]:
        """Run one step of the simulation using differential equations.
        
        Returns:
            Tuple of (susceptible, believers, immune) counts
        """
        # Calculate transitions using difference equations
        # dS/dt = -β*S*I/N
        # dI/dt = β*S*I/N - γ*I
        # dR/dt = γ*I
        
        N = self.total_population
        S = self.susceptible
        I = self.believers
        R = self.immune
        
        # Calculate new infections
        new_believers = (self.contact_rate * self.belief_rate * S * I / N)
        new_believers = min(new_believers, S)  # Cannot exceed susceptible population
        
        # Calculate recoveries
        recoveries = self.recovery_rate * I
        
        # Update compartments
        self.susceptible = max(0, S - new_believers)
        self.believers = max(0, I + new_believers - recoveries)
        self.immune = R + recoveries
        
        # Store history
        self.susceptible_history.append(int(self.susceptible))
        self.believers_history.append(int(self.believers))
        self.immune_history.append(int(self.immune))
        
        return (int(self.susceptible), int(self.believers), int(self.immune))
    
    def get_history(self) -> Dict[str, List[int]]:
        """Get the simulation history.
        
        Returns:
            Dictionary with lists of population counts over time
        """
        return {
            'susceptible': [int(x) for x in self.susceptible_history],
            'believers': [int(x) for x in self.believers_history],
            'immune': [int(x) for x in self.immune_history]
        }
        
    def get_final_stats(self) -> Dict[str, float]:
        """Calculate final statistics from the simulation.
        
        Returns:
            Dictionary with final statistics
        """
        total_affected = max(self.believers_history)  # Peak believers
        final_immune = self.immune_history[-1]
        spread_rate = total_affected / self.total_population
        
        return {
            'peak_believers': total_affected,
            'final_immune': final_immune,
            'spread_rate': spread_rate,
            'susceptible_remaining': self.susceptible
        }
