"""Visualization utilities for comparing ABM and PBM results."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ComparisonVisualizer:
    def __init__(self, abm_results: Dict, pbm_results: Dict, context: str):
        """Initialize the comparison visualizer.
        
        Args:
            abm_results: Results from agent-based simulation
            pbm_results: Results from population-based simulation
            context: The news context used in simulation
        """
        self.abm_results = abm_results
        self.pbm_results = pbm_results
        self.context = context
        
    def show_comparison(self, window: tk.Toplevel) -> None:
        """Display comparison visualizations in the provided window.
        
        Args:
            window: Tkinter window to display the comparison
        """
        # Create main container frame
        main_frame = tk.Frame(window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 8))
        
        # Time series comparison
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
        self._plot_time_series(ax1)
        
        # Final state comparison
        ax2 = plt.subplot2grid((2, 3), (0, 2))
        self._plot_final_state(ax2)
        
        # Metrics comparison
        ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
        self._plot_metrics_comparison(ax3)
        
        plt.tight_layout()
        
        # Create canvas and pack it
        canvas = FigureCanvasTkAgg(fig, master=main_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Add text summary below
        summary_frame = tk.Frame(main_frame)
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        self._add_text_summary(summary_frame)
        
        # Configure window size and position
        window.geometry("1200x800")
        canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        
        # No need for additional summary frame as it's already added above
        
    def _plot_time_series(self, ax):
        """Plot time series comparison."""
        # Calculate ABM believer counts
        abm_history = self.abm_results['believer_counts']
        abm_counts = [sum(1 for x in round_data if x) for round_data in abm_history]
        rounds = range(len(abm_counts))
        
        # Get PBM believer counts
        pbm_believers = self.pbm_results['believers']
        
        # Plot results
        ax.plot(rounds, abm_counts, 'b-', label='ABM Believers', linewidth=2)
        ax.plot(rounds, pbm_believers[:len(rounds)], 'r--', label='PBM Believers', linewidth=2)
        
        # Add points for better visibility
        ax.scatter(rounds, abm_counts, color='blue', s=30, alpha=0.6)
        ax.scatter(rounds, pbm_believers[:len(rounds)], color='red', s=30, alpha=0.6)
        
        ax.set_title('Belief Spread Over Time: ABM vs PBM', pad=10)
        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel('Number of Believers', fontsize=10)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
    def _plot_final_state(self, ax):
        """Plot final state comparison pie charts."""
        abm_final_believers = sum(1 for x in self.abm_results['believer_counts'][-1] if x)
        abm_final = [
            abm_final_believers,
            self.abm_results['total_agents'] - abm_final_believers
        ]
        
        pbm_final = [
            int(self.pbm_results['believers'][-1]),
            int(self.pbm_results['susceptible'][-1])
        ]
        
        # Create mini pie charts
        # Colors: red for believers, blue for non-believers
        colors = ['red', 'blue']  # [believers, non-believers]
        
        # Create outer ring (ABM)
        wedges1, texts1, autotexts1 = ax.pie(
            abm_final, radius=1, center=(0, 0),
            colors=colors,
            labels=['Believers', 'Non-believers'],
            autopct='%1.1f%%',
            wedgeprops=dict(width=0.3)
        )
        
        # Create inner ring (PBM)
        wedges2, texts2, autotexts2 = ax.pie(
            pbm_final, radius=0.7, center=(0, 0),
            colors=['lightcoral', 'lightblue'],  # lighter shades for inner ring
            autopct='%1.1f%%',
            wedgeprops=dict(width=0.3)
        )

        # Hide inner ring labels and percentages
        for text in texts2 + autotexts2:
            text.set_text('')
            
        ax.set_title('Final State Comparison\nOuter: ABM, Inner: PBM')

        # Remove inner ring labels and adjust text properties
        ax.set_title('Final State Comparison\nOuter: ABM, Inner: PBM')
        
    def _plot_metrics_comparison(self, ax):
        """Plot comparison of key metrics."""
        metrics = ['Peak Believers', 'Spread Rate', 'Time to Peak']
        
        # Calculate ABM metrics
        abm_counts = [sum(1 for x in round_data if x) 
                     for round_data in self.abm_results['believer_counts']]
        abm_max_believers = max(abm_counts)
        abm_peak_time = np.argmax(abm_counts)
        total_agents = self.abm_results['total_agents']
        
        abm_values = [
            abm_max_believers,
            abm_max_believers / total_agents,
            abm_peak_time
        ]
        
        # Calculate PBM metrics
        pbm_believers = np.array(self.pbm_results['believers'])
        pbm_max = max(pbm_believers)
        
        pbm_values = [
            pbm_max,
            pbm_max / total_agents,  # Use same total agents as ABM
            np.argmax(pbm_believers)
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, abm_values, width, label='ABM', color='blue', alpha=0.6)
        ax.bar(x + width/2, pbm_values, width, label='PBM', color='red', alpha=0.6)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_title('Comparison of Key Metrics')
        
    def _add_text_summary(self, frame):
        """Add text summary of comparison."""
        # Calculate statistics
        abm_peak = max(sum(1 for x in round_data if x) 
                      for round_data in self.abm_results['believer_counts'])
        pbm_peak = max(self.pbm_results['believers'])
        abm_final = sum(1 for x in self.abm_results['believer_counts'][-1] if x)
        pbm_final = int(self.pbm_results['believers'][-1])
        total_agents = self.abm_results['total_agents']
        
        # Calculate time to peak and convergence
        abm_peak_time = next(i for i, round_data in enumerate(self.abm_results['believer_counts']) 
                            if sum(1 for x in round_data if x) == abm_peak)
        pbm_peak_time = next(i for i, believers in enumerate(self.pbm_results['believers']) 
                            if believers == pbm_peak)
        
        # Create header frame with title
        header_frame = tk.Frame(frame, relief=tk.RIDGE, borderwidth=2, bg='#f0f0f0')
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        # Main title
        tk.Label(header_frame, text="Simulation Analysis", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack(pady=5)
        
        # Context section
        context_frame = tk.Frame(frame, relief=tk.RIDGE, borderwidth=2)
        context_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(context_frame, text="Context Analysis", font=('Arial', 11, 'bold')).pack(pady=5)
        tk.Label(context_frame, text=self.context, font=('Arial', 10), wraplength=400).pack(pady=5)
        
        # Statistics frame
        stats_frame = tk.Frame(frame, relief=tk.RIDGE, borderwidth=2)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(stats_frame, text="Model Performance Metrics", font=('Arial', 11, 'bold')).pack(pady=5)
        
        # Create two columns
        cols_frame = tk.Frame(stats_frame)
        cols_frame.pack(fill=tk.X, padx=10, pady=5)
        left_col = tk.Frame(cols_frame)
        right_col = tk.Frame(cols_frame)
        left_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        right_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # ABM Statistics (Left Column)
        tk.Label(left_col, text="Agent-Based Model (ABM)", font=('Arial', 10, 'bold')).pack(anchor='w', pady=2)
        tk.Label(left_col, text=f"• Peak Believers: {abm_peak} ({(abm_peak/total_agents*100):.1f}%)", 
                font=('Arial', 10)).pack(anchor='w')
        tk.Label(left_col, text=f"• Time to Peak: Round {abm_peak_time}", 
                font=('Arial', 10)).pack(anchor='w')
        tk.Label(left_col, text=f"• Final State: {abm_final} believers ({(abm_final/total_agents*100):.1f}%)", 
                font=('Arial', 10)).pack(anchor='w')
        
        # PBM Statistics (Right Column)
        tk.Label(right_col, text="Population-Based Model (PBM)", font=('Arial', 10, 'bold')).pack(anchor='w', pady=2)
        tk.Label(right_col, text=f"• Peak Believers: {pbm_peak} ({(pbm_peak/total_agents*100):.1f}%)", 
                font=('Arial', 10)).pack(anchor='w')
        tk.Label(right_col, text=f"• Time to Peak: Round {pbm_peak_time}", 
                font=('Arial', 10)).pack(anchor='w')
        tk.Label(right_col, text=f"• Final State: {pbm_final} believers ({(pbm_final/total_agents*100):.1f}%)", 
                font=('Arial', 10)).pack(anchor='w')
        
        # Analysis frame
        analysis_frame = tk.Frame(frame, relief=tk.RIDGE, borderwidth=2)
        analysis_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(analysis_frame, text="Comparative Analysis", font=('Arial', 11, 'bold')).pack(pady=5)
        
        # Model comparison analysis
        comparison_text = f"""• Diffusion Speed: {'ABM' if abm_peak_time < pbm_peak_time else 'PBM'} shows faster initial spread
• Peak Difference: {abs(abm_peak - pbm_peak)} agents ({abs(abm_peak/total_agents*100 - pbm_peak/total_agents*100):.1f}%)
• Final State Difference: {abs(abm_final - pbm_final)} agents ({abs(abm_final/total_agents*100 - pbm_final/total_agents*100):.1f}%)
• Model Characteristics:
  - ABM: Captures individual variations and network effects
  - PBM: Shows smoother, population-level dynamics"""
        
        tk.Label(analysis_frame, text=comparison_text, font=('Arial', 10), justify=tk.LEFT).pack(padx=10, pady=5, anchor='w')
