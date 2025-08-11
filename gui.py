"""GUI implementation for the Fake News Simulator."""

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os

from simulator import FakeNewsSimulator
from analysis import analyze_context_juiciness, infer_topic_from_context
from config import TOPICS, TOPIC_CATEGORIES

class FakeNewsSimulatorGUI:
    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Fake News Circulation Simulator")
        
        # Initialize variables
        self.topic = tk.StringVar(value="Ransomware Alert")
        self.juiciness = tk.IntVar(value=50)
        self.agent_source = tk.StringVar(value="CSV")
        self.context_text = tk.StringVar(value="")
        
        # Simulation state
        self.round = 0
        self.max_rounds = 10
        self.intervention = False
        self.round_history = []
        self.scam_history = []
        self.intervention_rounds = []
        self.simulator = None
        
        # Load agents data
        self._load_agent_data()
        
        # Set up the UI
        self.setup_ui()

    def _load_agent_data(self):
        """Load agent profiles from CSV."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        agents_path = os.path.join(script_dir, "data", "agent_profiles1.csv")
        self.df_agents = pd.read_csv(agents_path)

    def setup_ui(self):
        """Set up the user interface."""
        # Configuration frame
        config_frame = tk.Frame(self.root)
        config_frame.pack(pady=10)

        # Context entry
        tk.Label(config_frame, text="Fake News Context:").grid(row=0, column=0, padx=5)
        self.context_entry = tk.Entry(config_frame, textvariable=self.context_text, width=40)
        self.context_entry.grid(row=0, column=1, padx=5)

        # Juiciness slider
        tk.Label(config_frame, text="Juiciness:").grid(row=2, column=0, padx=5)
        self.juiciness_scale = tk.Scale(
            config_frame, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.juiciness
        )
        self.juiciness_scale.grid(row=2, column=1, padx=5)

        # Agent source selection
        tk.Label(config_frame, text="Agent Source:").grid(row=3, column=0, padx=5)
        agent_source_menu = ttk.Combobox(
            config_frame, textvariable=self.agent_source,
            values=["CSV", "Random"], state="readonly"
        )
        agent_source_menu.grid(row=3, column=1, padx=5)

        # Round label
        self.round_label = tk.Label(config_frame, text=f"Round: {self.round}")
        self.round_label.grid(row=0, column=2, padx=10)

        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=(0, 10))

        # Control buttons
        self.run_btn = tk.Button(button_frame, text="Run Simulation",
                                command=self.init_simulation)
        self.run_btn.grid(row=0, column=0, padx=5)

        self.reset_btn = tk.Button(button_frame, text="Run Another Simulation",
                                  command=self.reset_simulation)
        self.reset_btn.grid(row=0, column=1, padx=5)
        self.reset_btn.grid_remove()

        # Canvas frame for visualization
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack()

        # Initialize matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack()

        self.ax.set_title("Fake News Spread - Round 0")
        self.canvas.draw()

    def init_simulation(self):
        """Initialize a new simulation."""
        # Analyze context and set juiciness
        context = self.context_text.get().lower()
        juiciness_score = analyze_context_juiciness(context)
        self.juiciness.set(juiciness_score)
        self.juiciness_scale.set(juiciness_score)

        # Infer topic
        topic, topic_weight, topic_category = infer_topic_from_context(
            context, TOPICS, TOPIC_CATEGORIES
        )
        self._sim_topic = topic
        self._sim_topic_weight = topic_weight
        self._sim_topic_category = topic_category

        # Reset simulation state
        self.round = 0
        self.round_label.config(text=f"Round: {self.round}")
        self.intervention = False
        self.extra_rounds = False
        self.round_history = []
        self.scam_history = []
        self.intervention_rounds = []

        # Initialize simulator
        num_agents = len(self.df_agents)
        agent_data = None if self.agent_source.get() == "Random" else self.df_agents
        self.simulator = FakeNewsSimulator(num_agents, agent_data)
        
        # Seed initial state
        self.simulator.seed_initial_state(is_scam=(topic == "Financial Scam"))

        # Update UI
        self.reset_btn.grid_remove()
        self.run_btn.grid_remove()
        self.update_graph()
        self.root.after(500, self.automate_rounds)

    def automate_rounds(self):
        """Automatically run simulation rounds."""
        if self.round < self.max_rounds:
            if self.round == 5 and not self.intervention:
                self.intervention = True
                self.intervention_rounds.append(self.round)
            self.run_next_round()
            self.root.after(500, self.automate_rounds)

    def run_next_round(self):
        """Run the next simulation round."""
        if self.round >= self.max_rounds:
            return

        # Run simulation step
        if self._sim_topic == "Financial Scam":
            result = self.simulator.simulate_scam_round()
            self.scam_history.append(result)
        else:
            result = self.simulator.simulate_fake_news_round(
                self.juiciness.get() / 100.0,
                self._sim_topic_weight,
                self._sim_topic_category,
                self.intervention,
                getattr(self, 'extra_rounds', False)
            )
            self.round_history.append(result)

        # Update round counter and UI
        self.round += 1
        self.round_label.config(text=f"Round: {self.round}")
        
        if self.round >= self.max_rounds:
            self.show_summary()
            
        self.update_graph()

    def update_graph(self):
        """Update the network visualization."""
        self.ax.clear()
        
        is_scam = self._sim_topic == "Financial Scam"
        colors = self.simulator.get_node_colors(is_scam)
        title = f"{'Scam' if is_scam else 'Fake News'} Spread - Round {self.round}"
        
        nx.draw(
            self.simulator.G,
            pos=self.simulator.get_graph_layout(),
            node_color=colors,
            ax=self.ax,
            with_labels=True,
            node_size=500
        )
        
        self.ax.set_title(title)
        self.canvas.draw()

    def show_summary(self):
        """Show simulation summary window."""
        summary_win = tk.Toplevel(self.root)
        summary_win.title("Simulation Summary")

        if self._sim_topic == "Financial Scam":
            self._show_scam_summary(summary_win)
        else:
            self._show_fake_news_summary(summary_win)

    def _show_scam_summary(self, summary_win):
        """Show summary for scam simulation."""
        scam_counts = [sum(1 for a in round_data if a) 
                      for round_data in self.scam_history]
        total_scammed = sum(scam_counts)

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(range(1, len(scam_counts)+1), scam_counts, marker='o', color='red')
        
        # Draw intervention lines
        for r in self.intervention_rounds:
            ax.axvline(
                r+1, color='green', linestyle='--',
                label='Intervention' if r == self.intervention_rounds[0] else ""
            )
            
        if self.intervention_rounds:
            ax.legend()
            
        ax.set_title("Scam Victims Over Time")
        ax.set_xlabel("Round")
        ax.set_ylabel("Victims")

        # Add to window
        canvas = FigureCanvasTkAgg(fig, master=summary_win)
        canvas.get_tk_widget().pack()

        # Add summary text
        text_frame = tk.Frame(summary_win)
        text_frame.pack(pady=5)
        summary = f"Topic: {self._sim_topic}\nTotal Scam Victims: {total_scammed}"
        tk.Label(text_frame, text=summary, justify="left", font=("Arial", 10)).pack()

        self._add_summary_controls(summary_win, fig)

    def _show_fake_news_summary(self, summary_win):
        """Show summary for fake news simulation."""
        victim_counts = [sum(1 for a in round_data if a) 
                        for round_data in self.round_history]
        total_shares = sum(victim_counts)
        
        # Calculate losses
        losses = self._calculate_losses(total_shares)
        
        # Create plots
        fig = self._create_summary_plots(victim_counts, losses)
        
        # Add to window
        canvas = FigureCanvasTkAgg(fig, master=summary_win)
        canvas.get_tk_widget().pack()

        # Add summary text
        text_frame = tk.Frame(summary_win)
        text_frame.pack(pady=5)
        summary = (
            f"Inferred Topic: {self._sim_topic}\n"
            f"News Context: {self.context_text.get()}\n"
            f"Juiciness: {self.juiciness.get()}\n"
            f"Intervention Used: {'Yes' if self.intervention else 'No'}\n"
            f"Total Shares: {total_shares}"
        )
        tk.Label(text_frame, text=summary, justify="left", font=("Arial", 10)).pack()

        self._add_summary_controls(summary_win, fig)

    def _calculate_losses(self, total_shares):
        """Calculate various loss probabilities."""
        juice_factor = self.juiciness.get() / 100.0
        financial_loss = reputational_loss = trust_loss = 0.05

        if self._sim_topic_category == 'Reputation-based Rumors':
            reputational_loss = min(1.0, 0.1 + 0.025 * total_shares * juice_factor)
        elif self._sim_topic_category in ['Phishing', 'Scare Tactics']:
            trust_loss = min(1.0, 0.15 + 0.035 * total_shares * 
                           (1 if not self.intervention else 0.5))
        elif (self._sim_topic_category == 'Policy Manipulation' or 
              self._sim_topic in ["Financial Scam", "Fake Scholarship Scam",
                                "Emergency VPN Update", "Student Aid Sabotage",
                                "University Database Hacked", "Data Breach"]):
            financial_loss = min(1.0, 0.2 + 0.012 * total_shares + 
                               0.007 * self.juiciness.get())

        return {
            "Financial Loss": financial_loss,
            "Reputation Loss": reputational_loss,
            "Trust Loss": trust_loss
        }

    def _create_summary_plots(self, victim_counts, losses):
        """Create summary visualization plots."""
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        
        # Victims over time plot
        axs[0].plot(range(1, len(victim_counts)+1), victim_counts, marker='o')
        for r in self.intervention_rounds:
            axs[0].axvline(
                r+1, color='green', linestyle='--',
                label='Intervention' if r == self.intervention_rounds[0] else ""
            )
        if self.intervention_rounds:
            axs[0].legend()
        axs[0].set_title("Victims Over Time")
        axs[0].set_xlabel("Round")
        axs[0].set_ylabel("Victims")

        # Loss probabilities plot
        bar_colors = ["#E15759", "#F28E2B", "#4E79A7"]
        bars = axs[1].bar(losses.keys(), losses.values(), color=bar_colors)
        axs[1].set_ylim(0, 1)
        axs[1].set_ylabel("Probability (0-1)")
        axs[1].set_title("Simulated Loss Probabilities (Higher = Worse)")

        # Add value labels
        for bar in bars:
            xpos = bar.get_x() + bar.get_width() / 2
            value = bar.get_height()
            axs[1].text(xpos, value - 0.03, f"{value:.2f}", 
                       ha='center', va='center', fontsize=10,
                       fontweight='bold', color='white')

        fig.tight_layout(rect=[0, 0.18, 1, 1])
        fig.subplots_adjust(bottom=0.28)
        fig.text(0.5, 0.13, 
                "Each bar shows the risk of each loss type from this simulation "
                "(0 = none, 1 = certain)",
                ha='center', fontsize=9, color='#333')

        return fig

    def _add_summary_controls(self, summary_win, fig):
        """Add control buttons to summary window."""
        def add_more_rounds():
            self.max_rounds += 10
            self.extra_rounds = True
            plt.close(fig)
            summary_win.destroy()
            self.root.after(500, self.automate_rounds)

        def end_simulation_now():
            plt.close(fig)
            summary_win.destroy()
            self.reset_btn.grid()

        button_frame = tk.Frame(summary_win)
        button_frame.pack(pady=8)
        
        more_btn = tk.Button(button_frame, text="Add 10 More Rounds",
                            command=add_more_rounds)
        more_btn.pack(side="left", padx=5)
        
        end_btn = tk.Button(button_frame, text="End Simulation Now",
                           command=end_simulation_now)
        end_btn.pack(side="left", padx=5)

        def on_close():
            plt.close(fig)
            summary_win.destroy()
            self.reset_btn.grid()
        summary_win.protocol("WM_DELETE_WINDOW", on_close)

    def reset_simulation(self):
        """Reset the simulation state."""
        self.round = 0
        self.max_rounds = 10
        self.reset_btn.grid_remove()
        self.run_btn.grid()
        self.ax.clear()
        self.ax.set_title("Fake News Spread - Round 0")
        self.canvas.draw()
        self.round_label.config(text="Round: 0")
