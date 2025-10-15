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
from pbm_simulator import PopulationSimulator
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
        self.canvas_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)

        # Initialize matplotlib figure
        # Create figure with fixed size and DPI
        self.fig = plt.figure(figsize=(12, 6), dpi=100, constrained_layout=False)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(expand=True, fill=tk.BOTH)
        
        # Create initial empty subplot (will be replaced in update_graph)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_visible(False)
        
        # Set initial title
        self.fig.suptitle("Fake News Spread - Round 0", y=0.95, fontsize=12)
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
        
        # Initialize simulators and start simulation
        self._init_simulators()

        # Update UI
        self.reset_btn.grid_remove()
        self.run_btn.grid_remove()
        self.update_graph()
        self.root.after(500, self.automate_rounds)

    def automate_rounds(self):
        """Automatically run simulation rounds."""
        if self.round >= self.max_rounds:
            return
            
        # Apply intervention at round 5
        if self.round == 5 and not self.intervention:
            self.intervention = True
            self.intervention_rounds.append(self.round)
            
        # Run the next round
        self.run_next_round()
        
        # Schedule the next round after a delay
        if self.round < self.max_rounds:  # Double check we haven't hit max rounds
            self.root.after(1000, self.automate_rounds)  # Increased delay to 1 second for better visualization

    def run_next_round(self):
        """Run the next simulation round for both models."""
        if self.round >= self.max_rounds:
            return

        # Run ABM simulation step
        if self._sim_topic == "Financial Scam":
            abm_result = self.abm_simulator.simulate_scam_round()
            self.scam_history.append(abm_result)
            self.abm_results['believer_counts'].append(sum(1 for x in abm_result if x))
        else:
            abm_result = self.abm_simulator.simulate_fake_news_round(
                self.juiciness.get() / 100.0,
                self._sim_topic_weight,
                self._sim_topic_category,
                self.intervention,
                getattr(self, 'extra_rounds', False)
            )
            self.round_history.append(abm_result)
            self.abm_results['believer_counts'].append(sum(1 for x in abm_result if x))

        # Run PBM simulation step and update rates if needed
        if self.intervention:
            self.pbm_simulator.adjust_rates(
                self._sim_topic_weight,
                self.juiciness.get() / 100.0,
                True
            )
        susceptible, believers, immune = self.pbm_simulator.simulate_step()
        self.pbm_results['susceptible'].append(susceptible)
        self.pbm_results['believers'].append(believers)
        self.pbm_results['immune'].append(immune)

        # Update round counter and UI
        self.round += 1
        self.round_label.config(text=f"Round: {self.round}")
        
        # Update visualization
        self.update_graph()
        
        # Show summary if we've reached max rounds
        if self.round >= self.max_rounds:
            self.show_summary()

    def update_graph(self):
        """Update the network visualization."""
        # Clear the entire figure
        self.fig.clear()
        
        # Create a new gridspec with proper spacing
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
        
        # Set the main title with proper spacing
        self.fig.suptitle(f"Fake News Spread - Round {self.round}", y=0.95, fontsize=12)
        
        # Create subplots with the gridspec
        ax1 = self.fig.add_subplot(gs[0])  # ABM subplot
        ax2 = self.fig.add_subplot(gs[1])  # PBM subplot
        
        # Setup colors and patches for legends
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='Believers')
        blue_patch = mpatches.Patch(color='blue', label='Non-believers')
        
        # ABM Network visualization (left)
        is_scam = self._sim_topic == "Financial Scam"
        colors = self.abm_simulator.get_node_colors(is_scam)
        
        # Calculate node positions if not already done
        if not hasattr(self, 'pos') or self.pos is None:
            self.pos = nx.spring_layout(self.abm_simulator.G)
            
        # Get or create fixed positions for nodes
        if not hasattr(self, 'pos') or self.pos is None:
            self.pos = nx.spring_layout(self.abm_simulator.G, k=1, iterations=50, seed=42)
            
        # Draw the ABM network with enhanced styling
        nx.draw_networkx_edges(self.abm_simulator.G, 
                             pos=self.pos,
                             edge_color='gray',
                             width=1.0,
                             alpha=0.5,
                             ax=ax1)
                             
        # Draw nodes with better visibility
        nx.draw_networkx_nodes(self.abm_simulator.G,
                             pos=self.pos,
                             node_color=colors,
                             node_size=700,  # Increased node size
                             edgecolors='black',  # Black border for nodes
                             linewidths=1.5,
                             alpha=0.9,
                             ax=ax1)
                             
        # Add labels with better visibility
        nx.draw_networkx_labels(self.abm_simulator.G,
                              pos=self.pos,
                              font_size=10,
                              font_weight='bold',
                              font_color='black',
                              ax=ax1)
        
        # Add legend and border for ABM plot
        ax1.legend(handles=[red_patch, blue_patch],
                  loc='upper right',
                  title="Agent States",
                  title_fontsize=10,
                  fontsize=9,
                  framealpha=0.9,
                  edgecolor='black')
        
        # Set title and style plot
        ax1.set_title("Agent-Based Model (ABM)", pad=10, fontsize=12, fontweight='bold')
        ax1.set_facecolor('#f8f8f8')  # Light gray background
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('black')
            
        # Ensure proper aspect ratio and margins
        ax1.set_aspect('equal')
        ax1.margins(0.15)
        
        # PBM visualization (right)
        if hasattr(self, 'pbm_simulator'):
            # Get total population and current data
            total_population = len(self.abm_simulator.G.nodes())
            pbm_history = self.pbm_simulator.get_history()
            
            # Get number of rounds to plot
            current_round = self.round + 1
            rounds = range(current_round)
            
            # Plot PBM data
            pbm_believers = pbm_history['believers'][:current_round]
            if pbm_believers:  # Only plot if we have data
                # Plot susceptible population (blue dashed)
                pbm_susceptible = pbm_history['susceptible'][:current_round]
                ax2.plot(rounds, pbm_susceptible, 'b--', label='Susceptible', linewidth=2)
                ax2.scatter(rounds, pbm_susceptible, color='blue', s=50, alpha=0.6, marker='o')
                
                # Plot believer population (red solid)
                ax2.plot(rounds, pbm_believers, 'r-', label='Believers', linewidth=2)
                ax2.scatter(rounds, pbm_believers, color='red', s=50, alpha=0.6, marker='o')
                
                # Set axis limits for better visualization
                ax2.set_ylim(0, total_population * 1.1)
                ax2.set_xlim(-0.5, max(9.5, current_round - 0.5))
                
                # Add grid and labels
                ax2.grid(True, alpha=0.2, linestyle='--')
                ax2.set_xlabel('Round', fontsize=9)
                ax2.set_ylabel('Population', fontsize=9)
                ax2.set_title('Population-Based Model', pad=10, fontsize=10, fontweight='bold')
                ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
                
                # Add spines
                for spine in ax2.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.5)
                    spine.set_color('black')
        
        # Add titles
        ax1.set_title("Agent-Based Model (ABM)", pad=10, fontsize=10, fontweight='bold')
        ax2.set_title("Comparison of Believer Dynamics: ABM and PBM", pad=10, fontsize=10, fontweight='bold')
        
        # Add borders to both plots
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
                spine.set_color('black')
        
        # Add legend to ABM plot
        ax1.legend(handles=[red_patch, blue_patch],
                  loc='upper right',
                  fontsize=8,
                  framealpha=0.9)
        
        # Draw the updated canvas
        self.canvas.draw()
        
        # PBM visualization (right plot styling)
        if hasattr(self, 'pbm_simulator'):
            ax2.set_title("Population-Based Model", pad=10, fontsize=10, fontweight='bold')
            ax2.set_xlabel("Round", fontsize=9)
            ax2.set_ylabel("Population Size", fontsize=9)
            ax2.tick_params(axis='both', which='major', labelsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Update legend with improved styling
            ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
            ax2.grid(True, alpha=0.3)
            
            # Set y-axis limits to keep scale consistent
            ax2.set_ylim(0, len(self.abm_simulator.G.nodes()) * 1.1)
        
        # Adjust layout with specific padding
        self.fig.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.9)
        
        # Draw canvas
        self.canvas.draw()

    def show_summary(self):
        """Show simulation summary and comparison."""
        from comparison_viz import ComparisonVisualizer
        
        # Create summary window
        summary_win = tk.Toplevel(self.root)
        summary_win.title("Simulation Summary")
        summary_win.geometry("600x400")
        
        # Create frames for different sections
        summary_frame = tk.Frame(summary_win)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Show topic-specific summary
        if self._sim_topic == "Financial Scam":
            self._show_scam_summary(summary_frame)
        else:
            self._show_fake_news_summary(summary_frame)
        
        # Create comparison window
        comparison_win = tk.Toplevel(self.root)
        comparison_win.title("ABM vs PBM Comparison")
        comparison_win.geometry("1200x800")
        
        # Create visualization
        viz = ComparisonVisualizer(
            self.abm_results,
            self.pbm_results,
            self.context_text.get()
        )
        viz.show_comparison(comparison_win)
        
        # Position windows
        summary_win.geometry("+100+100")  # Position summary window
        comparison_win.geometry("+720+100")  # Position comparison window to the right
        
        def add_more_rounds():
            self.max_rounds += 10
            self.extra_rounds = True
            plt.close('all')
            summary_win.destroy()
            comparison_win.destroy()
            self.root.after(500, self.automate_rounds)

        def end_simulation_now():
            plt.close('all')
            summary_win.destroy()
            comparison_win.destroy()
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
            plt.close('all')
            summary_win.destroy()
            comparison_win.destroy()
            self.reset_btn.grid()
        summary_win.protocol("WM_DELETE_WINDOW", on_close)

    def _show_scam_summary(self, summary_frame):
        """Show summary for scam simulation."""
        scam_counts = [sum(1 for a in round_data if a) 
                      for round_data in self.scam_history]
        total_scammed = sum(scam_counts)

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(range(1, len(scam_counts)+1), scam_counts, marker='o', color='red', label='Victims')
        
        # Draw intervention lines
        for r in self.intervention_rounds:
            ax.axvline(
                r+1, color='green', linestyle='--',
                label='Intervention' if r == self.intervention_rounds[0] else None
            )
            
        # Always add legend if we have data
        if len(scam_counts) > 0:
            ax.legend(loc='upper left')
            
        ax.set_title("Scam Victims Over Time")
        ax.set_xlabel("Round")
        ax.set_ylabel("Victims")
        ax.grid(True, alpha=0.3)

        # Add plot to summary frame
        canvas = FigureCanvasTkAgg(fig, master=summary_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add summary text
        text_frame = tk.Frame(summary_frame)
        text_frame.pack(fill=tk.X, pady=5)
        summary = (
            f"Topic: {self._sim_topic}\n"
            f"Total Scam Victims: {total_scammed}\n"
            f"Peak Victims: {max(scam_counts) if scam_counts else 0}\n"
            f"Final Round State: {scam_counts[-1] if scam_counts else 0} victims"
        )

    def _show_fake_news_summary(self, summary_frame):
        """Show summary for fake news simulation."""
        # Calculate statistics
        victim_counts = [sum(1 for a in round_data if a) 
                        for round_data in self.round_history]
        total_shares = sum(victim_counts)
        peak_believers = max(victim_counts) if victim_counts else 0
        final_believers = victim_counts[-1] if victim_counts else 0
        peak_round = victim_counts.index(peak_believers) + 1 if victim_counts else 0
        total_agents = len(self.abm_simulator.G.nodes())
        
        # Calculate PBM statistics
        pbm_history = self.pbm_simulator.get_history()
        pbm_believers = pbm_history['believers']
        pbm_peak = max(pbm_believers)
        pbm_peak_round = pbm_believers.index(pbm_peak) + 1
        pbm_final = pbm_believers[-1]
        
        # Calculate intervention effects if applied
        pre_intervention = None
        post_intervention = None
        if self.intervention and len(victim_counts) > 5:
            pre_intervention = sum(victim_counts[:5]) / 5
            post_intervention = sum(victim_counts[5:10]) / 5
        
        # Calculate losses
        losses = self._calculate_losses(total_shares)
        
        # Create main info frame
        info_frame = tk.Frame(summary_frame, relief=tk.RIDGE, bd=2)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Add context section
        tk.Label(info_frame, text="News Context Analysis", font=('Arial', 12, 'bold')).pack(pady=(10,5))
        context_text = (
            f"Topic Category: {self._sim_topic_category}\n"
            f"Context: {self.context_text.get()}\n"
            f"Juiciness Score: {self.juiciness.get()}/100\n"
            f"Topic Weight: {self._sim_topic_weight:.2f}"
        )
        tk.Label(info_frame, text=context_text, justify=tk.LEFT, font=('Arial', 10), wraplength=500).pack(padx=10, pady=5)
        
        # Add model comparison section
        tk.Label(info_frame, text="Model Comparison", font=('Arial', 12, 'bold')).pack(pady=(10,5))
        comparison_text = (
            f"Agent-Based Model (ABM):\n"
            f"• Peak Believers: {peak_believers} ({peak_believers/total_agents*100:.1f}%) at round {peak_round}\n"
            f"• Final State: {final_believers} ({final_believers/total_agents*100:.1f}%)\n"
            f"• Total Shares: {total_shares}\n\n"
            f"Population-Based Model (PBM):\n"
            f"• Peak Believers: {int(pbm_peak)} ({pbm_peak/total_agents*100:.1f}%) at round {pbm_peak_round}\n"
            f"• Final State: {int(pbm_final)} ({pbm_final/total_agents*100:.1f}%)\n"
            f"• Diffusion Rate: {pbm_peak/total_agents:.3f}"
        )
        tk.Label(info_frame, text=comparison_text, justify=tk.LEFT, font=('Arial', 10)).pack(padx=10, pady=5)
        
        # Add intervention effects if applicable
        if pre_intervention is not None:
            tk.Label(info_frame, text="Intervention Analysis", font=('Arial', 12, 'bold')).pack(pady=(10,5))
            intervention_text = (
                f"Pre-intervention average: {pre_intervention:.1f} believers/round\n"
                f"Post-intervention average: {post_intervention:.1f} believers/round\n"
                f"Effect: {((post_intervention - pre_intervention) / pre_intervention * 100):.1f}% change"
            )
            tk.Label(info_frame, text=intervention_text, justify=tk.LEFT, font=('Arial', 10)).pack(padx=10, pady=5)
        
        # Add risk analysis section
        tk.Label(info_frame, text="Risk Analysis", font=('Arial', 12, 'bold')).pack(pady=(10,5))
        risk_text = (
            f"Financial Impact Risk: {losses['Financial Loss']*100:.1f}%\n"
            f"Reputational Damage Risk: {losses['Reputation Loss']*100:.1f}%\n"
            f"Trust Erosion Risk: {losses['Trust Loss']*100:.1f}%"
        )
        tk.Label(info_frame, text=risk_text, justify=tk.LEFT, font=('Arial', 10)).pack(padx=10, pady=5)
        
        # Add recommendations
        tk.Label(info_frame, text="Recommendations", font=('Arial', 12, 'bold')).pack(pady=(10,5))
        recommendations = self._generate_recommendations(losses, peak_believers/total_agents)
        tk.Label(info_frame, text=recommendations, justify=tk.LEFT, font=('Arial', 10), wraplength=500).pack(padx=10, pady=(5,10))

    def _generate_recommendations(self, losses, peak_spread_rate):
        """Generate recommendations based on simulation results."""
        recommendations = []
        
        # Spread rate based recommendations
        if peak_spread_rate > 0.7:
            recommendations.append("URGENT: Immediate response required - viral spread detected")
        elif peak_spread_rate > 0.4:
            recommendations.append("High spread rate - prioritize containment measures")
        
        # Risk-based recommendations
        max_risk = max(losses.items(), key=lambda x: x[1])
        if max_risk[1] > 0.6:
            recommendations.append(f"Critical {max_risk[0].lower()} risk - implement mitigation strategies")
        
        # General recommendations
        if self.intervention:
            recommendations.append("Continue monitoring intervention effectiveness")
        else:
            recommendations.append("Consider implementing intervention measures")
            
        return "\n".join(f"• {r}" for r in recommendations)

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

    def _init_simulators(self):
        """Initialize both ABM and PBM simulators."""
        # Reset simulation state
        self.round = 0
        self.round_label.config(text=f"Round: {self.round}")
        self.intervention = False
        self.extra_rounds = False
        self.round_history = []
        self.scam_history = []
        self.intervention_rounds = []

        # Initialize ABM simulator
        num_agents = len(self.df_agents)
        agent_data = None if self.agent_source.get() == "Random" else self.df_agents
        self.abm_simulator = FakeNewsSimulator(num_agents, agent_data)
        self.abm_simulator.seed_initial_state(is_scam=(self._sim_topic == "Financial Scam"))

        # Initialize PBM simulator with the same number of agents
        self.pbm_simulator = PopulationSimulator(num_agents)
        self.pbm_simulator.adjust_rates(
            self._sim_topic_weight,
            self.juiciness.get() / 100.0,
            self.intervention
        )

        # Initialize results containers with initial state
        if self._sim_topic == "Financial Scam":
            initial_believers = sum(1 for i in self.abm_simulator.agent_states 
                                if self.abm_simulator.agent_states[i]['scammed'])
        else:
            initial_believers = sum(1 for i in self.abm_simulator.agent_states 
                                if self.abm_simulator.agent_states[i]['shared'])
        
        self.abm_results = {
            'believer_counts': [initial_believers],  # Start with 2 initial believers
            'total_agents': num_agents
        }
        
        # Initialize PBM results with same initial state
        self.pbm_results = {
            'susceptible': [num_agents - 2],  # Initial susceptible
            'believers': [2],                 # Initial believers
            'immune': [0]                     # Initial immune
        }

        # Update UI
        self.reset_btn.grid_remove()
        self.run_btn.grid_remove()
        self.update_graph()
        self.root.after(500, self.automate_rounds)

    def run_next_round(self):
        """Run the next simulation round for both models."""
        if self.round >= self.max_rounds:
            return

        # Run ABM simulation step
        if self._sim_topic == "Financial Scam":
            abm_result = self.abm_simulator.simulate_scam_round()
            self.scam_history.append(abm_result)
            self.abm_results['believer_counts'].append(
                sum(1 for x in abm_result if x)
            )
        else:
            abm_result = self.abm_simulator.simulate_fake_news_round(
                self.juiciness.get() / 100.0,
                self._sim_topic_weight,
                self._sim_topic_category,
                self.intervention,
                getattr(self, 'extra_rounds', False)
            )
            self.round_history.append(abm_result)
            # Store the actual count of believers
            believer_count = sum(1 for x in abm_result if x)
            self.abm_results['believer_counts'].append(believer_count)

        # Run PBM simulation step
        if self.intervention:
            self.pbm_simulator.adjust_rates(
                self._sim_topic_weight,
                self.juiciness.get() / 100.0,
                True
            )
        susceptible, believers, immune = self.pbm_simulator.simulate_step()
        self.pbm_results['susceptible'].append(susceptible)
        self.pbm_results['believers'].append(believers)
        self.pbm_results['immune'].append(immune)

        # Update round counter and UI
        self.round += 1
        self.round_label.config(text=f"Round: {self.round}")
        
        if self.round >= self.max_rounds:
            self.show_summary()
            
        self.update_graph()

    def show_summary(self):
        """Show simulation summary and comparison."""
        from comparison_viz import ComparisonVisualizer
        
        # Prepare the data for visualization
        history = self.scam_history if self._sim_topic == "Financial Scam" else self.round_history
        abm_results = {
            'believer_counts': history,  # Pass the full history array
            'total_agents': len(self.abm_simulator.G.nodes())
        }
        pbm_results = self.pbm_simulator.get_history()
        
        # Create summary window
        summary_win = tk.Toplevel(self.root)
        summary_win.title("Simulation Summary")
        summary_win.geometry("600x600")  # Made taller to fit all content
        summary_win.minsize(600, 600)    # Set minimum size
        
        # Create main container frame with scrolling
        main_container = tk.Frame(summary_win)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Add a canvas for scrolling
        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Create frames for different sections
        summary_frame = tk.Frame(scrollable_frame)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Show topic-specific summary
        if self._sim_topic == "Financial Scam":
            self._show_scam_summary(summary_frame)
        else:
            self._show_fake_news_summary(summary_frame)
        
        # Create control buttons
        button_frame = tk.Frame(summary_frame)
        button_frame.pack(pady=8)
        
        # Create comparison window
        comparison_win = tk.Toplevel(self.root)
        comparison_win.title("ABM vs PBM Comparison")
        comparison_win.geometry("1200x800")
        comparison_win.minsize(1000, 600)  # Set minimum size
        
        # Create visualization
        viz = ComparisonVisualizer(
            abm_results,
            pbm_results,
            self.context_text.get()
        )
        viz.show_comparison(comparison_win)
        
        # Position windows
        summary_win.geometry("+100+100")     # Position summary window
        comparison_win.geometry("+720+100")  # Position comparison window to the right
        
        def add_more_rounds():
            self.max_rounds += 10
            self.extra_rounds = True
            plt.close('all')
            summary_win.destroy()
            comparison_win.destroy()
            self.root.after(500, self.automate_rounds)
        
        def end_simulation_now():
            plt.close('all')
            summary_win.destroy()
            comparison_win.destroy()
            self.reset_btn.grid()
        
        more_btn = tk.Button(button_frame, text="Add 10 More Rounds",
                            command=add_more_rounds)
        more_btn.pack(side="left", padx=5)
        
        end_btn = tk.Button(button_frame, text="End Simulation Now",
                           command=end_simulation_now)
        end_btn.pack(side="left", padx=5)
        
        # Handle window closing
        def on_summary_close():
            plt.close('all')
            summary_win.destroy()
            comparison_win.destroy()
            self.reset_btn.grid()
        
        def on_comparison_close():
            plt.close('all')
            summary_win.destroy()
            comparison_win.destroy()
            self.reset_btn.grid()
        
        # Set up window closing protocols
        summary_win.protocol("WM_DELETE_WINDOW", on_summary_close)
        comparison_win.protocol("WM_DELETE_WINDOW", on_comparison_close)
        
        # Raise windows to front
        summary_win.lift()
        comparison_win.lift()
        
        # Update windows to ensure they're drawn
        summary_win.update()
        comparison_win.update()

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
