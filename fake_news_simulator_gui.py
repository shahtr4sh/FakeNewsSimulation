import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import random
import os
import sys

# Load agent profiles
script_dir = os.path.dirname(os.path.abspath(__file__))
agents_path = os.path.join(script_dir, "data", "agent_profiles.csv")
df_agents = pd.read_csv(agents_path)

# Constants
NUM_AGENTS = len(df_agents)
THETA_B = 0.5
THETA_S = 0.7
A = 0.6
BASE_GAMMA = 0.3
LAMBDA_DECAY = 0.1
DELTA_T = 1.0
BETA1, BETA2, BETA3, BETA4 = 0.4, 0.3, 0.2, 0.3

TOPICS = {
    "Ransomware Alert": 1.0,
    "Data Breach": 0.9,
    "Zero-day Exploit": 1.2,
    "Phishing Campaign": 1.1,
    "Financial Scam": 1.0,
    "University Database Hacked": 1.3,
    "Fake Scholarship Scam": 1.4,
    "Emergency VPN Update": 1.2,
    "Email Server Compromise": 1.2,
    "Lecturer Scandal": 1.3,
    "WiFi Surveillance Rumor": 1.0,
    "Fake Exam Timetable": 1.4,
    "Student Aid Sabotage": 1.1,
    "AI Surveillance Ethics": 1.0,
    "Campus Virus Leak": 1.5
}

# Topic categories and their main agent trait influence
TOPIC_CATEGORIES = {
    "Phishing": [
        "Phishing Campaign", "Fake Scholarship Scam", "Emergency VPN Update", "Email Server Compromise"
    ],
    "Reputation-based Rumors": [
        "Lecturer Scandal", "WiFi Surveillance Rumor", "Fake Exam Timetable", "AI Surveillance Ethics"
    ],
    "Policy Manipulation": [
        "Student Aid Sabotage", "Ransomware Alert", "Zero-day Exploit", "Data Breach"
    ],
    "Scare Tactics": [
        "Financial Scam", "University Database Hacked", "Campus Virus Leak"
    ]
}

# Map category to main trait
CATEGORY_TRAIT = {
    "Phishing": "trust_level",
    "Reputation-based Rumors": "confirmation_bias",
    "Policy Manipulation": "critical_thinking",
    "Scare Tactics": "emotional_susceptibility"
}

class FakeNewsSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fake News Circulation Simulator")
        self.topic = tk.StringVar(value="Ransomware Alert")
        self.juiciness = tk.IntVar(value=50)
        self.agent_source = tk.StringVar(value="CSV")  # New: agent source selection
        self.context_text = tk.StringVar(value="")  # News context
        self.round = 0
        self.max_rounds = 10
        self.intervention = False
        self.agent_states = {}
        self.round_history = []  # Track victims each round
        self.scam_history = []   # Track scam victims each round
        self.intervention_rounds = []  # Track rounds when intervention is enabled
        self.G = None
        self._graph_pos = None
        self.setup_ui()
        # Do not call self.init_simulation() here

    def setup_ui(self):
        config_frame = tk.Frame(self.root)
        config_frame.pack(pady=10)

        tk.Label(config_frame, text="Fake News Context:").grid(row=0, column=0, padx=5)
        self.context_entry = tk.Entry(config_frame, textvariable=self.context_text, width=40)
        self.context_entry.grid(row=0, column=1, padx=5)

        # Remove duplicate context entry (now at row 0)

        tk.Label(config_frame, text="Juiciness:").grid(row=2, column=0, padx=5)
        self.juiciness_scale = tk.Scale(config_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.juiciness)
        self.juiciness_scale.grid(row=2, column=1, padx=5)

        # Agent source selection
        tk.Label(config_frame, text="Agent Source:").grid(row=3, column=0, padx=5)
        agent_source_menu = ttk.Combobox(config_frame, textvariable=self.agent_source, values=["CSV", "Random"], state="readonly")
        agent_source_menu.grid(row=3, column=1, padx=5)

        self.round_label = tk.Label(config_frame, text=f"Round: {self.round}")
        self.round_label.grid(row=0, column=2, padx=10)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=(0, 10))

        self.run_btn = tk.Button(button_frame, text="Run Simulation", command=self.init_simulation)
        self.run_btn.grid(row=0, column=0, padx=5)

        self.reset_btn = tk.Button(button_frame, text="Run Another Simulation", command=self.reset_simulation)
        self.reset_btn.grid(row=0, column=1, padx=5)
        self.reset_btn.grid_remove()  # Hide initially

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack()

        # Set standard figure size for network graph visualization
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack()

        # Configure the figure and axis for consistent sizing
        self.fig.set_tight_layout(True)
        self.ax.set_title("Fake News Spread - Round 0")
        self.canvas.draw()

    def init_simulation(self):
        # Analyze context and set juiciness
        context = self.context_text.get().lower()
        juiciness_score = self.analyze_context_juiciness(context)
        self.juiciness.set(juiciness_score)
        self.juiciness_scale.set(juiciness_score)

        # Infer topic for weighting and display
        topic, topic_weight, topic_category = self.infer_topic_from_context(context)
        self._sim_topic = topic
        self._sim_topic_weight = topic_weight
        self._sim_topic_category = topic_category  # New: store topic category

        self.round = 0
        self.round_label.config(text=f"Round: {self.round}")
        if self.round >= self.max_rounds:
            self.show_summary()
        self.intervention = False
        self.extra_rounds = False  # Reset extra rounds flag
        self.G = nx.erdos_renyi_graph(NUM_AGENTS, 0.3)
        self.agent_states = {}
        self.round_history = []  # Track victims each round
        self.scam_history = []   # Track scam victims each round
        self.intervention_rounds = []  # Reset intervention rounds
        
        # --- PBM initialization (independent model) ---
        self.pbm_believers = [0.3]  # Start with 30% believers (normalized 0–1)
        self.pbm_history = []       # To store PBM believers over time

        # Choose agent source
        if self.agent_source.get() == "Random":
            agent_df = self.generate_random_agents(NUM_AGENTS)
        else:
            agent_df = df_agents

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
        # Initial seeding
        if topic == "Financial Scam":
            for node in random.sample(list(self.agent_states.keys()), 2):
                self.agent_states[node]['scammed'] = True
        else:
            for node in random.sample(list(self.agent_states.keys()), 2):
                self.agent_states[node]['belief'] = 1.0
                self.agent_states[node]['shared'] = True
        # Precompute layout with increased spacing
        self._graph_pos = nx.spring_layout(self.G, k=1.5, iterations=50, seed=42)  # k=1.5 increases spacing
        self.reset_btn.grid_remove()
        self.run_btn.grid_remove()
        self.update_graph()
        self.root.after(500, self.automate_rounds)
    def infer_topic_from_context(self, context):
        # Heuristic: match keywords to topics for weighting and category
        topic_keywords = {
            "Ransomware Alert": ["ransomware", "ransom"],
            "Data Breach": ["breach", "leak", "data"],
            "Zero-day Exploit": ["zero-day", "exploit", "vulnerability"],
            "Phishing Campaign": ["phishing", "phish", "email", "campaign"],
            "Financial Scam": ["scam", "fraud", "money", "bank", "finance", "financial"],
            "University Database Hacked": ["university database hacked", "student records", "dark web"],
            "Fake Scholarship Scam": ["scholarship", "register now", "uum id", "grant"],
            "Emergency VPN Update": ["vpn update", "block access"],
            "Email Server Compromise": ["email server", "compromised", "password reset"],
            "Lecturer Scandal": ["lecturer", "plagiarism", "scandal", "leaked documents"],
            "WiFi Surveillance Rumor": ["wifi", "monitored", "proxy", "encrypted browsers"],
            "Fake Exam Timetable": ["fake exam", "timetable", "verify"],
            "Student Aid Sabotage": ["financial aid", "delayed", "sabotage"],
            "AI Surveillance Ethics": ["ai surveillance", "privacy rights", "violated"],
            "Campus Virus Leak": ["virus", "covid", "authorities silent"]
        }
        for topic, keywords in topic_keywords.items():
            for word in keywords:
                if word in context:
                    # Find category
                    for cat, topic_list in TOPIC_CATEGORIES.items():
                        if topic in topic_list:
                            return topic, TOPICS[topic], cat
                    return topic, TOPICS[topic], None
        return "Other", 1.0, None  # Default weight and no category

    def analyze_context_juiciness(self, context):
        # Refined heuristic for juiciness based on urgency, emotional charge, and scenario
        very_high = [
            "virus", "covid", "authorities silent", "for sale", "emergency", "block access", "identity theft", "sabotage", "privacy violated"
        ]
        high = [
            "hacked", "hack", "leaked", "leak", "scandal", "urgent", "phishing", "compromised", "password reset", "plagiarism", "fake exam", "panic", "malware", "fraud", "scam", "stolen", "for sale", "dark web", "delayed", "failure to comply"
        ]
        med_high = [
            "vpn update", "proxy", "dissatisfaction", "distrust", "sabotage", "privacy", "protest", "mass sharing", "grant", "scholarship", "register now", "financial aid"
        ]
        medium = [
            "warning", "breach", "incident", "risk", "threat", "phishing", "data", "security", "fake", "rumor", "monitored", "encrypted", "browser", "campus wifi", "ai surveillance", "ethical", "concerns"
        ]
        low = [
            "update", "news", "report", "statement", "info", "information", "timetable", "verify"
        ]
        context = context.lower()
        score = 20  # default
        for word in very_high:
            if word in context:
                return 98
        for word in high:
            if word in context:
                return 90
        for word in med_high:
            if word in context:
                return 75
        for word in medium:
            if word in context:
                return 60
        for word in low:
            if word in context:
                return 30
        return score

    def generate_random_agents(self, n):
        # Generate a DataFrame with n random agents
        data = {
            'confirmation_bias': np.random.uniform(0, 1, n),
            'emotional_susceptibility': np.random.uniform(0, 1, n),
            'trust_level': np.random.uniform(0, 1, n),
            'critical_thinking': np.random.uniform(0, 1, n),
            'fact_check_signal': np.random.uniform(0, 1, n),
            'risk_perception': np.random.uniform(0, 1, n)
        }
        return pd.DataFrame(data)

    def automate_rounds(self):
        if self.round < self.max_rounds:
            # Automatically enable intervention at round 5
            if self.round == 5 and not self.intervention:
                self.intervention = True
                self.intervention_rounds.append(self.round)
            self.run_next_round()
            self.root.after(500, self.automate_rounds)

    def run_next_round(self):
        if self.round >= self.max_rounds:
            return

        nodes = list(self.G.nodes())
        if getattr(self, '_sim_topic', None) == "Financial Scam":
            # Scam simulation logic (unchanged)
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
            self.scam_history.append([self.agent_states[i]['scammed'] for i in nodes])
        else:
            # Refined fake news logic with much stronger intervention effect on sharing
            juice_factor = self.juiciness.get() / 100.0
            topic_weight = getattr(self, '_sim_topic_weight', 1.0)
            topic_category = getattr(self, '_sim_topic_category', None)
            gamma = BASE_GAMMA * (1.5 if self.intervention else 1)  # Reduced from 2x to 1.5x
            shared = {i: self.agent_states[i]['shared'] for i in nodes}
            beliefs = {i: self.agent_states[i]['belief'] for i in nodes}
            new_beliefs = {}
            # Adjust trait weights based on topic category
            weights = {
                'confirmation_bias': BETA1,
                'emotional_susceptibility': BETA2,
                'trust_level': BETA3,
                'critical_thinking': BETA4
            }
            if topic_category and topic_category in CATEGORY_TRAIT:
                main_trait = CATEGORY_TRAIT[topic_category]
                for k in weights:
                    if k == main_trait:
                        weights[k] *= 1.7
                    else:
                        weights[k] *= 0.8
            # Dynamic thresholds and decay for high juiciness
            if juice_factor >= 0.95:
                theta_b = 0.35  # Lower belief threshold for viral news
                theta_s = 0.55  # Lower sharing threshold
                lambda_decay = 0.03  # Reduce decay for viral news
                exposure_alpha = 1.2  # Steepen exposure curve
            elif juice_factor >= 0.8:
                theta_b = 0.42
                theta_s = 0.62
                lambda_decay = 0.06
                exposure_alpha = 1.0
            else:
                theta_b = THETA_B
                theta_s = THETA_S
                lambda_decay = LAMBDA_DECAY
                exposure_alpha = A
            # More moderate intervention effect
            if self.intervention:
                theta_s *= 1.15  # 15% increase in sharing threshold (instead of fixed 0.92)
                lambda_decay *= 1.3  # 30% faster decay (instead of fixed 0.25)
            # If extra rounds, make decay stronger but not extreme
            if hasattr(self, 'extra_rounds') and self.extra_rounds:
                lambda_decay *= 1.2  # 20% additional decay in extra rounds
            for i in nodes:
                neighbors = list(self.G.neighbors(i))
                exposures = sum(shared[j] for j in neighbors)
                a = self.agent_states[i]
                if hasattr(self, 'extra_rounds') and self.extra_rounds:
                    # In extra rounds, always apply strong decay regardless of exposures
                    new_belief = beliefs[i] * np.exp(-lambda_decay)
                else:
                    if exposures == 0:
                        new_belief = beliefs[i] * np.exp(-lambda_decay)
                    else:
                        P_star = (
                            weights['confirmation_bias'] * a['confirmation_bias'] +
                            weights['emotional_susceptibility'] * a['emotional_susceptibility'] * (topic_weight + 0.2 * juice_factor) +
                            weights['trust_level'] * a['trust_level'] +
                            0.15 * juice_factor -
                            weights['critical_thinking'] * a['critical_thinking']
                        )
                        exposure_factor = (1 / (1 + np.exp(-exposure_alpha * exposures))) * (1 + 1.2 * juice_factor)
                        P_believe = exposure_factor * P_star - gamma * a['fact_check_signal']
                        new_belief = 1.0 if P_believe > theta_b else beliefs[i] * np.exp(-lambda_decay)
                new_beliefs[i] = new_belief
            for i in nodes:
                self.agent_states[i]['belief'] = new_beliefs[i]
                self.agent_states[i]['shared'] = new_beliefs[i] > theta_s
            self.round_history.append([self.agent_states[i]['shared'] for i in nodes])
        
            # --- PBM update (independent logistic model) ---
            beta = 0.8 * (self.juiciness.get() / 100.0) * self._sim_topic_weight  # strong dependence on topic & juiciness
            gamma = 0.05 + (0.05 if self.intervention else 0.02)  # Decay rate higher with intervention
            B_t = self.pbm_believers[-1]
            B_next = B_t + beta * B_t * (1 - B_t) - gamma * B_t  # Logistic diffusion equation
            B_next = max(0, min(1, B_next))  # Keep value within [0,1]
            self.pbm_believers.append(B_next)
            self.pbm_history.append(B_next)
        
        self.round += 1
        self.round_label.config(text=f"Round: {self.round}")
        if self.round >= self.max_rounds:
            self.show_summary()
        self.update_graph()

    def update_graph(self):
        self.ax.clear()
        if self.topic.get() == "Financial Scam":
            colors = ['red' if self.agent_states[n]['scammed'] else 'blue' for n in self.G.nodes()]
            title = f"Scam Spread - Round {self.round}"
        else:
            colors = ['red' if self.agent_states[n]['shared'] else 'blue' for n in self.G.nodes()]
            title = f"Fake News Spread - Round {self.round}"
        # Use precomputed layout
        pos = getattr(self, '_graph_pos', None)
        if pos is None:
            pos = nx.spring_layout(self.G, seed=42)
        # Draw network with increased spacing and better visibility
        nx.draw(self.G, pos=pos, node_color=colors, ax=self.ax, with_labels=True,
                node_size=700,  # Larger nodes
                alpha=0.8,      # More opaque nodes
                width=0.8,      # Slightly thinner edges
                edge_color='lightgray',  # Lighter edges
                font_size=9,    # Slightly larger font
                font_weight='bold',  # Bold labels
                label=None)     # No legend needed for single color groups
        
        # Set consistent axis limits with padding
        self.ax.margins(0.15)  # Add padding around the network
        self.ax.set_title(title)
        self.ax.set_aspect('equal')
        self.fig.set_tight_layout(True)
        self.canvas.draw()

    def reset_simulation(self):
        # Removed next_btn and intervene_btn since they no longer exist
        self.round = 0
        self.max_rounds = 10 
        self.reset_btn.grid_remove()  # Hide reset button
        self.run_btn.grid()           # Show run button
        self.ax.clear()
        self.ax.set_title("Fake News Spread - Round 0")
        self.canvas.draw()
        self.round_label.config(text="Round: 0")
        # Do not call init_simulation here

    def show_summary(self):
        summary_win = tk.Toplevel(self.root)    
        summary_win.title("Simulation Summary")

        # ==== 1. Data preparation ====
        topic = getattr(self, '_sim_topic', 'Unknown')
        topic_category = getattr(self, '_sim_topic_category', 'Unknown')
        context_text = self.context_text.get()
        juiciness_score = self.juiciness.get()
        topic_weight = getattr(self, '_sim_topic_weight', 1.0)
        intervention_label = "Yes" if self.intervention else "No"

        # --- ABM results ---
        if self.round_history:
            abm_counts = [sum(1 for a in self.round_history[r] if a) for r in range(len(self.round_history))]
            abm_peak = max(abm_counts)
            abm_peak_round = abm_counts.index(abm_peak) + 1
            abm_final = abm_counts[-1]
            abm_total_shares = sum(abm_counts)
        else:
            abm_counts = []
            abm_peak = abm_peak_round = abm_final = abm_total_shares = 0

        # --- PBM results (from independent model) ---
        if self.pbm_history:
            pbm_values = np.array(self.pbm_history) * len(self.agent_states)
            pbm_peak = int(pbm_values.max())
            pbm_peak_round = int(pbm_values.argmax() + 1)
            pbm_final = int(pbm_values[-1])
            pbm_diffusion = round((pbm_peak - pbm_final) / max(1, pbm_peak), 3)
        else:
            pbm_peak = pbm_peak_round = pbm_final = pbm_diffusion = 0

        # --- Intervention analysis ---
        if abm_counts:
            pre_avg = np.mean(abm_counts[:max(1, len(abm_counts)//2)])
            post_avg = np.mean(abm_counts[max(1, len(abm_counts)//2):])
            change = ((post_avg - pre_avg) / pre_avg) * 100 if pre_avg != 0 else 0
        else:
            pre_avg = post_avg = change = 0

        # --- Risk analysis (standardized for all topics) ---
        financial_risk = min(1.0, 0.2 + 0.012 * abm_total_shares + 0.007 * juiciness_score)
        reputational_risk = min(1.0, 0.05 + 0.01 * abm_peak)
        trust_risk = min(1.0, 0.05 + 0.008 * juiciness_score / 10)

        # ==== 2. Text summary layout ====
        text_frame = tk.Frame(summary_win)
        text_frame.pack(padx=15, pady=10)

        text = (
            "News Context Analysis\n"
            f"Topic Category: {topic_category}\n"
            f"Context: {context_text}\n"
            f"Juiciness Score: {juiciness_score}/100\n"
            f"Topic Weight: {topic_weight:.2f}\n\n"

            "Model Comparison\n"
            f"Agent-Based Model (ABM):\n"
            f"• Peak Believers: {abm_peak} ({abm_peak/len(self.agent_states)*100:.1f}%) at round {abm_peak_round}\n"
            f"• Final State: {abm_final} ({abm_final/len(self.agent_states)*100:.1f}%)\n"
            f"• Total Shares: {abm_total_shares}\n\n"

            f"Population-Based Model (PBM):\n"
            f"• Peak Believers: {pbm_peak} ({pbm_peak/len(self.agent_states)*100:.1f}%) at round {pbm_peak_round}\n"
            f"• Final State: {pbm_final} ({pbm_final/len(self.agent_states)*100:.1f}%)\n"
            f"• Diffusion Rate: {pbm_diffusion:.3f}\n\n"

            "Intervention Analysis\n"
            f"Pre-intervention average: {pre_avg:.1f} believers/round\n"
            f"Post-intervention average: {post_avg:.1f} believers/round\n"
            f"Effect: {change:.1f}% change\n\n"

            "Risk Analysis\n"
            f"Financial Impact Risk: {financial_risk*100:.1f}%\n"
            f"Reputational Damage Risk: {reputational_risk*100:.1f}%\n"
            f"Trust Erosion Risk: {trust_risk*100:.1f}%\n\n"

            "Recommendations\n"
            "• URGENT: Immediate response required – viral spread detected\n"
            "• Critical financial loss risk – implement mitigation strategies\n"
            "• Continue monitoring intervention effectiveness\n"
        )

        label = tk.Label(
            text_frame,
            text=text,
            justify="left",
            font=("Segoe UI", 10),
            anchor="w"
        )
        label.pack()

        # ==== 3. Control buttons ====
        def add_more_rounds():
            self.max_rounds += 10
            self.extra_rounds = True
            summary_win.destroy()
            self.root.after(500, self.automate_rounds)

        def end_simulation_now():
            summary_win.destroy()
            self.reset_btn.grid()

        button_frame = tk.Frame(summary_win)
        button_frame.pack(pady=8)
        tk.Button(button_frame, text="Add 10 More Rounds", command=add_more_rounds).pack(side="left", padx=5)
        tk.Button(button_frame, text="End Simulation Now", command=end_simulation_now).pack(side="left", padx=5)

        def on_close():
            summary_win.destroy()
            self.reset_btn.grid()
        summary_win.protocol("WM_DELETE_WINDOW", on_close)

# Run the GUI if this script is executed directly
if __name__ == "__main__":
    root = tk.Tk()
    app = FakeNewsSimulatorGUI(root)

    def on_main_close():
        root.quit()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_main_close)

    root.mainloop()
    sys.exit()
