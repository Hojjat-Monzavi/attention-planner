 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# TASK DEFINITION
# ============================================================================

class Task:
    def __init__(self, priority, min_duration, max_duration, due_time):
        self.priority = priority
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.due_time = due_time

    def to_vector(self, current_time=0.0):
        avg_dur = (self.min_duration + self.max_duration) / 2
        # Calculate standard deviation (using uniform distribution formula)
        duration_sd = (self.max_duration - self.min_duration) / np.sqrt(12)
        slack = max(0, self.due_time - current_time - avg_dur)
        
        return np.array([
            self.priority / 10.0,
            avg_dur / 4.0,
            self.due_time / 30.0,
            slack / 20.0,
            current_time / 20.0,
            (self.priority * avg_dur) / 40.0,
            duration_sd / 2.0,  # Normalized SD
            (duration_sd / max(avg_dur, 0.1))  # Coefficient of variation
        ], dtype=np.float32)

def generate_tasks(num_tasks=10):
    tasks = []
    # Dynamic range of tasks
    count = np.random.randint(num_tasks-2, num_tasks+2) if num_tasks > 2 else num_tasks

    for i in range(count):
        priority = np.random.uniform(1, 10)
        min_dur = np.random.uniform(0.5, 3.0)
        max_dur = min_dur + np.random.uniform(0, 2.0)
        # Randomize due times to create conflict
        due_time = np.random.uniform(3 + i, count * 1.5)
        tasks.append(Task(priority, min_dur, max_dur, due_time))

    rest_factor = 1
    return tasks, rest_factor

# ============================================================================
# MULTI-HEAD SELF ATTENTION
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape to [batch, num_heads, seq_len, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, heads, seq, seq]
        
        # Apply mask if provided (mask unavailable tasks)
        if mask is not None:
            # mask shape: [batch, seq_len] -> [batch, 1, 1, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~mask, -1e9)
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]
        
        # Reshape back to [batch, seq, hidden_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output

# ============================================================================
# PROGRESSIVE ATTENTION PLANNER WITH MULTI-HEAD ATTENTION
# ============================================================================

class ProgressiveAttentionPlanner(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout

        self.ln_in = nn.LayerNorm(input_dim)
        self.ln_task = nn.LayerNorm(hidden_dim)
        self.ln_context = nn.LayerNorm(hidden_dim)

        # Task encoder with 2 layers
        self.task_fc1 = nn.Linear(input_dim, hidden_dim)
        self.task_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.task_dropout = nn.Dropout(dropout)

        # Multi-head self-attention for context
        self.self_attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.context_fc = nn.Linear(hidden_dim, hidden_dim)
        self.context_dropout = nn.Dropout(dropout)

        # Attention with 3 layers
        self.attn_fc1 = nn.Linear(hidden_dim * 2 + 1, hidden_dim)
        self.attn_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.attn_fc3 = nn.Linear(hidden_dim, 1)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, task_features, rest_factor, mask):
        x = self.ln_in(task_features)

        # Task encoder
        task_embeds = torch.relu(self.task_fc1(x))
        task_embeds = self.task_dropout(task_embeds)
        task_embeds = torch.relu(self.task_fc2(task_embeds))
        task_embeds = self.task_dropout(task_embeds)
        task_embeds = self.ln_task(task_embeds + x.mean(dim=-1, keepdim=True))

        # Multi-head self-attention context encoder
        task_embeds_batch = task_embeds.unsqueeze(0)  # [1, seq_len, hidden_dim]
        mask_batch = mask.unsqueeze(0)  # [1, seq_len]
        
        # Apply self-attention
        attended = self.self_attention(task_embeds_batch, mask_batch)  # [1, seq_len, hidden_dim]
        attended = attended.squeeze(0)  # [seq_len, hidden_dim]
        
        # Process attended features
        context = torch.relu(self.context_fc(attended))
        context = self.context_dropout(context)
        context = self.ln_context(context + attended)

        # Attention scoring
        rest_tensor = torch.full((len(task_embeds), 1),
                                 float(rest_factor),
                                 dtype=task_embeds.dtype,
                                 device=task_embeds.device)
        combined = torch.cat([task_embeds, context, rest_tensor], dim=-1)

        logits = torch.relu(self.attn_fc1(combined))
        logits = self.attn_dropout(logits)
        logits = torch.relu(self.attn_fc2(logits))
        logits = self.attn_dropout(logits)
        logits = self.attn_fc3(logits).squeeze(-1)

        logits = logits.masked_fill(~mask, -1e9)
        return logits

    def expand_width(self, new_hidden_dim, init_strategy='zeros'):
        """
        Expand the hidden dimension while preserving existing functionality.

        Args:
            new_hidden_dim: New hidden dimension (must be > current hidden_dim)
            init_strategy: 'zeros' (safe), 'small_random', or 'orthogonal'
        """
        if new_hidden_dim <= self.hidden_dim:
            raise ValueError(f"New hidden_dim ({new_hidden_dim}) must be > current ({self.hidden_dim})")
        
        # Ensure new hidden_dim is divisible by num_heads
        if new_hidden_dim % self.num_heads != 0:
            new_hidden_dim = ((new_hidden_dim // self.num_heads) + 1) * self.num_heads
            print(f"⚠️  Adjusted new_hidden_dim to {new_hidden_dim} (must be divisible by num_heads={self.num_heads})")

        old_dim = self.hidden_dim
        expansion = new_hidden_dim - old_dim

        print(f"\n🔄 Expanding model from {old_dim} to {new_hidden_dim} dimensions...")
        print(f"   Adding {expansion} new dimensions to each layer")

        def expand_linear_layer(old_layer, new_in, new_out, is_output=False):
            """Expand a linear layer while preserving output for old inputs."""
            new_layer = nn.Linear(new_in, new_out, bias=old_layer.bias is not None).to(device)

            with torch.no_grad():
                old_out, old_in = old_layer.weight.shape

                if is_output:
                    # For output layers: preserve exact output by zero-padding inputs
                    new_layer.weight[:old_out, :old_in] = old_layer.weight
                    new_layer.weight[:old_out, old_in:] = 0
                    if old_layer.bias is not None:
                        new_layer.bias[:old_out] = old_layer.bias
                        new_layer.bias[old_out:] = 0
                else:
                    # For input/hidden layers: preserve old pathway
                    new_layer.weight[:old_out, :old_in] = old_layer.weight

                    # Initialize new neurons (new output dimensions)
                    if new_out > old_out:
                        if init_strategy == 'zeros':
                            new_layer.weight[old_out:, :] = 0
                        elif init_strategy == 'small_random':
                            scale = 0.01
                            new_layer.weight[old_out:, :] = torch.randn(new_out - old_out, new_in, device=device) * scale
                        elif init_strategy == 'orthogonal':
                            nn.init.orthogonal_(new_layer.weight[old_out:, :], gain=0.01)

                    # Initialize connections from new input dimensions to old outputs
                    if new_in > old_in:
                        if init_strategy == 'zeros':
                            new_layer.weight[:old_out, old_in:] = 0
                        elif init_strategy == 'small_random':
                            scale = 0.01
                            new_layer.weight[:old_out, old_in:] = torch.randn(old_out, new_in - old_in, device=device) * scale
                        elif init_strategy == 'orthogonal':
                            if old_out > 1 and (new_in - old_in) > 1:
                                nn.init.orthogonal_(new_layer.weight[:old_out, old_in:], gain=0.01)
                            else:
                                new_layer.weight[:old_out, old_in:] = torch.randn(old_out, new_in - old_in, device=device) * 0.01

                    if old_layer.bias is not None:
                        new_layer.bias[:old_out] = old_layer.bias
                        if new_out > old_out:
                            new_layer.bias[old_out:] = 0

            return new_layer

        def expand_layernorm(old_ln, new_dim):
            """Expand LayerNorm."""
            new_ln = nn.LayerNorm(new_dim).to(device)
            with torch.no_grad():
                old_dim = old_ln.normalized_shape[0]
                new_ln.weight[:old_dim] = old_ln.weight
                new_ln.weight[old_dim:] = 1.0
                new_ln.bias[:old_dim] = old_ln.bias
                new_ln.bias[old_dim:] = 0.0
            return new_ln

        # Expand LayerNorms
        self.ln_task = expand_layernorm(self.ln_task, new_hidden_dim)
        self.ln_context = expand_layernorm(self.ln_context, new_hidden_dim)

        # Expand task encoder
        self.task_fc1 = expand_linear_layer(self.task_fc1, self.input_dim, new_hidden_dim)
        self.task_fc2 = expand_linear_layer(self.task_fc2, new_hidden_dim, new_hidden_dim)

        # Expand multi-head self-attention
        old_attn = self.self_attention
        new_attn = MultiHeadSelfAttention(new_hidden_dim, self.num_heads, self.dropout_rate).to(device)
        
        with torch.no_grad():
            # Expand Q, K, V projections
            new_attn.q_proj = expand_linear_layer(old_attn.q_proj, old_dim, new_hidden_dim)
            new_attn.k_proj = expand_linear_layer(old_attn.k_proj, old_dim, new_hidden_dim)
            new_attn.v_proj = expand_linear_layer(old_attn.v_proj, old_dim, new_hidden_dim)
            new_attn.out_proj = expand_linear_layer(old_attn.out_proj, new_hidden_dim, new_hidden_dim)
        
        self.self_attention = new_attn
        
        # Expand context processing
        self.context_fc = expand_linear_layer(self.context_fc, new_hidden_dim, new_hidden_dim)

        # Expand attention layers
        self.attn_fc1 = expand_linear_layer(self.attn_fc1, new_hidden_dim * 2 + 1, new_hidden_dim)
        self.attn_fc2 = expand_linear_layer(self.attn_fc2, new_hidden_dim, new_hidden_dim)
        self.attn_fc3 = expand_linear_layer(self.attn_fc3, new_hidden_dim, 1, is_output=True)

        self.hidden_dim = new_hidden_dim
        print(f"✅ Expansion complete! New parameter count: {sum(p.numel() for p in self.parameters()):,}")

# ============================================================================
# COST FUNCTION
# ============================================================================

def compute_cost(plan: List[Task], rest_factor: float, n_sim=400):
    n_tasks = len(plan)
    min_durs = np.array([t.min_duration for t in plan])
    max_durs = np.array([t.max_duration for t in plan])
    priorities = np.array([t.priority for t in plan])
    due_times = np.array([t.due_time for t in plan])

    mu = (min_durs + max_durs) / 2
    sigma = (max_durs - min_durs) / 6
    durations = np.random.normal(loc=mu, scale=sigma, size=(n_sim, n_tasks))
    durations = np.maximum(0, durations)
    durations *= rest_factor
    completion_times = np.cumsum(durations, axis=1)
    overdue = completion_times > due_times
    cost = np.sum(priorities * overdue.mean(axis=0))/n_tasks
    return cost

# ============================================================================
# REINFORCE TRAINING
# ============================================================================

def generate_plan(model, tasks, rest_factor):
    remaining = list(range(len(tasks)))
    plan_indices = []
    log_probs = []
    current_time = 0.0

    while remaining:
        features = torch.tensor(
            np.array([tasks[i].to_vector(current_time) for i in remaining]),
            dtype=torch.float32, device=device
        )
        mask = torch.ones(len(remaining), dtype=torch.bool, device=device)

        logits = model(features, rest_factor, mask)
        probs = torch.softmax(logits, dim=0)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        selected_idx = remaining[action.item()]
        plan_indices.append(selected_idx)
        log_probs.append(log_prob)

        task = tasks[selected_idx]
        avg_dur = (task.min_duration + task.max_duration) / 2
        current_time += avg_dur * rest_factor
        remaining.remove(selected_idx)

    plan = [tasks[i] for i in plan_indices]
    return plan, log_probs

def get_baseline_cost(tasks, rest_factor):
    priority_plan = sorted(tasks, key=lambda t: t.priority, reverse=True)
    return compute_cost(priority_plan, rest_factor, n_sim=100)

def train(model, optimizer, scheduler, num_episodes=3000, num_tasks=10):
    costs, baselines = [], []
    best_cost = float('inf')

    for episode in tqdm(range(num_episodes), desc="Training"):
        tasks, rest_factor = generate_tasks(num_tasks)
        plan, log_probs = generate_plan(model, tasks, rest_factor)
        cost = compute_cost(plan, rest_factor, n_sim=200)
        baseline = get_baseline_cost(tasks, rest_factor)
        advantage = baseline - cost

        loss = -torch.stack(log_probs).sum() * advantage
        if episode < num_episodes // 2:
            loss = loss + 0.00005 * sum(p.pow(2).sum() for p in model.parameters())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        costs.append(cost)
        baselines.append(baseline)
        best_cost = min(best_cost, cost)

        if (episode + 1) % 100 == 0:
            rc, rb = np.mean(costs[-100:]), np.mean(baselines[-100:])
            imp = ((rb - rc) / rb) * 100
            print(f"Ep {episode+1}: Cost={rc:.2f}, Base={rb:.2f}, Δ={imp:+.1f}%, Best={best_cost:.2f}")

    return costs, baselines

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, num_eval=500, num_tasks=10):
    model.eval()
    agent_costs, random_costs, priority_costs, due_costs, optimal_costs = [], [], [], [], []

    with torch.no_grad():
        for _ in tqdm(range(num_eval), desc="Evaluating"):
            tasks, rest_factor = generate_tasks(num_tasks)
            plan, _ = generate_plan(model, tasks, rest_factor)
            agent_costs.append(compute_cost(plan, rest_factor, n_sim=500))

            rand_plan = tasks.copy()
            np.random.shuffle(rand_plan)
            random_costs.append(compute_cost(rand_plan, rest_factor, n_sim=500))
            priority_plan = sorted(tasks, key=lambda t: t.priority, reverse=True)
            priority_costs.append(compute_cost(priority_plan, rest_factor, n_sim=500))
            due_plan = sorted(tasks, key=lambda t: t.due_time)
            due_costs.append(compute_cost(due_plan, rest_factor, n_sim=500))
            optimal_plan = sorted(tasks, key=lambda t: t.priority /
                                 ((t.min_duration + t.max_duration) / 2), reverse=True)
            optimal_costs.append(compute_cost(optimal_plan, rest_factor, n_sim=500))

    model.train()
    imp = lambda base, ag: ((base - ag) / base) * 100
    return {
        'agent': np.mean(agent_costs),
        'agent_std': np.std(agent_costs),
        'random': np.mean(random_costs),
        'priority': np.mean(priority_costs),
        'due_time': np.mean(due_costs),
        'optimal': np.mean(optimal_costs),
        'vs_random': imp(np.mean(random_costs), np.mean(agent_costs)),
        'vs_priority': imp(np.mean(priority_costs), np.mean(agent_costs)),
        'vs_due': imp(np.mean(due_costs), np.mean(agent_costs)),
        'vs_optimal': imp(np.mean(optimal_costs), np.mean(agent_costs)),
    }

# ============================================================================
# MAIN WITH PROGRESSIVE TRAINING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("PROGRESSIVE ATTENTION PLANNER — Multi-Head Self-Attention")
    print("="*70)

    # Stage 1: Train small model
    print("\n📍 STAGE 1: Training with hidden_dim=256")
    print("-" * 70)
    model = ProgressiveAttentionPlanner(input_dim=8, hidden_dim=256, num_heads=4, dropout=0.2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    costs_stage1, baselines_stage1 = train(model, optimizer, scheduler, num_episodes=5000, num_tasks=30)

    # Evaluate before expansion
    print("\n📊 Evaluation before expansion:")
    results_before = evaluate(model, num_eval=200, num_tasks=10)
    print(f"Agent cost: {results_before['agent']:.2f}")

    # Stage 2: Expand and continue training
    print("\n📍 STAGE 2: Expanding to hidden_dim=1024 and continuing training")
    print("-" * 70)
    model.expand_width(1024, init_strategy='orthogonal')

    # Create new optimizer for expanded model with lower learning rate
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2)

    costs_stage2, baselines_stage2 = train(model, optimizer, scheduler, num_episodes=3000, num_tasks=50)

    # Combine costs for plotting
    all_costs = costs_stage1 + costs_stage2
    all_baselines = baselines_stage1 + baselines_stage2

    # Plot results
    plt.figure(figsize=(14, 5))
    window = 50

    plt.subplot(1, 2, 1)
    if len(all_costs) > window:
        agent_ma = np.convolve(all_costs, np.ones(window)/window, mode='valid')
        base_ma = np.convolve(all_baselines, np.ones(window)/window, mode='valid')
        plt.plot(agent_ma, label='Agent', linewidth=2)
        plt.plot(base_ma, '--', label='Priority Baseline', linewidth=2)
        plt.axvline(len(costs_stage1), color='red', linestyle=':', linewidth=2, label='Expansion Point')
        plt.xlabel('Episode'); plt.ylabel('Cost')
        plt.title('Training Progress with Progressive Growth')
        plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    improvements = [((np.mean(all_baselines[i-100:i]) - np.mean(all_costs[i-100:i])) /
                     np.mean(all_baselines[i-100:i]) * 100)
                    for i in range(100, len(all_costs))]
    if improvements:
        plt.plot(improvements, linewidth=2)
        plt.axhline(0, color='r', linestyle='--', alpha=0.4)
        plt.axvline(len(costs_stage1)-100, color='red', linestyle=':', linewidth=2, label='Expansion')
        plt.xlabel('Episode'); plt.ylabel('Improvement (%)')
        plt.title('Performance vs Baseline'); plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig('progressive_training_sd.png', dpi=150)
    plt.show()

    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION (after expansion)")
    print("="*70)

    results = evaluate(model, num_eval=500, num_tasks=10)

    print(f"\n{'Method':<25} {'Cost':<12} {'vs Agent'}")
    print("-" * 55)
    print(f"{'Agent':<25} {results['agent']:.2f} ± {results['agent_std']:.2f}  baseline")
    print(f"{'Random':<25} {results['random']:.2f}       {results['vs_random']:+.1f}%")
    print(f"{'Priority Sort':<25} {results['priority']:.2f}       {results['vs_priority']:+.1f}%")
    print(f"{'Due Time Sort':<25} {results['due_time']:.2f}       {results['vs_due']:+.1f}%")
    print(f"{'Weighted SPT':<25} {results['optimal']:.2f}       {results['vs_optimal']:+.1f}%")

    print(f"\n📈 Improvement from expansion: {((results_before['agent'] - results['agent']) / results_before['agent'] * 100):+.2f}%")
    print(f"🎯 Enhanced with Multi-Head Self-Attention & duration uncertainty")

    torch.save(model.state_dict(), 'progressive_planner_expanded_sd.pth')
    print("\n💾 Model saved as progressive_planner_expanded_sd.pth")