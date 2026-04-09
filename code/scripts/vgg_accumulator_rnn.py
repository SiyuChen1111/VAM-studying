import torch
import torch.nn as nn
import torch.nn.functional as F

from vgg_wongwang_lim import DiffDecisionMultiClass, VGGFeatureExtractor


class AccumulatorRNNDecision(nn.Module):
    def __init__(self, n_classes=4, hidden_dim=8, dt=10, time_steps=120, threshold=0.5, noise_std=0.02):
        super().__init__()
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.time_steps = time_steps
        self.register_buffer('threshold', torch.tensor(float(threshold), dtype=torch.float32))
        self.noise_std = nn.Parameter(torch.tensor(float(noise_std), dtype=torch.float32))
        self.input_scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.input_proj = nn.Linear(1, hidden_dim)
        self.self_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.comp_proj = nn.Linear(1, hidden_dim, bias=False)
        self.evidence_head = nn.Linear(hidden_dim, 1)
        self.class_bias = nn.Parameter(torch.zeros(n_classes, hidden_dim, dtype=torch.float32))
        comp = torch.full((n_classes, n_classes), -0.05, dtype=torch.float32)
        comp[range(n_classes), range(n_classes)] = 0.25
        self.competition = nn.Parameter(comp)

    def _input_embed(self, logits):
        x = F.relu(logits * self.input_scale).unsqueeze(-1)
        return self.input_proj(x) + self.class_bias.unsqueeze(0)

    def rollout(self, logits):
        batch = logits.shape[0]
        device = logits.device
        state = torch.zeros(batch, self.n_classes, self.hidden_dim, device=device)
        evidence_traj = torch.zeros(batch, self.time_steps, self.n_classes, device=device)
        dsdt_traj = torch.zeros(batch, self.time_steps, self.n_classes, device=device)
        inp = self._input_embed(logits)
        for t in range(self.time_steps):
            evidence = self.evidence_head(state).squeeze(-1)
            comp = torch.matmul(evidence, self.competition).unsqueeze(-1)
            noise = torch.randn(batch, self.n_classes, self.hidden_dim, device=device) * self.noise_std
            candidate = torch.tanh(self.self_proj(state) + inp + self.comp_proj(comp) + noise)
            dsdt = candidate - state
            state = state + 0.2 * dsdt
            new_evidence = self.evidence_head(state).squeeze(-1)
            evidence_traj[:, t, :] = new_evidence
            dsdt_traj[:, t, :] = new_evidence - evidence
        threshold = torch.as_tensor(self.threshold, device=device, dtype=torch.float32)
        decision_times = torch.as_tensor(DiffDecisionMultiClass.apply(evidence_traj - threshold, dsdt_traj, self.dt, self.time_steps))
        return decision_times / 1000.0, evidence_traj, threshold

    def forward(self, logits):
        decision_times, _, _ = self.rollout(logits)
        return decision_times

    def inference(self, logits):
        return self.rollout(logits)


class VGGAccumulatorRNNLIM(nn.Module):
    def __init__(self, pretrained=True, freeze_features=False, n_classes=4, dropout_rate=0.5, hidden_dim=8, dt=10, time_steps=120, threshold=0.5, noise_std=0.02):
        super().__init__()
        self.feature_extractor = VGGFeatureExtractor(pretrained=pretrained, freeze_features=freeze_features, n_classes=n_classes, dropout_rate=dropout_rate)
        self.decision = AccumulatorRNNDecision(n_classes=n_classes, hidden_dim=hidden_dim, dt=dt, time_steps=time_steps, threshold=threshold, noise_std=noise_std)

    def forward(self, x, return_logits=False):
        logits = self.feature_extractor(x)
        decision_times, evidence_traj, threshold = self.decision.inference(logits)
        final_dt, pred_choice = decision_times.min(dim=1)
        if return_logits:
            return logits, decision_times, final_dt, evidence_traj, threshold, pred_choice
        return logits, final_dt, pred_choice

    def get_logits(self, x):
        return self.feature_extractor(x)

    def get_decision_times(self, logits):
        return self.decision(logits)
