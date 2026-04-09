import torch
import torch.nn as nn
import torch.nn.functional as F

from vgg_wongwang_lim import DiffDecisionMultiClass, VGGFeatureExtractor


class AccumulatorRaceDecisionV2(nn.Module):
    def __init__(self, n_classes=4, dt=10, time_steps=120, threshold=0.5, noise_std=0.02):
        super().__init__()
        self.n_classes = n_classes
        self.dt = dt
        self.time_steps = time_steps
        self.register_buffer('threshold', torch.tensor(float(threshold), dtype=torch.float32))
        self.input_scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.leak = nn.Parameter(torch.full((n_classes,), 0.08, dtype=torch.float32))
        self.self_excitation = nn.Parameter(torch.full((n_classes,), 0.12, dtype=torch.float32))
        self.inhibition = nn.Parameter(torch.tensor(0.08, dtype=torch.float32))
        self.noise_std = nn.Parameter(torch.tensor(float(noise_std), dtype=torch.float32))
        self.evidence_proj = nn.Linear(1, 1)

    def _momentary_evidence(self, logits):
        x = F.relu(logits * self.input_scale).unsqueeze(-1)
        return self.evidence_proj(x).squeeze(-1)

    def rollout(self, logits):
        batch, n_classes = logits.shape
        device = logits.device
        acc = torch.zeros(batch, n_classes, device=device)
        traj = torch.zeros(batch, self.time_steps, n_classes, device=device)
        dsdt_traj = torch.zeros(batch, self.time_steps, n_classes, device=device)
        evidence = self._momentary_evidence(logits)
        leak = F.softplus(self.leak)
        self_exc = F.softplus(self.self_excitation)
        inhib = F.softplus(self.inhibition)
        noise_std = F.softplus(self.noise_std)

        for t in range(self.time_steps):
            total_other = acc.sum(dim=1, keepdim=True) - acc
            drive = evidence + self_exc * acc - inhib * total_other - leak * acc
            noise = torch.randn(batch, n_classes, device=device) * noise_std
            dsdt = F.softplus(drive + noise) - acc
            acc = torch.clamp(acc + 0.2 * dsdt, min=0.0)
            traj[:, t, :] = acc
            dsdt_traj[:, t, :] = dsdt

        threshold = torch.as_tensor(self.threshold, device=device, dtype=torch.float32)
        decision_times = torch.as_tensor(DiffDecisionMultiClass.apply(traj - threshold, dsdt_traj, self.dt, self.time_steps))
        return decision_times / 1000.0, traj, threshold

    def forward(self, logits):
        decision_times, _, _ = self.rollout(logits)
        return decision_times

    def inference(self, logits):
        return self.rollout(logits)


class VGGAccumulatorRNNLIMV2(nn.Module):
    def __init__(self, pretrained=True, freeze_features=False, n_classes=4, dropout_rate=0.5, dt=10, time_steps=120, threshold=0.5, noise_std=0.02):
        super().__init__()
        self.feature_extractor = VGGFeatureExtractor(pretrained=pretrained, freeze_features=freeze_features, n_classes=n_classes, dropout_rate=dropout_rate)
        self.decision = AccumulatorRaceDecisionV2(n_classes=n_classes, dt=dt, time_steps=time_steps, threshold=threshold, noise_std=noise_std)

    def forward(self, x, return_logits=False):
        logits = self.feature_extractor(x)
        decision_times, traj, threshold = self.decision.inference(logits)
        pred_choice = decision_times.argmin(dim=1)
        final_dt = decision_times.min(dim=1)[0]
        if return_logits:
            return logits, decision_times, final_dt, traj, threshold, pred_choice
        return logits, final_dt, pred_choice

    def get_logits(self, x):
        return self.feature_extractor(x)

    def get_decision_times(self, logits):
        return self.decision(logits)
