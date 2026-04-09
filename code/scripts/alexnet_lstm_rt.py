"""
AlexNet + LSTM Model with Evidence Accumulation Noise

Architecture:
- AlexNet (pretrained on ImageNet, adapted for MNIST)
- LSTM for temporal processing
- Evidence accumulation with internal noise
- SAT-conditioned thresholds

Key features:
- Input noise based on difficulty (Easy: 0.25, Difficult: 0.4)
- Evidence noise (internal noise for RT variability)
- Separate thresholds for Speed/Accuracy conditions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiffDecision(torch.autograd.Function):
    """Differentiable decision time computation."""
    
    @staticmethod
    def forward(ctx, trajectory, dsdt_trajectory):
        mask = trajectory > 0
        decision_time = mask.float().argmax(dim=1).float()
        decision_time[mask.sum(dim=1) == 0] = float(trajectory.shape[1] - 1)
        ctx.save_for_backward(dsdt_trajectory, decision_time, trajectory)
        return decision_time
    
    @staticmethod
    def backward(ctx, grad_output):
        dsdt_trajectory, decision_times, trajectory = ctx.saved_tensors
        device = dsdt_trajectory.device
        mask = trajectory > 0
        idx1 = (mask.sum(dim=1) == 0)
        idx2 = dsdt_trajectory[torch.arange(dsdt_trajectory.size(0), device=device), decision_times.long()] < 0
        idx = torch.logical_and(idx1, idx2)
        grads = torch.zeros_like(dsdt_trajectory)
        batch_indices = torch.arange(decision_times.size(0), device=device)
        grads[batch_indices, decision_times.long()] = -1.0 / (dsdt_trajectory[batch_indices, decision_times.long()] + 1e-6)
        grads[batch_indices[idx], decision_times[idx].long()] = 1e-6
        grads = grads * grad_output.unsqueeze(1)
        return grads, None


class AlexNetFeatureExtractor(nn.Module):
    """AlexNet-based feature extractor for MNIST."""
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNetLSTM_RT(nn.Module):
    """
    AlexNet + LSTM model for RT prediction.
    
    Features:
    - AlexNet feature extractor
    - LSTM for temporal processing
    - Evidence accumulation with internal noise
    - SAT-conditioned thresholds
    """
    
    def __init__(self, 
                 num_classes=8,
                 time_steps=40,
                 lstm_hidden=256,
                 sigma=2.0,
                 evidence_noise_std=0.5,
                 evidence_mask_p=0.4,
                 evidence_dropout_rescale=True,
                 use_sat=True):
        super().__init__()
        
        self.time_steps = time_steps
        self.sigma = sigma
        self.evidence_noise_std = evidence_noise_std
        self.evidence_mask_p = evidence_mask_p
        self.evidence_dropout_rescale = evidence_dropout_rescale
        self.use_sat = use_sat
        
        self.feature_extractor = AlexNetFeatureExtractor()
        
        self.lstm = nn.LSTM(
            input_size=4096,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=False
        )
        
        self.fc = nn.Linear(lstm_hidden, num_classes)
        
        self.evidence = nn.Linear(lstm_hidden, 1)
        
        if use_sat:
            self.threshold_speed = nn.Parameter(torch.tensor(3.0))
            self.threshold_accuracy = nn.Parameter(torch.tensor(5.0))
        else:
            self.threshold = nn.Parameter(torch.tensor(4.0))
        
        self.sat_mapping = {
            'speed focus': 0,
            'accuracy focus': 1,
            'speed': 0,
            'accuracy': 1,
            'unknown': 0
        }
    
    def _get_threshold_batch(self, sat_conditions, batch_size, device):
        thresholds = []
        for sat in sat_conditions:
            if isinstance(sat, str):
                sat_lower = sat.lower()
                if sat_lower in ['speed focus', 'speed']:
                    thresholds.append(self.threshold_speed)
                elif sat_lower in ['accuracy focus', 'accuracy']:
                    thresholds.append(self.threshold_accuracy)
                else:
                    thresholds.append(self.threshold_speed)
            else:
                thresholds.append(self.threshold_speed)
        return torch.stack(thresholds)
    
    def _add_evidence_noise(self, s_traj):
        if self.evidence_noise_std > 0:
            noise = torch.randn_like(s_traj) * self.evidence_noise_std
            s_traj = s_traj + noise
        
        if self.evidence_mask_p > 0:
            mask = torch.bernoulli(torch.ones_like(s_traj) * (1 - self.evidence_mask_p))
            s_traj = s_traj * mask
            if self.evidence_dropout_rescale:
                s_traj = s_traj / (1 - self.evidence_mask_p)
        
        return s_traj
    
    def forward(self, x, sat_condition=None):
        device = x.device
        B = x.shape[0]
        
        features = self.feature_extractor(x)
        
        features_seq = features.unsqueeze(0).repeat(self.time_steps, 1, 1)
        
        lstm_out, _ = self.lstm(features_seq)
        
        lstm_out = lstm_out.permute(1, 0, 2)
        
        logit_trajectory = self.fc(lstm_out)
        
        s_traj = self.evidence(lstm_out).squeeze(-1)
        
        s_traj = self._add_evidence_noise(s_traj)
        
        s_accumulated = torch.cumsum(s_traj, dim=1)
        dsdt_trajectory = torch.diff(s_accumulated, dim=1)
        dsdt_trajectory = torch.cat((dsdt_trajectory[:, 0].unsqueeze(1), dsdt_trajectory), dim=1)
        
        if self.use_sat and sat_condition is not None:
            threshold_batch = self._get_threshold_batch(sat_condition, B, device)
        elif self.use_sat:
            threshold_batch = self.threshold_speed.expand(B)
        else:
            threshold_batch = self.threshold.expand(B)
        
        decision_time = DiffDecision.apply(
            s_accumulated - threshold_batch.unsqueeze(1),
            dsdt_trajectory
        )
        
        soft_index = torch.exp(
            -0.5 * (decision_time.unsqueeze(1) - torch.arange(self.time_steps, device=device)) ** 2 / self.sigma ** 2
        )
        soft_index = soft_index / soft_index.sum(dim=-1, keepdim=True)
        decision_logits = (logit_trajectory * soft_index.unsqueeze(-1)).sum(dim=1)
        
        probs = F.softmax(decision_logits, dim=-1)
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[:, 0] - sorted_probs[:, 1]
        
        return decision_logits, (decision_time + 1) / self.time_steps, confidence
    
    def get_threshold_values(self):
        if self.use_sat:
            return {
                'threshold_speed': self.threshold_speed.item(),
                'threshold_accuracy': self.threshold_accuracy.item()
            }
        else:
            return {'threshold': self.threshold.item()}


class AlexNetLSTM_RT_Simple(nn.Module):
    """
    Simplified version without LSTM temporal dynamics.
    Uses repeated feature extraction for each time step.
    """
    
    def __init__(self,
                 num_classes=8,
                 time_steps=40,
                 sigma=2.0,
                 evidence_noise_std=0.5,
                 evidence_mask_p=0.4,
                 evidence_dropout_rescale=True,
                 use_sat=True):
        super().__init__()
        
        self.time_steps = time_steps
        self.sigma = sigma
        self.evidence_noise_std = evidence_noise_std
        self.evidence_mask_p = evidence_mask_p
        self.evidence_dropout_rescale = evidence_dropout_rescale
        self.use_sat = use_sat
        
        self.feature_extractor = AlexNetFeatureExtractor()
        
        self.fc = nn.Linear(4096, num_classes)
        
        self.evidence = nn.Linear(4096, 1)
        
        if use_sat:
            self.threshold_speed = nn.Parameter(torch.tensor(3.0))
            self.threshold_accuracy = nn.Parameter(torch.tensor(5.0))
        else:
            self.threshold = nn.Parameter(torch.tensor(4.0))
        
        self.sat_mapping = {
            'speed focus': 0,
            'accuracy focus': 1,
            'speed': 0,
            'accuracy': 1,
            'unknown': 0
        }
    
    def _get_threshold_batch(self, sat_conditions, batch_size, device):
        thresholds = []
        for sat in sat_conditions:
            if isinstance(sat, str):
                sat_lower = sat.lower()
                if sat_lower in ['speed focus', 'speed']:
                    thresholds.append(self.threshold_speed)
                elif sat_lower in ['accuracy focus', 'accuracy']:
                    thresholds.append(self.threshold_accuracy)
                else:
                    thresholds.append(self.threshold_speed)
            else:
                thresholds.append(self.threshold_speed)
        return torch.stack(thresholds)
    
    def _add_evidence_noise(self, s_traj):
        if self.evidence_noise_std > 0:
            noise = torch.randn_like(s_traj) * self.evidence_noise_std
            s_traj = s_traj + noise
        
        if self.evidence_mask_p > 0:
            mask = torch.bernoulli(torch.ones_like(s_traj) * (1 - self.evidence_mask_p))
            s_traj = s_traj * mask
            if self.evidence_dropout_rescale:
                s_traj = s_traj / (1 - self.evidence_mask_p)
        
        return s_traj
    
    def forward(self, x, sat_condition=None):
        device = x.device
        B = x.shape[0]
        
        features = self.feature_extractor(x)
        
        features_seq = features.unsqueeze(1).repeat(1, self.time_steps, 1)
        
        logit_trajectory = self.fc(features_seq)
        
        s_traj = self.evidence(features_seq).squeeze(-1)
        
        s_traj = self._add_evidence_noise(s_traj)
        
        s_accumulated = torch.cumsum(s_traj, dim=1)
        dsdt_trajectory = torch.diff(s_accumulated, dim=1)
        dsdt_trajectory = torch.cat((dsdt_trajectory[:, 0].unsqueeze(1), dsdt_trajectory), dim=1)
        
        if self.use_sat and sat_condition is not None:
            threshold_batch = self._get_threshold_batch(sat_condition, B, device)
        elif self.use_sat:
            threshold_batch = self.threshold_speed.expand(B)
        else:
            threshold_batch = self.threshold.expand(B)
        
        decision_time = DiffDecision.apply(
            s_accumulated - threshold_batch.unsqueeze(1),
            dsdt_trajectory
        )
        
        soft_index = torch.exp(
            -0.5 * (decision_time.unsqueeze(1) - torch.arange(self.time_steps, device=device)) ** 2 / self.sigma ** 2
        )
        soft_index = soft_index / soft_index.sum(dim=-1, keepdim=True)
        decision_logits = (logit_trajectory * soft_index.unsqueeze(-1)).sum(dim=1)
        
        probs = F.softmax(decision_logits, dim=-1)
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[:, 0] - sorted_probs[:, 1]
        
        return decision_logits, (decision_time + 1) / self.time_steps, confidence
    
    def get_threshold_values(self):
        if self.use_sat:
            return {
                'threshold_speed': self.threshold_speed.item(),
                'threshold_accuracy': self.threshold_accuracy.item()
            }
        else:
            return {'threshold': self.threshold.item()}
