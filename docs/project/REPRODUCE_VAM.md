# Reproducing Original VAM Model with ELBO

## Overview

This guide helps you understand and run the original VAM model implementation.

## Key Differences

| Aspect | Simplified Version (PyTorch) | Original VAM (JAX/Flax) |
|--------|------------------------------|-------------------------|
| **Framework** | PyTorch | JAX/Flax |
| **Loss Function** | Custom drift_rate_loss | ELBO (Evidence Lower Bound) |
| **Training Method** | Supervised Learning | Variational Inference |
| **Optimization Target** | Drift rates only | CNN + LBA parameters jointly |
| **Cognitive Model** | None | Full LBA model |
| **RT Prediction** | Simplified | Full LBA simulation |

## How to Run Original VAM

### Step 1: Install Dependencies

```bash
cd /Users/siyu/Documents/GitHub/VAM-studying/vam
pip install -r training_requirements.txt
```

### Step 2: Train VAM Model

```bash
# Basic VAM training
python -m vam.training --project test --expt_name exp001

# Task-optimized model
python -m vam.training --model_type task_opt --n_epochs 30

# Binned RT model
python -m vam.training --model_type binned_rt --n_rt_bins 5 --rt_bin 3
```

## Understanding ELBO

### ELBO Formula

```
ELBO = E_q[log p(x|z)] + E_q[log p(z)] - E_q[log q(z)] + Jacobian
```

### Components Explained

#### 1. Reconstruction Term: `log p(x|z)`
- **What**: LBA model log-likelihood
- **Purpose**: Ensures drift rates can predict real RT and choices
- **How**: Uses LBA model to compute probability of observed data

#### 2. Prior Term: `log p(z)`
- **What**: Regularization on LBA parameters
- **Purpose**: Keeps parameters in reasonable range
- **How**: Assumes log-normal distribution for (a, c, t0)

#### 3. Entropy Term: `log q(z)`
- **What**: Uncertainty in variational posterior
- **Purpose**: Encourages exploration, prevents overfitting
- **How**: Measures spread of variational distribution

#### 4. Jacobian Term
- **What**: Density adjustment for variable transformation
- **Purpose**: Ensures probability conservation
- **How**: Adjusts for log-transformation of LBA parameters

## Key Files to Study

### 1. `vam/vam/models.py`
**Purpose**: Model architecture

**Key Classes**:
- `VAM`: Main model combining CNN + LBAVI
- `CNN`: Generates drift rates from images
- `LBAVI`: Variational inference for LBA parameters

**Important Code**:
```python
class VAM(nn.Module):
    def setup(self):
        self.get_drifts = CNN(...)  # CNN generates drift rates
        self.get_elbo = LBAVI(...)  # LBAVI computes ELBO
    
    def __call__(self, stimuli, rts, responses, key, training):
        drift_mean = self.get_drifts(stimuli, training)
        elbo = self.get_elbo(rts, responses, drift_mean, key)
        return elbo, drift_mean
```

### 2. `vam/vam/lba.py`
**Purpose**: LBA model implementation

**Key Functions**:
- `lba_logp`: Computes log-likelihood for LBA model
- `generate_vam_rts`: Simulates RT and choices using drift rates

**Important Code**:
```python
def lba_logp(t, c, v, b, A, t0, s):
    """
    Computes log-likelihood for LBA model
    
    Parameters:
        t: reaction time
        c: choice
        v: drift rates (from CNN)
        b: threshold
        A: start point range
        t0: non-decision time
        s: drift rate std
    
    Returns:
        log probability
    """
    # Implementation details...
```

### 3. `vam/vam/training.py`
**Purpose**: Training loop with ELBO

**Key Functions**:
- `vam_train_step`: Training step with ELBO optimization
- `vam_eval_step`: Evaluation step

**Important Code**:
```python
@jax.jit
def vam_train_step(state, batch, key):
    def loss_fn(params):
        elbo, _ = state.apply_fn(
            {"params": params},
            imgs, rts, choices, mc_key,
            training=True,
        )
        return -elbo  # Maximize ELBO = Minimize -ELBO
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
```

### 4. `vam/vam/config.py`
**Purpose**: Configuration management

**Key Functions**:
- `get_default_config`: Default VAM configuration
- `get_task_opt_config`: Task-optimized model config
- `get_binned_rt_config`: Binned RT model config

## ELBO Optimization Process

### Step-by-Step

1. **Forward Pass**:
   ```
   Image → CNN → Drift Rates
   ```

2. **Variational Sampling**:
   ```
   Sample LBA parameters (a, c, t0) from variational posterior
   ```

3. **ELBO Computation**:
   ```
   ELBO = LBA_log_likelihood + Prior + Entropy + Jacobian
   ```

4. **Backward Pass**:
   ```
   Update CNN weights and LBA parameters
   ```

### Why ELBO Works Better

1. **Joint Optimization**:
   - CNN learns to generate meaningful drift rates
   - LBA parameters adapt to data distribution

2. **Uncertainty Quantification**:
   - Variational inference provides uncertainty estimates
   - Helps prevent overfitting

3. **Cognitive Modeling**:
   - ELBO connects visual features to cognitive process
   - Provides interpretable parameters

## Comparison: Simplified vs Original

### Simplified Version (PyTorch)

**Pros**:
- Easy to understand
- Quick to experiment
- No JAX/Flax dependency

**Cons**:
- No cognitive modeling
- Simplified loss function
- Limited interpretability

**Best for**:
- Learning basic concepts
- Quick prototyping
- Understanding drift rates

### Original Version (JAX/Flax)

**Pros**:
- Complete cognitive modeling
- Full ELBO optimization
- Better performance
- Interpretable parameters

**Cons**:
- Steeper learning curve
- JAX/Flax dependency
- More complex code

**Best for**:
- Research
- Understanding VAM deeply
- Reproducing paper results

## Recommended Learning Path

### Phase 1: Understand Basics (1-2 days)
1. Run simplified PyTorch version
2. Understand drift rate concept
3. Visualize Flanker stimuli

### Phase 2: Study Original Code (2-3 days)
1. Read `models.py` to understand architecture
2. Study `lba.py` to understand LBA model
3. Analyze `training.py` to understand ELBO

### Phase 3: Run Experiments (3-5 days)
1. Train VAM on Flanker task
2. Analyze drift rates
3. Compare with simplified version

### Phase 4: Deep Dive (1-2 weeks)
1. Read the paper thoroughly
2. Modify model architecture
3. Experiment with different configurations

## Quick Start Commands

```bash
# Navigate to vam directory
cd /Users/siyu/Documents/GitHub/VAM-studying/vam

# Install dependencies
pip install -r training_requirements.txt

# Quick training (few epochs for testing)
python -m vam.training --project test --expt_name quick_test --n_epochs 5

# Full training
python -m vam.training --project test --expt_name full_run --n_epochs 30
```

## Expected Outputs

### Training Logs
```
Epoch [1/30]
  Train - Loss: 2.345, Accuracy: 0.45
  Val - Loss: 2.123, Accuracy: 0.52
  Learning rate: 0.001
------------------------------
```

### Model Checkpoints
```
checkpoints/
  ├── model_001.pkl
  ├── model_002.pkl
  └── model_final.pkl
```

### Metrics
- ELBO values
- Drift rate distributions
- RT prediction accuracy
- Choice prediction accuracy

## Troubleshooting

### Issue 1: JAX Installation
```bash
# For CPU only
pip install jax jaxlib

# For GPU (CUDA)
pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Issue 2: Memory Issues
```bash
# Reduce batch size
python -m vam.training --batch_size 16

# Use smaller model
python -m vam.training --model_type task_opt
```

### Issue 3: Data Loading
```bash
# Check data files exist
ls vam/*.zip
ls vam/*.csv
```

## Next Steps

1. **Run the original VAM code** following the commands above
2. **Compare results** between simplified and original versions
3. **Analyze drift rates** to understand model behavior
4. **Read the paper** for deeper understanding

## Questions to Explore

1. How do drift rates differ between congruent and incongruent trials?
2. How does ELBO optimization affect drift rate learning?
3. What is the relationship between drift rates and RT?
4. How do LBA parameters (a, c, t0) change during training?

---

**Created**: 2026-03-26  
**Purpose**: Guide for reproducing original VAM model with ELBO
