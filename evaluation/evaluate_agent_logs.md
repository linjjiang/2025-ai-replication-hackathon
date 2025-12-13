# Audit Evaluation Log

## Audit Session: 2025-12-12
**Auditor**: Senior Scientific Auditor  
**Target Paper**: Piwek et al. (2023) - "A recurrent neural network model of prefrontal brain activity during a working memory task"  
**Target Replication**: AI Agent-generated code in `src/`, `scripts/`

---

## Step 1: Document Ingestion & Initial Review

### 1.1 Files Examined
- **Original Paper (PDF)**: `journal.pcbi.1011555.pdf`
  - Focus: Methods (p. 23-27), Eq. 2.1 (Loss), Eq. 1.1 (Input)
  - Key figures: 2B, 4A, 4C, 5B
  
- **Ground Truth Code (Authors)**: `retrocueing_RNN/`
  - `retrocue_model.py`: Model architecture, training, loss function
  - `generate_data_von_mises.py`: Data generation
  - `rep_geom_analysis.py`: Geometry analysis
  - `plane_angles_analysis.py`: Plane angle calculations
  - `subspace_alignment_index.py`: AI computation
  - `learning_dynamics_and_connectivity_analysis.py`: Plateau detection
  - `constants/constants_expt1.py`: Hyperparameters

- **Replication Code (AI Agents)**: `src/`, `scripts/`
  - `src/model.py`: RNN implementation
  - `src/task.py`: Task/dataset generation
  - `src/utils.py`: Loss function, utilities
  - `src/analysis.py`: Geometry analysis
  - `scripts/train.py`: Training loop
  - `scripts/analyze_geometry.py`: Geometry analysis
  - `scripts/stress_test.py`: Delay sweep experiment

- **Self-Reports**:
  - `reports/replication_report.tex`
  - `reports/internal_monologue.md`
  - `agent_logs.md`
  - `code_comparison.md`

- **Generated Outputs**:
  - `results/figures/fig4A_loss.png`
  - `results/figures/fig2B_planes.png`
  - `results/figures/fig4C_angles_ai.png`
  - `results/figures/fig5B_alignment_vs_delay.png`
  - `results/logs/training_summary.json`

---

## Step 2: Loss Function Verification (Eq. 2.1)

### 2.1 Original Paper Specification (Page 25)
The loss is defined as:
> L = mean((circ_dist * (target_1hot - output))^2)

Where `circ_dist` is the circular distance between each output unit's tuning center and the target angle.

### 2.2 Ground Truth Implementation (`retrocue_model.py`, lines 946-976)
```python
def custom_MSE_loss(params, output, target_scalar):
    target_1hot = make_target_1hot(params, target_scalar)
    circ_dist = helpers.circ_diff(params['phi'], target_scalar)
    loss = ((circ_dist * (target_1hot - output)) ** 2).mean()
    return loss
```
**Key observations**:
- Uses `circ_diff()` which wraps angles to [-π, π] (helpers.py line 301)
- `phi` is the tuning centers linspace from -π to π
- target_1hot is binary with a 1 at the matching phi index
- Loss computed on softmax output

### 2.3 Replication Implementation (`src/utils.py`, lines 113-128)
```python
def eq21_loss(logits: torch.Tensor, target_idx: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    target_onehot = F.one_hot(target_idx, num_classes=logits.shape[-1]).float().to(logits.device)
    target_angles = basis.to(logits.device)[target_idx]
    angle_term = (basis.to(logits.device)[None, :] - target_angles[:, None])
    loss = torch.mean(((target_onehot - probs) * angle_term) ** 2)
    return loss
```

### 2.4 **CRITICAL FINDING: Loss Function Deviation**
**Issue**: The replication DOES NOT wrap the angular difference to [-π, π].

The original code uses:
```python
circ_dist = helpers.circ_diff(params['phi'], target_scalar)
```

Which wraps via:
```python
def circ_diff(angle1, angle2):
    return wrap_angle(angle1 - angle2)

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
```

The replication simply computes:
```python
angle_term = basis - target_angles  # NO WRAPPING!
```

**Impact**: For angles near ±π boundaries, the replication will compute incorrect loss values. E.g., if target is at π-0.1 and an output unit is at -π+0.1, the true circular distance is 0.2, but the replication computes ~2π - 0.2 ≈ 6.08.

**Severity**: MODERATE-HIGH - affects gradient flow for boundary conditions

---

## Step 3: Input Encoding Verification (Eq. 1.1)

### 3.1 Paper Specification
Von Mises tuning with κ=5, scaled to peak at 1.

### 3.2 Ground Truth (`generate_data_von_mises.py`, lines 116-132)
```python
scale = max(vonmises.pdf(np.linspace(-np.pi, np.pi, 100), params['kappa_val'], loc=0))
c1p[:, c] = vonmises.pdf(c1, params['kappa_val'], phi[c]) / scale
```
Uses `scipy.stats.vonmises.pdf()`.

### 3.3 Replication (`src/utils.py`, lines 51-53)
```python
def von_mises_tuning(centers: np.ndarray, stimulus: float, kappa: float = 5.0) -> np.ndarray:
    return np.exp(kappa * np.cos(stimulus - centers)) / (2 * np.pi * np.i0(kappa))
```

**Observation**: Uses the von Mises PDF formula directly. Mathematically equivalent to scipy's implementation.

### 3.4 Task Structure (`src/task.py`, lines 59-66)
```python
self.colours = np.linspace(0, 2 * np.pi, config.num_colours, endpoint=False)
```

**Issue**: Colours are in [0, 2π) while paper uses [-π, π).

**Severity**: LOW - von Mises is periodic, so functionally equivalent if consistent throughout

---

## Step 4: Model Architecture Verification

### 4.1 Paper Specification
- 200 hidden units
- ReLU activation
- Orthogonal W_hh initialization
- Xavier W_ih, W_out initialization
- Gaussian hidden noise σ=0.07

### 4.2 Ground Truth (`retrocue_model.py`, lines 45-77)
```python
self.Wrec = nn.Parameter(torch.nn.init.orthogonal_(torch.empty((self.n_rec, self.n_rec))))
self.inp = nn.Linear(self.n_inp, self.n_rec)
self.inp.weight = nn.Parameter(self.inp.weight * params['init_scale'])
```
Note: Xavier init is achieved via PyTorch's default Linear initialization scaled by init_scale=1.

### 4.3 Replication (`src/model.py`, lines 40-47)
```python
def _init_parameters(self) -> None:
    nn.init.orthogonal_(self.rnn.weight_hh_l0)
    nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
    nn.init.zeros_(self.rnn.bias_hh_l0)
    nn.init.zeros_(self.rnn.bias_ih_l0)
    nn.init.xavier_uniform_(self.output.weight)
    nn.init.zeros_(self.output.bias)
```

**Finding**: Replication uses explicit `xavier_uniform_` while original uses default + scaling. Similar but not identical.

**Severity**: LOW - both are Xavier-like initializations

---

## Step 5: Training Dynamics & Plateau Detection

### 5.1 Paper Specification (Methods, Page 25-26)
Plateau detection:
1. Smooth loss with Gaussian filter (σ=3)
2. Compute derivative
3. Find first local maximum of derivative after loss falls below 95% of initial

### 5.2 Ground Truth (`learning_dynamics_and_connectivity_analysis.py`, lines 72-88)
```python
def find_learning_plateau(dLoss, loss_clean):
    ix = argrelextrema(np.array(dLoss), np.greater)[0]
    ix = np.setdiff1d(ix, ix[np.where(loss_clean[ix] >= loss_clean[0] * .95)[0]])
    plateau_ix = ix[0]
    return plateau_ix
```
Uses `scipy.signal.argrelextrema` for finding local maxima.

### 5.3 Replication (`src/utils.py`, lines 76-90)
```python
def detect_plateau_epoch(loss_history: np.ndarray, initial_loss: float) -> int:
    smoothed, derivative = smooth_and_derivative(loss_history)
    threshold_loss = initial_loss * 0.95
    valid_epochs = np.where(loss_history < threshold_loss)[0]
    if valid_epochs.size == 0:
        return int(len(loss_history) - 1)
    start_idx = valid_epochs[0]
    for i in range(start_idx + 1, len(derivative) - 1):
        if derivative[i - 1] < derivative[i] > derivative[i + 1]:
            return i
    return int(len(loss_history) - 1)
```

**Issue 1**: Replication uses `loss_history < threshold_loss` while original uses `loss_clean[ix] >= loss_clean[0] * .95`. Logic is inverted - replication looks for epochs BELOW threshold, original excludes epochs ABOVE threshold.

**Issue 2**: Replication uses σ=4 for smoothing (line 71: `sigma: float = 4.0`), paper uses σ=3.

**Severity**: MODERATE - affects which epoch is identified as plateau

---

## Step 6: Geometry Analysis

### 6.1 Alignment Index (AI) Computation

#### Ground Truth (`subspace_alignment_index.py`, lines 66-115)
```python
def get_simple_AI(X, Y, max_dim):
    X_preproc = (X - X.mean(0)[None, :]).T
    Y_preproc = (Y - Y.mean(0)[None, :]).T
    c_mat_x = np.cov(X_preproc)
    c_mat_y = np.cov(Y_preproc)
    # eigendecomposition...
    ai_Y_in_X = np.trace(np.dot(np.dot(eig_vecs_x.T, c_mat_y), eig_vecs_x)) / np.sum(eig_vals_y)
    ai_X_in_Y = np.trace(np.dot(np.dot(eig_vecs_y.T, c_mat_x), eig_vecs_y)) / np.sum(eig_vals_x)
    return (ai_Y_in_X + ai_X_in_Y) / 2
```
Data is transposed before covariance: `X.T` giving shape (neurons, conditions).

#### Replication (`src/analysis.py`, lines 95-107)
```python
def alignment_index(S1: np.ndarray, S2: np.ndarray, subspace_dim: int = 3) -> float:
    cov1 = np.cov(S1, rowvar=False)
    cov2 = np.cov(S2, rowvar=False)
    # projection...
```
Uses `rowvar=False` meaning input is (samples, features).

**Finding**: Both compute covariance on (conditions, neurons) data. Replication is correct.

### 6.2 Plane Angle Computation

#### Ground Truth (`subspace.py` - not fully examined but referenced)
Uses SVD to fit plane, computes angle between normal vectors.

#### Replication (`src/analysis.py`, lines 57-92)
Similar SVD-based approach with additional vertex ordering for orientation stability.

**Finding**: Replication adds orientation fixes not in original. May affect angle sign but not magnitude.

---

## Step 7: Figure Analysis

### 7.1 Fig 4A (Loss Curve)

**Paper Description**: Shows training loss with visible plateau "shelf" mid-training before continued descent.

**Replication Output**: 
- Loss starts at ~0.020
- Marked plateau at epoch 22
- Final loss ~0.004
- Curve is relatively smooth without pronounced plateau

**Visual Comparison**:
| Aspect | Paper | Replication |
|--------|-------|-------------|
| Plateau visibility | Pronounced shelf | Minimal/subtle |
| Final loss | Not specified | 0.004 |
| Epochs to converge | ~100-200 | 103-149 |

**Assessment**: The replication shows a smooth curve without the characteristic mid-training plateau. The derivative-based detection finds epoch 22, but there's no visible "shelf" in the loss.

### 7.2 Fig 2B (Geometry Planes)

**Paper Description**: 3D PCA projection with transparent planes fitted to each location's color representations. Pre-cue: orthogonal planes (~90°). Post-cue: aligned planes (~0°).

**Replication Output**:
- Shows 3D scatter with plane surfaces (alpha=0.2)
- Pre-cue: Planes not clearly orthogonal visually
- Post-cue: Planes partially aligned but not coplanar
- Axis labels show variance explained

**Visual Assessment**:
- Pre-cue θ ≈ 65.7° (paper target: ~90°)
- Post-cue θ ≈ 12.6° (paper target: ~0°)

**Finding**: Qualitative pattern correct (pre > post angle), but quantitative values don't reach paper extremes.

### 7.3 Fig 4C (Angles & AI)

**Paper Description**: Polar plots showing angle distributions for untrained/plateau/trained stages. Bar plots with AI values.

**Replication Output**:
- Polar plots show angle clusters
- Untrained: angles near 125°
- Plateau: angles clustered near 0-45°
- Trained: angles near 45-90°
- AI bars: Plateau shows AI=1.0 (saturated)

**Issues Identified**:
1. Plateau AI = 1.0 indicates saturation/clipping issue
2. Trained pre-cue angles don't reach 90°
3. Paper separates pre/post in different panels; replication combines

### 7.4 Fig 5B (AI vs Delay)

**Paper Description**: Post-cue AI increases monotonically with delay length.

**Replication Output**:
- Post-cue AI (blue): Increases from ~0.1 (delay=0) to ~0.94 (delay=7)
- Pre-cue AI (red): Relatively flat around 0.3-0.5

**Assessment**: Qualitative pattern matches paper - post-cue AI increases with delay. This is a successful aspect of the replication.

---

## Step 8: Self-Report Analysis

### 8.1 agent_logs.md Review

**Noted friction points**:
- Sandbox permission issues (correctly identified)
- AI > 1.0 saturation (correctly identified and logged as ERROR)
- Plateau detection (claimed working but visual evidence doesn't support pronounced plateau)

**Claimed successes**:
- "Eq.2.1 loss reproduced smooth colour tuning" - PARTIALLY TRUE (missing angle wrapping)
- "plane-fit visualization now shows clearer pre-orthogonal/post-parallel shift" - PARTIALLY TRUE (doesn't reach 90°/0°)

### 8.2 replication_report.tex Review

**Accurate self-criticisms**:
- "Pre-cue angles (~65°) and post angles (~13-53°) did not reach the ideal 90° → 0°"
- "Plateau AI saturated at 1.0 after clipping"
- "Loss curves smoother than the paper's pronounced plateau"

**Hallucinated successes**:
- "Eq. 2.1 loss explicitly implemented" - TRUE but INCORRECT (missing circ_diff wrapping)
- "Plane visualization shows clearer orthogonal-to-parallel shift" - PARTIAL (shift exists but magnitudes wrong)

### 8.3 internal_monologue.md Review

Shows methodical approach but critical oversight:
- "Reworked utilities to use... explicit Eq. 2.1 loss" - Implemented but with missing circ_diff wrapping
- Did not catch the loss function deviation from the paper

---

## Step 9: Quantitative Comparison

### Training Metrics
| Metric | Paper (N=30) | Replication (N=3) |
|--------|--------------|-------------------|
| Final loss | <0.0036 | 0.0018-0.0067 |
| Plateau epoch | ~variable | 22-25 |
| Total epochs | Up to 1500 | 103-149 |

### Geometry Metrics (Trained)
| Metric | Paper | Replication |
|--------|-------|-------------|
| Pre-cue θ | ~90° | 65-95° |
| Post-cue θ | ~0° | 13-53° |
| Pre-cue AI | ~0.2-0.4 | 0.74 (mean) |
| Post-cue AI | ~0.6-0.7 | 0.84 (mean) |

---

## Step 10: Root Cause Analysis

### Primary Failures
1. **Loss function bug**: Missing circular distance wrapping
2. **Insufficient seeds**: N=3 vs N=30 increases variance
3. **Early stopping criteria**: May terminate before full convergence

### Secondary Issues
1. Gaussian smoothing σ mismatch (4 vs 3)
2. Color space range difference ([0,2π) vs [-π,π))
3. AI saturation due to small sample size in geometry analysis

---

## Audit Complete: 2025-12-12

