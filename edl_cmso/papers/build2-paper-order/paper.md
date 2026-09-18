# A Secure and Efficient Deep Learning-Based Intrusion Detection Framework for the Internet of Vehicles

**Citation** · Hasim Khan, Ghanshyam G. Tejani, Rayed AlGhamdi, Sultan Alasmari, Naveen Kumar Sharma, Sunil Kumar Sharma — *Scientific Reports*, 2025 · DOI `10.1038/s41598-025-94445-9`
**Link** · https://doi.org/10.1038/s41598-025-94445-9   **Code** · none released
**Local copy** · [`../../s41598-025-94445-9.md`](../../s41598-025-94445-9.md)

## Problem

Intrusion detection in Internet-of-Vehicles (IoV) networks. Input is tabular
network/bus traffic records (flow features for CIC-IDS 2017, CAN-bus messages for
the CAN dataset, decimal payload frames for CICIoV2024). Output is a class label
— binary (normal / anomaly) or multi-class (attack type). The paper's stated
contribution is an end-to-end pipeline named **EDL-CMSO** that pairs a
privacy-preserving cryptographic front end with a hybrid deep classifier, and
claims both higher accuracy and lower inference cost than each of its four
backbone networks used alone.

## Method

The published pipeline has six stages. Stages 1–2 are cryptographic/preprocessing;
stages 3–5 are the learned model.

```
raw traffic
  → (1) IoVCipherGuard: HE + SMPC + AES-256          §4.3–4.6
  → (2) preprocessing: median imputation, Z-score     §4.7
  → (3) feature extraction: DWT → ViT → GAT → fusion  §4.8
  → (4) feature selection: CMSO (COA ⊕ MOA)           §4.9
  → (5) classification: DAGSNet                       §4.10
  → (6) AFPHA federated aggregation (FedAvg/FedProx/HFL) §4.6
```

### (3) Feature extraction — §4.8

**Discrete wavelet transform.** The input matrix $X \in \mathbb{R}^{n \times m}$
($n$ samples, $m$ features) is decomposed into approximation and detail bands:

$$X_{\text{wavelet}} = \mathrm{DWT}(X) = \{C_A, C_D\} \tag{19}$$

**ViT patch embedding.** $X_{\text{wavelet}}$ is cut into $k$ patches of size
$p \times p$, flattened and linearly projected to dimension $d$:

$$z_i = \mathrm{Linear}\!\left(P(X_{\text{wavelet}}[i])\right), \quad i \in [1,k] \tag{20}$$

$$z_i' = z_i + E_{\text{pos}}(i) \tag{21}$$

**Scaled dot-product attention.**

$$\mathrm{attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right)V \tag{22}$$

$$Q = z_i' W_Q, \quad K = z_i' W_K, \quad V = z_i' W_V \tag{23}$$

$$\mathrm{MultiHead}(z) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\,W_O \tag{24}$$

$$z_{\text{final}} = \mathrm{FFN}(\mathrm{MultiHead}(z)) + z \tag{25}$$

**GAT over the patch nodes.** The patches $z_{\text{final}}$ become nodes of a
graph $G=(V,E)$; the adjacency matrix is *built dynamically from the attention
coefficients* rather than given:

$$\alpha_{ij} = \frac{\exp\!\left(\mathrm{LeakyReLU}\!\left(a^{\top}\left[W_h z_i \,\|\, W_h z_j\right]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\!\left(\mathrm{LeakyReLU}\!\left(a^{\top}\left[W_h z_i \,\|\, W_h z_k\right]\right)\right)} \tag{26}$$

$$z^{\text{new}}_i = \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W_h z_j\right) \tag{27}$$

**Fusion.** The three representations are concatenated:

$$F = \mathrm{Concat}\!\left(X_{\text{wavelet}},\, z_{\text{final}},\, z^{\text{new}}_i\right) \tag{28}$$

### (4) Feature selection — CMSO, §4.9

A hybrid metaheuristic: the **exploration** phase of the Crayfish Optimization
Algorithm (COA) fused with the **exploitation** (upbringing) phase of the Mother
Optimization Algorithm (MOA). Population $X$ of $N$ candidates over $\dim$
dimensions; each candidate is a feature subset.

$$X_{i,j} = lb_j + (ub_j - lb_j)\times \mathrm{rand} \tag{30}$$

$$\text{temp} = \mathrm{rand}\times 15 + 20 \tag{31}$$

$$p = C_1 \times \frac{1}{\sqrt{2\pi}\,\sigma}\exp\!\left(-\frac{(\text{temp}-\mu)^2}{2\sigma^2}\right) \tag{32}$$

*Summer-resort (exploration), triggered when $\text{temp} > 30$:*

$$X_{\text{shade}} = \frac{X_G + X_L}{2} \tag{33}$$

$$X^{t+1}_{i,j} = X^{t}_{i,j} + C_2 \times \mathrm{rand} \times \left(X_{\text{shade}} - X^{t}_{i,j}\right) \tag{34}$$

$$C_2 = 2 - \frac{t}{T} \tag{35}$$

*Upbringing (MOA exploitation):*

$$x^{P3}_{i,j} = x_{i,j} + \left(1 - 2\,\mathrm{rand}(0,1)\right)\cdot\frac{ub_j - lb_j}{t} \tag{36}$$

$$X_i = \begin{cases} X^{P3}_i, & F^{P3}_i \le F_i \\ X_i, & \text{otherwise} \end{cases} \tag{37}$$

Fitness is stated only qualitatively: "classification performance (e.g., accuracy,
F-score)" on the candidate subset, plus a redundancy-minimising intent. A mutation
mechanism perturbs subsets to escape local optima.

### (5) Classification — DAGSNet, §4.10

Four backbones run **in parallel** on the selected feature map; their outputs are
concatenated and passed to a fully connected head.

*DenseNet* — every layer consumes all preceding feature maps:

$$F_l = H_l\!\left(\left[F_0, F_1, \dots, F_{l-1}\right]\right) \tag{38}$$

$$F_{\text{DenseNet}} = F_L \tag{39}$$

*GoogleNet* — inception modules, parallel multi-scale branches:

$$F_{\text{inception}} = \mathrm{Concat}\!\left(F_{1\times1}, F_{3\times3}, F_{5\times5}, F_{\text{pool}}\right) \tag{40}$$

$$F_{\text{GoogleNet}} = F_{\text{inception-final}} \tag{41}$$

*AlexNet* — plain conv + ReLU + max-pool stack:

$$F_l = \mathrm{ReLU}\!\left(\mathrm{Conv}(F_{l-1}) + b_l\right) \tag{42}$$

$$F_{\text{pooled}} = \mathrm{MaxPool}(F_l) \tag{43}$$

$$F_{\text{AlexNet}} = F_{\text{pooled-final}} \tag{44}$$

*SqueezeNet* — fire modules, a $1\times1$ squeeze feeding parallel $1\times1$/$3\times3$ expands:

$$F_{\text{squeeze}} = \mathrm{Conv}_{1\times1}(F_{\text{input}}) \tag{46}$$

$$F_{\text{Fire}} = \mathrm{Concat}\!\left(\mathrm{Conv}_{1\times1}(F_{\text{squeeze}}), \mathrm{Conv}_{3\times3}(F_{\text{squeeze}})\right) \tag{45}$$

*Integration:*

$$F_{\text{combined}} = \mathrm{Concat}\!\left(F_{\text{DenseNet}}, F_{\text{GoogleNet}}, F_{\text{AlexNet}}, F_{\text{SqueezeNet}}\right) \tag{47}$$

$$y = \sigma\!\left(W F_{\text{combined}} + b\right) \tag{48}$$

$$y \in [0,1] \ \text{(binary)} \tag{49}$$

$$y \in \mathbb{R}^{C} \ \text{(multi-class)} \tag{50}$$

Stated regularisation: batch normalisation, dropout, Adam with adaptive learning
rate, attention over traffic features. Post-training compression by pruning and
quantization is mentioned but not quantified.

## Training setup as published

| Item | Paper (Table 1) |
|---|---|
| Dataset | CIC-IDS 2017, CAN dataset, CICIoV2024DecimalCSV |
| Classes | binary (normal / anomaly) for the reported tables; Eq. (50) allows multi-class |
| Split | `unstated` |
| Preprocessing | median imputation; Z-score normalization |
| Feature extraction | GAT, ViT, wavelet transforms |
| Architecture | DAGSNet |
| Optimizer | listed as "CMSO"; §4.9.2 also states Adam with adaptive learning rates |
| Learning rate | 0.001 with adaptive decay |
| Batch size | 64 |
| Epochs | 100 |
| Loss | categorical cross-entropy |
| Activations | ReLU hidden, softmax output |
| Hardware | Intel Core i9, NVIDIA RTX 3090, 64 GB RAM, SSD |
| Frameworks | Python 3.8+, TensorFlow 2.x / PyTorch 1.x |
| Seeds / repeats | `unstated` |

## Reported results

**Table 3 — with and without feature selection** (dataset not identified in the table):

| Metric | Without FS | With FS |
|---|---|---|
| Accuracy | 0.980981 | **0.99125** |
| Precision | 0.973654 | **0.985433** |
| Recall | 0.97218 | **0.98452** |
| Sensitivity | 0.975436 | **0.985453** |
| Specificity | 0.98321 | **0.990158** |
| F-measure | 0.979421 | **0.98455** |
| NPV | 0.982157 | **0.990835** |
| FPR | 0.020045 | **0.001148** |
| FNR | 0.092452 | **0.00368** |
| MCC | 0.972166 | **0.984523** |

**Table 4 — CIC-IDS 2017**, proposed vs. each backbone alone:

| Metric | Proposed | AlexNet | DenseNet | SqueezeNet | GoogleNet |
|---|---|---|---|---|---|
| Sensitivity | **0.985** | 0.950 | 0.973 | 0.954 | 0.953 |
| Specificity | **0.990** | 0.954 | 0.972 | 0.951 | 0.949 |
| Accuracy | **0.991** | 0.966 | 0.979 | 0.957 | 0.955 |
| Precision | **0.985** | 0.951 | 0.972 | 0.950 | 0.949 |
| Recall | **0.985** | 0.940 | 0.973 | 0.934 | 0.929 |
| F-measure | **0.985** | 0.943 | 0.974 | 0.943 | 0.941 |
| NPV | **0.991** | 0.932 | 0.975 | 0.951 | 0.949 |
| FPR | **0.011** | 0.023 | 0.033 | 0.034 | 0.035 |
| FNR | **0.037** | 0.125 | 0.145 | 0.146 | 0.147 |
| MCC | **0.985** | 0.955 | 0.972 | 0.951 | 0.950 |
| Training time (s) | **320** | 400 | 560 | 340 | 450 |
| Testing time (s) | **85** | 100 | 110 | 92 | 98 |
| Inference time (ms) | **15** | 20 | 25 | 18 | 22 |

Tables 5 and 6 repeat the same layout for the CAN and CICIoV2024 datasets; the
abstract summarises the headline as precision 0.991 / 0.984 across two datasets.

## Gaps in the paper

Everything a reimplementation must decide, that the text does not fix:

1. **Wavelet family and level.** Eq. (19) names DWT and nothing else — no mother
   wavelet, no decomposition level, no boundary mode.
2. **How a tabular row becomes a 2-D patchable object.** Eq. (20) presupposes
   $p \times p$ patches, but the input is a flat feature vector. The reshape is
   never specified, nor is $p$, $k$, or $d$.
3. **ViT size.** Depth, number of heads $h$, FFN expansion ratio, dropout —
   all unstated.
4. **GAT graph.** The node set is the patch set, but the neighbourhood
   $\mathcal{N}(i)$ is never defined — Eq. (26) implies all-pairs, Eq. (27) sums
   over neighbours. Number of GAT layers and heads unstated.
5. **Shape mismatch in Eq. (28).** $X_{\text{wavelet}}$, $z_{\text{final}}$ and
   $z^{\text{new}}_i$ have different ranks; the concatenation axis and any
   pooling before it are unstated.
6. **CMSO hyperparameters.** $N$, $T$, $\dim$, $C_1$, $\mu$, $\sigma$, the
   mutation rate, the exact fitness function and its evaluator are all unstated.
   Equations (31)–(35) are continuous-valued while feature selection is binary —
   no binarisation rule (transfer function, threshold) is given.
7. **What CMSO actually optimises.** Table 1 lists CMSO under "Optimizer",
   while §4.9 describes it as a feature selector and §4.9.2 says Adam trains the
   network. The two roles are never reconciled.
8. **Backbone depths.** Growth rate and block counts for DenseNet, the number of
   inception modules, AlexNet's channel widths, the number of fire modules — none
   given. Whether the backbones are 1-D or 2-D convolutional is not stated.
9. **Data split.** No train/validation/test ratio, no statement of whether the
   Z-score statistics come from train only.
10. **Class definition.** The reported metrics are binary-shaped (TP/TN/FP/FN,
    specificity, NPV, MCC as in Eq. 56) although Eq. (50) permits multi-class.
11. **Federated setup.** AFPHA is defined qualitatively; client count, data
    partitioning, the FedProx $\mu$, cluster structure for HFL and the number of
    communication rounds are all unstated.
12. **Pruning / quantization** are proposed but never applied to any reported number.
