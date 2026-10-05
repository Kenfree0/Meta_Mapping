# Mechanistic Evidence for Conceptual Mapping in LLM Metaphor Understanding

Official implementation for **"Mechanistic Evidence for Conceptual Mapping in LLM Metaphor Understanding"**.

This repository provides the code and example data for investigating whether large language models (LLMs) internally represent **cross-domain conceptual mappings** during metaphor understanding.

<p align="center">
  <img src="figures/fig.pdf" width="900">
</p>

Our framework uses controlled metaphor contrasts to separate **semantic conflict** from **conceptual mapping**, and analyzes model representations from three complementary perspectives:

- **Activation analysis** — identifies MLP neurons sensitive to mapping validity.
- **Representation geometry** — examines whether mapping-consistent expressions preserve similar hidden-state subspaces.
- **Causal Analysis** — tests whether mapping-sensitive neurons functionally contribute to source-domain recovery.

---

## Overview

For each original metaphor \(M\), we construct three controlled variants:

| Variant | Description | Semantic Conflict | Valid Mapping |
|---|---|---:|---:|
| **M** | Original metaphor | ✓ | ✓ |
| **R** | Mapping-consistent source replacement | ✓ | ✓ |
| **U** | Mapping-inconsistent source replacement | ✓ | ✗ |
| **L** | Literal paraphrase | ✗ | ✗ |

Example:

| Variant | Sentence |
|---|---|
| **M** | The car is a slug. |
| **R** | The car is a snail. |
| **U** | The car is a rainbow. |
| **L** | The car is very slow. |

The central hypothesis is that, if LLMs internally represent conceptual mappings, representations elicited by \(M\) should be systematically closer to those elicited by \(R\) than by \(U\), despite both \(R\) and \(U\) preserving semantic conflict.

---

## Repository Structure

```text
Metaphor_Mapping/
├── extraction/          # Extract MLP activations and hidden representations
├── activation/          # Activation-pattern and mapping-neuron analyses
├── geometry/            # Hidden-state subspace geometry analysis
├── causal/              # Causal intervention and shared ablation utilities
├── model_adapters/      # Model-specific adapters
├── data/                # Chinese and English example data
├── models/              # Local model checkpoints (weights not included)
├── outputs/             # Extracted representations and analysis results
├── causal_outputs/      # Causal intervention results
├── figures/             # Figures used in the paper / README
├── tests/               # Regression tests
├── requirements.txt
└── model_paths.example.json
```

---

## Supported Models

The experiments currently support the following models:

- DeepSeek-LLM-7B
- Gemma 3-12B
- GLM-4-9B
- Llama-3.1-8B
- Qwen2.5-7B

Each model is evaluated independently on both **Chinese (`cn`)** and **English (`en`)** data.

Available model keys are:

```text
Gemma12b
qwen2.5-7b
llama-3.1-8b
deepseek_base
glm4-9b-0414
```

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/<USERNAME>/Metaphor_Mapping.git
cd Metaphor_Mapping

python -m pip install torch
python -m pip install -r requirements.txt
```

Model weights are **not included** in this repository.

Place the complete model checkpoints under:

```text
models/
```

and configure their paths according to:

```text
model_paths.example.json
```

---

## Running the Experiments

All commands should be executed from the project root directory.

### 1. Feature Extraction

Extract MLP activations and hidden representations:

```bash
python extraction/extract_features.py \
    --model qwen2.5-7b \
    --lang cn
```

For English:

```bash
python extraction/extract_features.py \
    --model qwen2.5-7b \
    --lang en
```

Feature extraction must be completed before running the activation and geometry analyses.

---

### 2. Activation Analysis

Identify mapping-sensitive activation patterns and neurons:

```bash
python activation/analyze_mapping_neurons.py \
    --model qwen2.5-7b \
    --lang cn
```

The analysis compares activation patterns across the four controlled conditions \(M\), \(R\), \(U\), and \(L\).

Mapping-sensitive neurons are identified from MLP activations according to the controlled mapping contrasts described in the paper.

---

### 3. Representation Geometry

Analyze whether mapping consistency is reflected in hidden-state subspace geometry:

```bash
python geometry/analyze_subspaces.py \
    --model qwen2.5-7b \
    --lang cn \
    --k 10
```

We construct representation shifts relative to the literal condition \(L\), apply PCA separately to the \(M\), \(R\), and \(U\) conditions, and retain the top **10 principal components**.

The resulting subspaces are compared using:

- principal angles;
- chordal Grassmann distance.

The main test examines whether the representation subspace of \(M\) is closer to \(R\) than to \(U\).

---

### 4. Causal Intervention

Run neuron-ablation experiments:

```bash
python causal/analyze_causality.py \
    --models qwen2.5-7b \
    --langs cn \
    --percents 1 2 5 10 20 30 50 100
```

The causal analysis compares:

1. the original model without intervention;
2. targeted ablation of mapping-sensitive neurons;
3. equal-sized random neuron ablation.

Mapping-sensitive neurons used for causal validation are selected from the \(R/U/L\) conditions and evaluated on the original metaphor \(M\), providing a cross-condition test that reduces dependence on the surface form of \(M\).

Results are saved to:

```text
causal_outputs/
```

---

## Data

The study uses controlled four-way metaphor contrast sets in **Chinese and English**.

Each example contains:

```text
M: Original metaphor
R: Mapping-consistent variant
U: Mapping-inconsistent variant
L: Literal paraphrase
```

The complete experimental dataset contains:

- **1,690 Chinese contrast sets**
- **1,541 English contrast sets**

Each generated contrast set was independently evaluated by three annotators, and only examples unanimously accepted by all three annotators were retained.

For the current anonymous submission, this repository provides:

- **10 Chinese examples**
- **10 English examples**

The complete Chinese and English datasets will be released upon paper acceptance.

Example files are located in:

```text
data/
├── chinese_samples.json
└── english_samples.json
```

---

## Main Analyses

The repository implements three complementary levels of mechanistic analysis.

### Activation Patterns

We test whether the activation-effect profile of the original metaphor \(M\) is more similar to the mapping-consistent condition \(R\) than to the mapping-inconsistent condition \(U\):

```text
M ≈ R ≠ U
```

This analysis is performed over MLP neuron activations across Transformer layers.

### Representation Geometry

We examine the geometry of hidden-state shifts relative to the literal condition:

```text
ΔM = H(M) - H(L)
ΔR = H(R) - H(L)
ΔU = H(U) - H(L)
```

The corresponding PCA subspaces are compared using principal-angle and Grassmann-distance measures.

### Causal Intervention

Finally, we ablate mapping-sensitive MLP neurons and test whether targeted intervention decreases the model's preference for recovering the correct source-domain expression more strongly than matched random ablation.

This provides causal evidence beyond correlational activation and representation analyses.

---

## Reproducing Other Model / Language Configurations

Replace the model and language arguments in the commands above.

For example:

```bash
python extraction/extract_features.py \
    --model llama-3.1-8b \
    --lang en

python activation/analyze_mapping_neurons.py \
    --model llama-3.1-8b \
    --lang en

python geometry/analyze_subspaces.py \
    --model llama-3.1-8b \
    --lang en \
    --k 10
```

The same pipeline applies to all supported model-language configurations.

---

## License

The source code is released for research purposes.

Please refer to the licenses of the original metaphor datasets and pretrained language models before redistributing derived data or model-related resources.