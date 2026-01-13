# Decision-Making Criteria on Choosing Appropriate Imputation Methods for Incomplete Datasets Prepared for Machine Learning

This repository provides the **code, datasets, and experimental framework** supporting the paper:

> **Decision-Making Criteria on Choosing Appropriate Imputation Methods for Incomplete Datasets Prepared for Machine Learning**

The goal of this work is **not to propose a new imputation algorithm**, but to establish and empirically evaluate **practical validation criteria** that help researchers and practitioners **choose appropriate imputation methods** when working with incomplete datasets for downstream machine learning tasks.

---

## 1. Motivation

Missing data is ubiquitous in real-world datasets. While many imputation methods exist, **selecting an appropriate imputer** for a given dataset remains a practical challenge.

Two validation paradigms are commonly used:

1. **Direct imputation error evaluation** using artificially introduced missingness (e.g., MCAR-based masking).
2. **Downstream task evaluation**, where imputers are judged indirectly by model prediction performance.

However, both paradigms have limitations:
- Direct evaluation may be infeasible when complete cases are scarce.
- Downstream evaluation may fail to distinguish imputers with similar predictive performance.

This repository implements a **systematic validation framework** that:
- formalizes multiple practical criteria,
- evaluates their **feasibility** and **distinguishability**, and
- provides guidance on when each criterion is appropriate.

---

## 2. Validation Criteria Implemented

The framework implements the following criteria:

### Baseline (Ideal Reference)
- Direct comparison between imputed values and ground truth.
- Uses the fully observed dataset.
- **Not available in real-world scenarios**, used only as a reference.

### Criterion 1.1 — MCAR Proxy Validation
- Artificial missingness introduced under MCAR on validation data.
- Evaluation restricted to entries with known ground truth.
- Reflects the most common practice in imputation studies.

### Criterion 1.2 — Mechanism-Aligned Proxy Validation
- Artificial missingness introduced using the *same mechanism* as the simulated missingness (MCAR, MAR, MNAR, and pairwise variants).
- Provides a closer approximation to realistic missing data processes.

### Criterion 2 — Downstream Task Performance
- Imputed datasets are evaluated via supervised learning models.
- Measures prediction error instead of direct imputation error.

Each criterion is evaluated in terms of:
- **Feasibility** (whether it can be applied under given data conditions),
- **Sensitivity / distinguishability** between imputers.

---

## 3. Datasets

Six benchmark datasets are included and preprocessed:

| ID | Dataset | Type |
|----|--------|------|
| 1 | Concrete compressive strength | Numerical |
| 2 | Composite material dataset | Numerical |
| 3 | Steel strength | Numerical |
| 4 | Energy efficiency | Mixed (categorical + numerical) |
| 5 | Student performance | Mixed |
| 6 | Wine quality | Numerical |

All datasets are stored in the `data/` directory and loaded via `datasets.py`.

---

## 4. Imputation Methods Evaluated

The following imputation methods are included:

- Mean / Mode
- Hot-deck
- k-Nearest Neighbors (kNN)
- Predictive Mean Matching (PMM)
- LightGBM-based imputation
- CatBoost-based imputation

These represent a range from **simple statistical** to **machine-learning-based** imputers.

---

## 5. Repository Structure

```text
Validation_framework_for_imputation/
├── data/                         # Raw datasets
│   ├── composite_1.xlsx
│   ├── concrete_X.csv
│   ├── concrete_y.csv
│   ├── Energy_efficiency.csv
│   ├── steel_strength.csv
│   ├── student_X.csv
│   ├── student_y.csv
│   └── wine.csv
│
├── datasets.py                   # Dataset loading and preprocessing
├── validation_v2.py              # Core validation framework (Criteria 1 & 2)
├── run_one_block.py              # Run one (dataset × mask × rate) experiment
├── run_parallel_blocks.py        # Parallel execution of multiple blocks
│
├── results/                      # Saved experiment outputs (.pkl)
│   ├── Concrete/
│   ├── Composite/
│   ├── Steel/
│   ├── Energy/
│   ├── Student/
│   └── Wine/
│
├── figures_and_tables/           # Generated figures and tables
│   ├── Figure5.png
│   ├── Figure6.png
│   ├── Figure7_Concrete_Composite_Steel.png
│   ├── Figure7_Energy_Student_Wine.png
│   └── AppendixTables.xlsx
│
├── plot_for_validation_v2.py     # Figure and table generation
└── README.md
```

---

## 6. Running Experiments

### 6.1 Run a Single Experiment Block

```bash
python run_one_block.py \
  --dataset-id 6 \
  --mask MCAR \
  --rate 0.05 \
  --n-repeats 100
```
This runs one dataset under a specific missingness mechanism and rate.


### 6.2 Run All Blocks in Parallel
```bash
python run_parallel_blocks.py \
  --dataset-id 6 \
  --n-jobs 6 \
  --n-repeats 100
```

Each block corresponds to one combination of:
- missingness mechanism,
- missing rate.
Intermediate progress is tracked via text files to allow recovery from interruptions.

## 7. Expected Runtime and Computational Resources

### 7.1 Hardware Environment

All large-scale experiments in this repository were executed on the **Imperial College London CX3 High Performance Computing (HPC) cluster**, using the following job configuration:

```bash
#PBS -l select=1:ncpus=16:mem=64gb
#PBS -l walltime=72:00:00
```

---

### 7.2 Experimental Granularity

Experiments are organised at the level of **independent blocks**, where each block corresponds to:

- one dataset,
- one missing rate (e.g. 5%, 10%, …, 30%),
- all missingness mechanisms (MCAR, MAR, MNAR, and pairwise variants),
- a fixed number of repetitions (typically 100).

Each block can be executed independently and in parallel, which enables efficient scaling across HPC nodes.

---

### 7.3 Typical Walltime per Block

Based on representative runs (100 repetitions per block, 16 CPU cores), the **walltime per block** varies substantially across datasets due to differences in:

- sample size,
- feature dimensionality,
- presence of categorical variables,
- complexity of downstream predictive models.

Observed walltimes are approximately:

| Dataset | Missing Rate (example) | Walltime (hh:mm) | CPU Time Used |
|-------|------------------------|------------------|---------------|
| Concrete | 25% | ~6:00 | ~77 CPU-hours |
| Composite | 20% | ~7:00 | ~88 CPU-hours |
| Steel | 15% | ~6:20 | ~75 CPU-hours |
| Energy | 10% | ~4:00 | ~48 CPU-hours |
| Student | 20% | ~9:40 | ~118 CPU-hours |
| Wine | 30% | ~17:20 | ~195 CPU-hours |

These values correspond to **single blocks** (one dataset × one missing rate × all missingness mechanisms × 100 repetitions).

---

### 7.4 Full Experimental Scale

A full experimental sweep for one dataset typically includes:

- 6 missing rates,
- all missingness mechanisms per rate,
- 100 repetitions per combination,

resulting in **6 independent blocks per dataset**.

Because blocks are embarrassingly parallel, the *total walltime* depends primarily on the number of concurrent jobs available rather than the aggregate computational load.

---

### 7.5 Practical Guidance

- On a personal workstation, users are advised to:
  - reduce the number of repetitions (e.g. 5–10),
  - limit the number of missingness mechanisms or rates,
  - or run blocks sequentially.

- On HPC systems, full reproduction is feasible by submitting blocks in parallel using `run_parallel_blocks.py`.

This design ensures that the validation framework remains **scalable, modular, and reproducible** across a wide range of computational environments.

## 8. Software Environment

All experiments were conducted using **Python 3.11**.  
The code has been tested on both local machines and the Imperial College London **CX3 HPC cluster**.

### 8.1 Core Dependencies

The validation framework relies on the following Python packages:

- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `seaborn`
- `joblib`
- `lightgbm`
- `catboost`

All dependencies are standard and widely used in the machine learning and data science community.

---

### 8.2 Example Environment Setup

A minimal environment can be created using `conda`:

```bash
conda create -n imputation_validation python=3.11
conda activate imputation_validation

pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib lightgbm catboost
```

### 8.3 HPC Environment Notes

Large-scale experiments were executed on the Imperial College London CX3 HPC cluster, using:

- Linux operating system
- PBS job scheduler
- CPU-only nodes (no GPU acceleration required)

No platform-specific code is used; all scripts are portable across Linux, macOS, and Windows systems with a compatible Python environment.

### 8.4 Reproducibility

- All random seeds are explicitly controlled within the experimental scripts.
- Each experimental block (dataset × missing rate) is independent and restartable.
- Intermediate and final results are saved as serialized .pkl files in the results/ directory.

This design ensures full reproducibility of all figures and tables reported in the paper.

## 9. Result Processing and Visualization

After experiments complete, figures and tables are generated via:
```markdown
python plot_for_validation_v2.py
```
This script:
-aggregates block-level results,
-reproduces all figures used in the paper,
-exports appendix tables to Excel.

## 10. Reproducibility

- All random seeds are fixed.
- Missingness mechanisms are explicitly controlled.
- Train/validation/test splits are consistent across imputers.
- Results are saved incrementally to avoid data loss.

## 11. Scope and Limitations

This repository focuses on **evaluation criteria for imputation methods**, rather than on developing new imputation algorithms.

Advanced generative imputers (e.g., MIWAE, GAN-based methods) are not included in this implementation due to their **substantially higher computational cost**, sensitivity to hyperparameter tuning, and limited scalability in large-scale benchmarking settings. These characteristics make them less suitable for systematic, repeated evaluation across multiple datasets, missingness mechanisms, and missing rates, which is the primary focus of this study.

The proposed framework is designed to be **method-agnostic** and can, in principle, accommodate such models. A dedicated investigation of generative imputers and enhanced proxy-based validation criteria is therefore deferred to a separate follow-up study.


## 12. Citation

If you use this code or framework, please cite the corresponding paper:

```bibtex
@article{Geng2025ImputationCriteria,
  title={Decision-Making Criteria on Choosing Appropriate Imputation Methods for Incomplete Datasets Prepared for Machine Learning},
  author={Geng, Xiangdong and Wu, Chao and Li, Yingzhen},
  year={2025},
  note={Manuscript under review}
}
```
Note: The paper is currently under review. This entry will be updated with journal and DOI information upon publication.

## 13. Contact

For questions or collaboration, please contact:

Chao Wu

Department of Civil and Environmental Engineering
Imperial College London