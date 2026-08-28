# Graph Report - SELE_Regularization  (2026-08-28)

## Corpus Check
- 87 files · ~376,277 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 425 nodes · 789 edges · 20 communities (16 shown, 4 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 27 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Data Loading
- Score Model Training
- Mesh Building & Run Reporting
- Regularization Operators
- Pipeline Orchestration & Config
- Test Set Construction
- Score Model Architecture & Solver
- Test Set Loader
- SELE Physics & Forward Model
- Score Model Testing & Visualization
- MATLAB Figure Reader
- Diffusion Model Testing
- New Model Toolkit Planning
- DVC Data Management
- Plotting Conventions
- Unused Optics Figure
- Project Root

## God Nodes (most connected - your core abstractions)
1. `load_csv()` - 21 edges
2. `run_regularization()` - 16 edges
3. `solve_gradient_descent()` - 14 edges
4. `ScoreNetwork` - 13 edges
5. `load_score_model()` - 12 edges
6. `RegularizationMethod` - 12 edges
7. `read_fig()` - 11 edges
8. `calc_mesh_and_G()` - 11 edges
9. `save_csv()` - 10 edges
10. `TestCurve` - 10 edges

## Surprising Connections (you probably didn't know these)
- `NON_UNIFORM_MESH regularization mode` --cites--> `Mapping Losses through Empirical Extraction of the Spatial External Luminescence Efficiency (Yeshurun, Fiegenbaum-Raz, Segev; ACS Appl. Energy Mater. 2024)`  [EXTRACTED]
  CLAUDE.md → Papers/Mapping Losses through Empirical Extraction of the Spatial External Luminescence Efficiency.pdf
- `Reflectance cancels via phi_abs = phi_0*A; no back-surface contribution to recover` --semantically_similar_to--> `phi_abs = phi_0*A normalizes reflectance out of the forward model`  [INFERRED] [semantically similar]
  Data/test_set/README.md → CLAUDE.md
- `Score Model Standalones section (root README)` --semantically_similar_to--> `Standalones folder overview (score-model tuning + model_training)`  [INFERRED] [semantically similar]
  README.md → src/regularization/score_model/standalones/README.md
- `Forward model: eta_ext = (1/phi_abs) * integral(DeltaG*S) dz` --cites--> `Mapping Losses through Empirical Extraction of the Spatial External Luminescence Efficiency (Yeshurun, Fiegenbaum-Raz, Segev; ACS Appl. Energy Mater. 2024)`  [EXTRACTED]
  CLAUDE.md → Papers/Mapping Losses through Empirical Extraction of the Spatial External Luminescence Efficiency.pdf
- `Math symbols cheat sheet` --conceptually_related_to--> `Forward model: eta_ext = (1/phi_abs) * integral(DeltaG*S) dz`  [INFERRED]
  Utils/math symbols.txt → CLAUDE.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Three regularization modes dispatched by the pipeline** — claude_non_uniform_mesh_mode, claude_total_variation_mode, claude_model_score_grad_mode, claude_pipeline_src [EXTRACTED 1.00]
- **ScoreNetwork(...) constructed identically in four separate files, motivating the build_score_network factory** — claude_score_model_grad_src, src_regularization_score_model_standalones_model_training_readme_training_script, plans_score_network_conv1d_plan_test_score_models_src, plans_score_network_conv1d_plan_sele_w_score_optimization_example_src [EXTRACTED 1.00]
- **Physical parameters that jointly shape the SELE depth profile** — claude_srv, claude_tau_srh, claude_diffusion_length, claude_photon_recycling, claude_sele [EXTRACTED 1.00]

## Communities (20 total, 4 thin omitted)

### Community 0 - "Data Loading"
Cohesion: 0.08
Nodes (40): Path, load_csv(), _load_csv_vector(), load_eta(), load_G(), load_score_model_S(), ndarray, Data loading and saving utilities. (+32 more)

### Community 1 - "Score Model Training"
Cohesion: 0.06
Nodes (35): DataLoader, device, Optimizer, Tensor, Initialize network weights. Kaiming for residual mode, Xavier for legacy., Forward pass through the score network. Args: x: Noisy data tensor of shape…, Args: t: Time tensor of shape (batch_size, 1) Returns: Embedding of shape…, Hidden layer with a skip connection. (+27 more)

### Community 2 - "Mesh Building & Run Reporting"
Cohesion: 0.09
Nodes (38): ArrayLike, generate_run_report(), Generate run report compatible with both 1-D and 2-D modes., Entry point for running in PyCharm or any IDE., calc_mesh_and_G(), _compute_front_generation(), _linear_mesh(), _non_uniform_mesh() (+30 more)

### Community 3 - "Regularization Operators"
Cohesion: 0.09
Nodes (35): SELE toolbox with non‑uniform mesh support., build_L(), ndarray, Derivative / regularization operators., Return the regularization matrix L. Parameters ---------- flag 'L0', 'L1', or…, find_knee(), NDArray, Core Tikhonov solver and κ-sweep utilities. (+27 more)

### Community 4 - "Pipeline Orchestration & Config"
Cohesion: 0.07
Nodes (34): src/types/config.py (CONFIG, ModelScoreGradConfig, presets), Ground-truth SELE test set (src/test_set), src/main.py entry point, MODEL_SCORE_GRAD regularization mode, NON_UNIFORM_MESH regularization mode, src/pipeline.py (run_regularization), tikhonov_non_uniform.py, TOTAL_VARIATION regularization mode (+26 more)

### Community 5 - "Test Set Construction"
Cohesion: 0.13
Nodes (34): save_csv(), build_test_set(), edges_from_samples(), _existing_ground_truth(), _extract_srv_curves(), _extract_tau_curves(), _first_axes_with_profiles(), _inset_series() (+26 more)

### Community 6 - "Score Model Architecture & Solver"
Cohesion: 0.07
Nodes (34): src/regularization/score_model/model_definition.py (ScoreNetwork), Nesterov Accelerated Gradient solver (solve_gradient_descent), Score Model deep dive (architecture, solver, normalization, weighting, training data), src/regularization/score_model/score_model_grad.py, ScoreNetwork model (d32/d500 checkpoints), Dhariwal & Nichol 2021, Diffusion Models Beat GANs (ADM), Conv1dScoreNetwork proposed architecture (dilated residual conv stack), Kong et al. 2021, DiffWave (+26 more)

### Community 7 - "Test Set Loader"
Cohesion: 0.12
Nodes (24): _build_curve(), GroundTruthCurve, _index_row(), _index_rows(), load_curve(), load_native_G(), load_on_solver_mesh(), load_test_set() (+16 more)

### Community 8 - "SELE Physics & Forward Model"
Cohesion: 0.10
Nodes (26): alpha_b: absorption coefficient excluding free-carrier term, alpha: absorption coefficient including free-carrier (Drude) term, Minority carrier diffusion length, ELE (External Luminescence Efficiency), Forward model: eta_ext = (1/phi_abs) * integral(DeltaG*S) dz, G matrix (Beer-Lambert photogeneration operator), G is a generation matrix: rows sum to alpha_b/alpha, not 1, src/mesh.py (G matrix builder) (+18 more)

### Community 9 - "Score Model Testing & Visualization"
Cohesion: 0.13
Nodes (22): Figure, attach_copy_shortcut(), _copy_fig_to_clipboard(), get_alon_model_grad(), get_my_trained_model_grad(), load_correct_curve(), load_my_model(), load_pipeline_gt() (+14 more)

### Community 10 - "MATLAB Figure Reader"
Cohesion: 0.15
Nodes (21): Any, _as_list(), FigAxes, FigLine, float64, NDArray, Minimal reader for MATLAB ``.fig`` files (HG2 / MAT v5 serialization). A…, Return one :class:`FigAxes` per axes object in the figure, in stored order. (+13 more)

### Community 11 - "Diffusion Model Testing"
Cohesion: 0.14
Nodes (16): Sequential, _AlonModelWrapper, _load_model(), main(), plot_generated_samples(), Module, ndarray, Tensor (+8 more)

### Community 12 - "New Model Toolkit Planning"
Cohesion: 0.50
Nodes (4): test_diffusion_generation.py bug fixes: remove clamp, power-law time grid, roughness readout (planned), auto_tune_hyperparameters.py, new_model_toolkit/main.py, test_diffusion_generation.py

## Knowledge Gaps
- **34 isolated node(s):** `sele-regularization`, `SRH lifetime (tau_SRH)`, `Minority carrier diffusion length`, `src/main.py entry point`, `src/mesh.py (G matrix builder)` (+29 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ScoreNetwork` connect `Score Model Training` to `Data Loading`, `Score Model Testing & Visualization`?**
  _High betweenness centrality (0.118) - this node is a cross-community bridge._
- **Why does `load_csv()` connect `Data Loading` to `Mesh Building & Run Reporting`, `Test Set Construction`, `Test Set Loader`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Why does `train_model()` connect `Score Model Training` to `Data Loading`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `solve_gradient_descent()` (e.g. with `Path` and `ModelScoreGradConfig`) actually correct?**
  _`solve_gradient_descent()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `sele-regularization`, `SRH lifetime (tau_SRH)`, `Minority carrier diffusion length` to the rest of the system?**
  _34 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Data Loading` be split into smaller, more focused modules?**
  _Cohesion score 0.0803633822501747 - nodes in this community are weakly interconnected._
- **Should `Score Model Training` be split into smaller, more focused modules?**
  _Cohesion score 0.06037414965986394 - nodes in this community are weakly interconnected._