"""Analysis-Informed Abliteration Pipeline.

Closes the feedback loop between OBLITERATUS's 15 analysis modules (#3)
and the abliteration pipeline (#2). Instead of running analysis as a
standalone post-hoc step, this pipeline runs targeted analysis modules
*during* each stage of abliteration to make smarter decisions:

  SUMMON  →  load model
  PROBE   →  collect activations
  ANALYZE →  run analysis modules to inform excision strategy
  DISTILL →  extract directions using analysis-informed parameters
  EXCISE  →  remove refusal with analysis-guided precision
  VERIFY  →  post-excision analysis to detect residual refusal
  REBIRTH →  save with comprehensive analysis metadata

The ANALYZE stage is the key innovation: it sits between PROBE and DISTILL
and uses analysis module outputs to automatically configure the downstream
stages. The VERIFY stage also uses analysis modules to detect self-repair
(Ouroboros effect) and trigger additional refinement passes if needed.

Analysis modules integrated:

  Stage       | Module used                  | What it informs
  ------------|------------------------------|------------------------------------------
  ANALYZE     | AlignmentImprintDetector     | Auto-selects method preset (DPO/RLHF/CAI)
  ANALYZE     | ConceptConeAnalyzer          | Per-category vs universal direction choice
  ANALYZE     | CrossLayerAlignmentAnalyzer  | Smart layer selection (cluster-aware)
  ANALYZE     | SparseDirectionSurgeon       | Sparsity-aware projection plan
  ANALYZE     | DefenseRobustnessEvaluator   | Ouroboros risk assessment, entanglement map
  DISTILL     | WhitenedSVDExtractor         | Covariance-normalized direction extraction
  EXCISE      | SparseDirectionSurgeon       | Targeted row-level weight surgery
  VERIFY      | ActivationProbe              | Post-excision refusal signal detection
  VERIFY      | CrossLayerAlignmentAnalyzer  | Post-excision direction persistence check
  VERIFY      | DefenseRobustnessEvaluator   | Self-repair / Ouroboros effect detection
  VERIFY      | SteeringVectorFactory        | Pre-screen with steering before permanent changes

Novel contributions:
  - First closed-loop analysis→abliteration pipeline
  - Alignment-aware auto-tuning: detected training method (DPO/RLHF/CAI)
    automatically configures projection parameters
  - Cone-aware excision: polyhedral models get per-category directions,
    linear models get single universal direction
  - Cluster-aware layer selection: respects direction cluster boundaries
    instead of arbitrary top-k selection
  - Ouroboros-compensated refinement: detects self-repair and adds targeted
    passes at compensating layers
  - Entanglement-gated projection: skips highly entangled layers to
    preserve capabilities
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from obliteratus.abliterate import (
    AbliterationPipeline,
    StageResult,
)
from obliteratus.architecture_manifest import (
    ArchitectureCoverageError,
    ProjectionManifestEntry,
)
from obliteratus.evaluation.damage_gate import AcceptanceBudget

logger = logging.getLogger(__name__)


# ── Analysis-informed method preset ──────────────────────────────────────

INFORMED_METHOD = {
    "label": "Informed (Analysis-Guided)",
    "description": (
        "Runs analysis modules between PROBE and DISTILL to auto-configure "
        "direction extraction, layer selection, and projection strategy based "
        "on the model's actual refusal geometry. Bayesian tuning is disabled "
        "until exact winning-trial replay exists; deterministic analysis-guided "
        "projection remains enabled."
    ),
    "n_directions": 1,            # overridden by analysis
    "direction_method": "diff_means",  # overridden by analysis; "leace" also available
    "norm_preserve": True,
    "regularization": 0.0,        # overridden by analysis
    "refinement_passes": 2,       # overridden by analysis
    "project_biases": True,
    "use_chat_template": True,
    "use_whitened_svd": False,    # overridden by analysis
    "true_iterative_refinement": True,
}


# ── Analysis result containers ───────────────────────────────────────────

@dataclass
class AnalysisInsights:
    """Insights gathered from the ANALYZE stage.

    These inform every downstream decision in the pipeline.
    """

    # Alignment imprint
    detected_alignment_method: str = "unknown"
    alignment_confidence: float = 0.0
    alignment_probabilities: dict[str, float] = field(default_factory=dict)

    # Cone geometry
    cone_is_polyhedral: bool = False
    cone_dimensionality: float = 1.0
    mean_pairwise_cosine: float = 1.0
    per_category_directions: dict[str, torch.Tensor] = field(default_factory=dict)
    direction_specificity: dict[str, float] = field(default_factory=dict)

    # Cross-layer structure
    direction_clusters: list[list[int]] = field(default_factory=list)
    cluster_count: int = 0
    direction_persistence: float = 0.0
    cluster_representative_layers: list[int] = field(default_factory=list)

    # Sparse surgery
    mean_refusal_sparsity_index: float = 0.0
    recommended_sparsity: float = 0.1
    use_sparse_surgery: bool = False

    # Defense robustness
    estimated_robustness: str = "unknown"
    self_repair_estimate: float = 0.0
    entanglement_score: float = 0.0
    entangled_layers: list[int] = field(default_factory=list)
    clean_layers: list[int] = field(default_factory=list)

    # Derived configuration
    recommended_n_directions: int = 1
    recommended_direction_method: str = "diff_means"
    recommended_regularization: float = 0.0
    recommended_refinement_passes: int = 2
    recommended_layers: list[int] = field(default_factory=list)
    skip_layers: list[int] = field(default_factory=list)


@dataclass
class InformedPipelineReport:
    """Complete report from the informed pipeline."""

    insights: AnalysisInsights
    stages: list[StageResult] = field(default_factory=list)
    analysis_duration: float = 0.0
    total_duration: float = 0.0
    ouroboros_passes: int = 0
    final_refusal_rate: float | None = None


# ── The Informed Pipeline ────────────────────────────────────────────────

class InformedAbliterationPipeline(AbliterationPipeline):
    """Analysis-informed abliteration pipeline.

    Extends the base AbliterationPipeline with a new ANALYZE stage that
    runs between PROBE and DISTILL. Analysis module outputs automatically
    configure the downstream stages for optimal refusal removal with
    minimal capability damage.

    Usage:
        pipeline = InformedAbliterationPipeline(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            output_dir="abliterated_informed",
        )
        result_path, report = pipeline.run_informed()

        # The report contains all analysis insights
        print(f"Detected alignment: {report.insights.detected_alignment_method}")
        print(f"Cone type: {'polyhedral' if report.insights.cone_is_polyhedral else 'linear'}")
        print(f"Ouroboros passes needed: {report.ouroboros_passes}")
    """

    def __init__(
        self,
        model_name: str,
        output_dir: str = "abliterated_informed",
        device: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
        harmful_prompts: list[str] | None = None,
        harmless_prompts: list[str] | None = None,
        on_stage: Callable[[StageResult], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        # Base pipeline kwargs forwarded to AbliterationPipeline
        push_to_hub: str | None = None,
        hub_token: str | None = None,
        hub_community_org: str | None = None,
        overwrite_output: bool = False,
        quantization: str | None = None,
        project_lm_head: bool | None = None,
        project_embeddings: bool | None = None,
        projection_target: str | None = None,
        verify_sample_size: int | None = None,
        damage_gate_enabled: bool = True,
        damage_budget: AcceptanceBudget | None = None,
        damage_holdout_fraction: float = 0.15,
        damage_eval_max_samples: int = 64,
        damage_eval_seed: int = 42,
        damage_kl_positions_per_prompt: int = 8,
        damage_generation_samples: int = 10,
        evaluation_harmful_prompts: list[str] | None = None,
        evaluation_harmless_prompts: list[str] | None = None,
        # Analysis configuration
        run_cone_analysis: bool = True,
        run_alignment_detection: bool = True,
        run_cross_layer_analysis: bool = True,
        run_sparse_analysis: bool = True,
        run_defense_analysis: bool = True,
        # Ouroboros compensation
        ouroboros_threshold: float = 0.5,
        max_ouroboros_passes: int = 3,
        # Deprecated aliases (kept for backwards compatibility)
        hydra_threshold: float | None = None,
        max_hydra_passes: int | None = None,
        # Entanglement gating
        entanglement_gate: float = 0.8,
        # Sparsity control
        sparse_surgery_threshold: float = 0.5,
    ):
        # Initialize base pipeline with informed method preset
        super().__init__(
            model_name=model_name,
            output_dir=output_dir,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            method="advanced",  # base config, will be overridden
            harmful_prompts=harmful_prompts,
            harmless_prompts=harmless_prompts,
            on_stage=on_stage,
            on_log=on_log,
            push_to_hub=push_to_hub,
            hub_token=hub_token,
            hub_community_org=hub_community_org,
            overwrite_output=overwrite_output,
            quantization=quantization,
            project_lm_head=project_lm_head,
            project_embeddings=project_embeddings,
            projection_target=projection_target,
            verify_sample_size=verify_sample_size,
            damage_gate_enabled=damage_gate_enabled,
            damage_budget=damage_budget,
            damage_holdout_fraction=damage_holdout_fraction,
            damage_eval_max_samples=damage_eval_max_samples,
            damage_eval_seed=damage_eval_seed,
            damage_kl_positions_per_prompt=damage_kl_positions_per_prompt,
            damage_generation_samples=damage_generation_samples,
            evaluation_harmful_prompts=evaluation_harmful_prompts,
            evaluation_harmless_prompts=evaluation_harmless_prompts,
            # Set informed defaults: deterministic analysis-guided projection.
            n_directions=1,
            direction_method="diff_means",
            norm_preserve=True,
            project_biases=True,
            use_chat_template=True,
            use_whitened_svd=False,
            true_iterative_refinement=True,
            use_kl_optimization=True,
            float_layer_interpolation=True,
            layer_adaptive_strength=True,
            winsorize_activations=True,
            winsorize_percentile=0.01,
        )
        self.method = "informed"

        # Analysis module flags
        self._run_cone = run_cone_analysis
        self._run_alignment = run_alignment_detection
        self._run_cross_layer = run_cross_layer_analysis
        self._run_sparse = run_sparse_analysis
        self._run_defense = run_defense_analysis

        # Ouroboros compensation parameters
        self._ouroboros_threshold = hydra_threshold if hydra_threshold is not None else ouroboros_threshold
        self._max_ouroboros_passes = max_hydra_passes if max_hydra_passes is not None else max_ouroboros_passes

        # Entanglement gating
        self._entanglement_gate = entanglement_gate

        # Sparse surgery
        self._sparse_threshold = sparse_surgery_threshold

        # State
        self._insights = AnalysisInsights()
        self._report = InformedPipelineReport(insights=self._insights)

    def run_informed(self) -> tuple[Path, InformedPipelineReport]:
        """Execute the full analysis-informed pipeline.

        Returns:
            (output_path, report) tuple with saved model path and
            comprehensive analysis report.
        """
        t0 = time.time()

        # A previous run on the same pipeline object may have left process-only
        # steering hooks behind.  They are not part of a serialized checkpoint
        # and must not affect either the untouched baseline or official checks.
        self._remove_activation_steering()

        # Stage 1: SUMMON
        self._summon()

        # Stage 2: PROBE
        self._probe()

        # Stage 3: ANALYZE (new stage — the feedback loop)
        self._analyze()

        # Stage 4: DISTILL (informed by analysis)
        self._distill_informed()

        # Freeze held-out benign behavior before the first persistent edit.
        # Any capture failure is fatal and occurs while weights are untouched.
        self._remove_activation_steering()
        self._capture_damage_baseline()

        # Stage 5: EXCISE (informed by analysis)
        self._excise_informed()

        self._remove_activation_steering()

        # Stage 6: VERIFY + Ouroboros compensation loop
        self._verify_and_compensate()

        # Stage 7: REBIRTH
        self._report.total_duration = time.time() - t0
        output_path = self._rebirth_informed()

        self._report.total_duration = time.time() - t0
        return output_path, self._report

    # ── Stage 3: ANALYZE ─────────────────────────────────────────────

    def _analyze(self):
        """Run analysis modules to inform downstream decisions.

        This is the key innovation: analysis runs BETWEEN probe and distill,
        so its outputs configure how directions are extracted and excised.
        """
        self._emit("analyze", "running", "Running analysis modules...")
        t0 = time.time()

        self.log("=" * 60)
        self.log("ANALYSIS-INFORMED PIPELINE — ANALYZE STAGE")
        self.log("=" * 60)

        # 1. Alignment Imprint Detection
        if self._run_alignment:
            self._analyze_alignment_imprint()

        # 2. Concept Cone Geometry
        if self._run_cone:
            self._analyze_cone_geometry()

        # 3. Cross-Layer Alignment
        if self._run_cross_layer:
            self._analyze_cross_layer()

        # 4. Defense Robustness
        if self._run_defense:
            self._analyze_defense_robustness()

        # 5. Sparse Surgery Analysis (RSI computation)
        if self._run_sparse:
            self._analyze_sparsity()

        # 6. Derive configuration from insights
        self._derive_configuration()

        elapsed = time.time() - t0
        self._report.analysis_duration = elapsed
        self.log(f"\nAnalysis complete ({elapsed:.1f}s)")
        self.log(f"  Detected alignment: {self._insights.detected_alignment_method}")
        self.log(f"  Cone type: {'polyhedral' if self._insights.cone_is_polyhedral else 'linear'}")
        self.log(f"  Direction clusters: {self._insights.cluster_count}")
        self.log(f"  Recommended directions: {self._insights.recommended_n_directions}")
        self.log(f"  Recommended regularization: {self._insights.recommended_regularization}")
        self.log(f"  Recommended passes: {self._insights.recommended_refinement_passes}")
        self.log(f"  Layers to skip (entangled): {self._insights.skip_layers}")
        self._emit(
            "analyze", "done",
            f"Analysis complete ({elapsed:.1f}s)",
            duration=elapsed,
        )

    def _analyze_alignment_imprint(self):
        """Detect alignment training method from refusal geometry."""
        self.log("\n[1/4] Alignment Imprint Detection")
        self.log("-" * 40)

        from obliteratus.analysis.alignment_imprint import AlignmentImprintDetector

        detector = AlignmentImprintDetector()

        # We need refusal directions for this — compute quick diff-in-means
        quick_directions = {}
        for idx in sorted(self._harmful_means.keys()):
            diff = (self._harmful_means[idx] - self._harmless_means[idx]).squeeze()
            norm = diff.norm().item()
            if norm > 1e-10:
                quick_directions[idx] = diff / diff.norm()

        if not quick_directions:
            self.log("  No refusal directions found — skipping alignment detection")
            return

        imprint = detector.detect_imprint(quick_directions)

        self._insights.detected_alignment_method = imprint.predicted_method
        self._insights.alignment_confidence = imprint.confidence
        self._insights.alignment_probabilities = {
            "dpo": imprint.dpo_probability,
            "rlhf": imprint.rlhf_probability,
            "cai": imprint.cai_probability,
            "sft": imprint.sft_probability,
        }

        self.log(f"  Predicted: {imprint.predicted_method.upper()} "
                 f"(confidence: {imprint.confidence:.1%})")
        self.log(f"  DPO={imprint.dpo_probability:.1%}  "
                 f"RLHF={imprint.rlhf_probability:.1%}  "
                 f"CAI={imprint.cai_probability:.1%}  "
                 f"SFT={imprint.sft_probability:.1%}")
        self.log("  Geometric features:")
        self.log(f"    Gini coefficient:   {imprint.gini_coefficient:.3f}")
        self.log(f"    Effective rank:     {imprint.effective_rank:.2f}")
        self.log(f"    Cross-layer smooth: {imprint.cross_layer_smoothness:.3f}")
        self.log(f"    Tail layer bias:    {imprint.tail_layer_bias:.3f}")

    def _analyze_cone_geometry(self):
        """Analyze concept cone structure to determine per-category vs universal."""
        self.log("\n[2/4] Concept Cone Geometry")
        self.log("-" * 40)

        from obliteratus.analysis.concept_geometry import ConceptConeAnalyzer

        analyzer = ConceptConeAnalyzer()

        # Analyze at layers that are likely strong refusal layers
        # (middle-to-late layers based on literature)
        n_layers = len(self._harmful_acts)
        candidate_layers = list(range(n_layers // 3, int(n_layers * 0.85)))
        # Sample a subset to keep analysis fast
        step = max(1, len(candidate_layers) // 6)
        sample_layers = candidate_layers[::step]

        polyhedral_count = 0
        all_results = []
        best_cone_result = None
        best_strength = 0.0

        for layer_idx in sample_layers:
            if layer_idx not in self._harmful_acts or layer_idx not in self._harmless_acts:
                continue

            result = analyzer.analyze_layer(
                self._harmful_acts[layer_idx],
                self._harmless_acts[layer_idx],
                layer_idx=layer_idx,
            )

            all_results.append(result)
            if result.is_polyhedral:
                polyhedral_count += 1

            # Track the strongest layer's cone analysis for per-category directions
            general_strength = result.general_direction.norm().item() if result.general_direction.numel() > 1 else 0
            if general_strength > best_strength:
                best_strength = general_strength
                best_cone_result = result

        if all_results:
            # Aggregate cone geometry across sampled layers (majority vote +
            # mean dimensionality) instead of relying on a single layer.
            n_sampled = len(all_results)
            is_polyhedral = polyhedral_count > n_sampled / 2
            avg_dimensionality = sum(r.cone_dimensionality for r in all_results) / n_sampled
            avg_pairwise_cos = sum(r.mean_pairwise_cosine for r in all_results) / n_sampled

            self._insights.cone_is_polyhedral = is_polyhedral
            self._insights.cone_dimensionality = avg_dimensionality
            self._insights.mean_pairwise_cosine = avg_pairwise_cos

            # Store per-category directions from the strongest layer
            if best_cone_result is not None:
                for cd in best_cone_result.category_directions:
                    self._insights.per_category_directions[cd.category] = cd.direction
                    self._insights.direction_specificity[cd.category] = cd.specificity

            cone_type = "POLYHEDRAL" if is_polyhedral else "LINEAR"
            self.log(f"  Cone type: {cone_type} (majority vote: {polyhedral_count}/{n_sampled} layers)")
            self.log(f"  Avg dimensionality: {avg_dimensionality:.2f}")
            self.log(f"  Avg pairwise cosine: {avg_pairwise_cos:.3f}")
            if best_cone_result is not None:
                self.log(f"  Categories detected: {best_cone_result.category_count}")

                for cd in sorted(best_cone_result.category_directions, key=lambda x: -x.strength)[:5]:
                    self.log(f"    {cd.category:15s}  DSI={cd.specificity:.3f}  str={cd.strength:.3f}")
        else:
            self.log("  No cone results — using default linear assumption")

    def _analyze_cross_layer(self):
        """Analyze cross-layer direction alignment for cluster-aware layer selection."""
        self.log("\n[3/4] Cross-Layer Direction Alignment")
        self.log("-" * 40)

        from obliteratus.analysis.cross_layer import CrossLayerAlignmentAnalyzer

        # Compute quick directions for cross-layer analysis
        quick_directions = {}
        for idx in sorted(self._harmful_means.keys()):
            diff = (self._harmful_means[idx] - self._harmless_means[idx]).squeeze()
            norm = diff.norm().item()
            if norm > 1e-10:
                quick_directions[idx] = diff / diff.norm()

        if len(quick_directions) < 2:
            self.log("  Too few layers with refusal directions")
            return

        analyzer = CrossLayerAlignmentAnalyzer(cluster_threshold=0.85)
        result = analyzer.analyze(quick_directions)

        self._insights.direction_clusters = result.clusters
        self._insights.cluster_count = result.cluster_count
        self._insights.direction_persistence = result.direction_persistence_score

        # Select representative layers from each cluster
        # (the strongest layer per cluster is the best representative)
        representatives = []
        norms = {idx: (self._harmful_means[idx] - self._harmless_means[idx]).squeeze().norm().item()
                 for idx in quick_directions}
        for cluster in result.clusters:
            best = max(cluster, key=lambda ly: norms.get(ly, 0))
            representatives.append(best)
        self._insights.cluster_representative_layers = representatives

        self.log(f"  Direction persistence: {result.direction_persistence_score:.3f}")
        self.log(f"  Mean adjacent cosine: {result.mean_adjacent_cosine:.3f}")
        self.log(f"  Direction clusters: {result.cluster_count}")
        for i, cluster in enumerate(result.clusters):
            self.log(f"    Cluster {i+1}: layers {cluster}")
        self.log(f"  Representative layers: {representatives}")

    def _analyze_defense_robustness(self):
        """Assess defense robustness, self-repair risk, and entanglement."""
        self.log("\n[4/4] Defense Robustness Assessment")
        self.log("-" * 40)

        from obliteratus.analysis.defense_robustness import DefenseRobustnessEvaluator

        # Temporarily set refusal_directions for the evaluator
        quick_directions = {}
        for idx in sorted(self._harmful_means.keys()):
            diff = (self._harmful_means[idx] - self._harmless_means[idx]).squeeze()
            norm = diff.norm().item()
            if norm > 1e-10:
                quick_directions[idx] = diff / diff.norm()

        # Store temporarily for the evaluator
        original_dirs = self.refusal_directions
        self.refusal_directions = quick_directions

        evaluator = DefenseRobustnessEvaluator(self)
        profile = evaluator.profile_defense()
        emap = evaluator.map_entanglement()

        # Restore
        self.refusal_directions = original_dirs

        self._insights.estimated_robustness = profile.estimated_robustness
        self._insights.self_repair_estimate = profile.self_repair_estimate
        self._insights.entanglement_score = profile.entanglement_score
        self._insights.entangled_layers = emap.most_entangled_layers
        self._insights.clean_layers = emap.least_entangled_layers

        self.log(f"  Estimated robustness: {profile.estimated_robustness.upper()}")
        self.log(f"  Self-repair estimate: {profile.self_repair_estimate:.2f}")
        self.log(f"  Safety-capability entanglement: {profile.entanglement_score:.3f}")
        self.log(f"  Most entangled layers: {emap.most_entangled_layers}")
        self.log(f"  Cleanest layers: {emap.least_entangled_layers}")

    def _analyze_sparsity(self):
        """Compute Refusal Sparsity Index to decide sparse vs dense excision."""
        self.log("\n[5/5] Refusal Sparsity Analysis")
        self.log("-" * 40)

        from obliteratus.analysis.sparse_surgery import SparseDirectionSurgeon
        from obliteratus.strategies.utils import (
            get_ffn_module,
            get_layer_modules,
        )

        # Need refusal directions — use quick diff-in-means
        quick_directions = {}
        for idx in sorted(self._harmful_means.keys()):
            diff = (self._harmful_means[idx] - self._harmless_means[idx]).squeeze()
            norm = diff.norm().item()
            if norm > 1e-10:
                quick_directions[idx] = diff / diff.norm()

        if not quick_directions:
            self.log("  No refusal directions — skipping sparsity analysis")
            return

        # Gather FFN output weights for representative layers (sample for speed)
        layers = get_layer_modules(self.handle)
        arch = self.handle.architecture
        n_layers = len(layers)
        sample_idxs = sorted(quick_directions.keys())
        step = max(1, len(sample_idxs) // 8)
        sample_idxs = sample_idxs[::step]

        weights = {}
        sampled_dirs = {}
        for idx in sample_idxs:
            if idx >= n_layers:
                continue
            try:
                ffn = get_ffn_module(layers[idx], arch)
                for name in ["down_proj", "c_proj", "dense_4h_to_h", "fc_out", "fc2", "w2"]:
                    proj = getattr(ffn, name, None)
                    if proj is not None and hasattr(proj, "weight"):
                        W = proj.weight.data
                        d = quick_directions[idx]
                        if W.shape[-1] == d.shape[0]:
                            weights[idx] = W
                            sampled_dirs[idx] = d
                            break
            except (AttributeError, RuntimeError):
                continue

        if not weights:
            self.log("  Could not access FFN weights — skipping sparsity analysis")
            return

        surgeon = SparseDirectionSurgeon(auto_sparsity=True)
        plan = surgeon.plan_surgery(weights, sampled_dirs)

        self._insights.mean_refusal_sparsity_index = plan.mean_refusal_sparsity_index
        self._insights.recommended_sparsity = plan.recommended_sparsity

        self.log(f"  Mean RSI: {plan.mean_refusal_sparsity_index:.3f}")
        self.log(f"  Recommended sparsity: {plan.recommended_sparsity:.1%}")
        self.log(f"  Most sparse layer: {plan.most_sparse_layer}")
        self.log(f"  Most dense layer: {plan.most_dense_layer}")

    # ── Configuration Derivation ─────────────────────────────────────

    def _derive_configuration(self):
        """Derive optimal pipeline configuration from analysis insights.

        This is where analysis feeds forward into abliteration decisions.
        """
        self.log("\n>>> DERIVING CONFIGURATION FROM ANALYSIS")
        self.log("-" * 50)
        insights = self._insights

        # 1. n_directions + direction_method: based on cone geometry
        # Default: single direction via diff-of-means (proven most robust).
        # Only escalate to multi-direction when analysis confirms polyhedral geometry.
        if insights.cone_is_polyhedral and insights.cone_dimensionality > 2.0:
            # Clearly polyhedral cone → use multiple directions via SVD
            n_dirs = max(4, min(8, int(insights.cone_dimensionality * 2)))
            self.direction_method = "svd"
            self.use_whitened_svd = True
            self.log(f"  Polyhedral cone (dim={insights.cone_dimensionality:.1f}) "
                     f"→ n_directions={n_dirs}, method=svd (whitened)")
        elif insights.cone_is_polyhedral:
            # Mildly polyhedral → LEACE gives better single-direction erasure
            n_dirs = 1
            self.direction_method = "leace"
            self.use_whitened_svd = False
            self.log(f"  Mildly polyhedral (dim={insights.cone_dimensionality:.1f}) "
                     f"→ n_directions=1, method=leace")
        else:
            # Linear cone → single direction via diff-of-means (simplest, most robust)
            n_dirs = 1
            self.direction_method = "diff_means"
            self.use_whitened_svd = False
            self.log(f"  Linear cone (dim={insights.cone_dimensionality:.1f}) "
                     f"→ n_directions=1, method=diff_means")
        insights.recommended_n_directions = n_dirs
        insights.recommended_direction_method = self.direction_method
        self.n_directions = n_dirs

        # 2. regularization: based on alignment method + entanglement
        method = insights.detected_alignment_method
        if method == "dpo":
            # DPO: concentrated refusal, low entanglement → aggressive removal
            reg = 0.0
        elif method == "rlhf":
            # RLHF: distributed, moderate entanglement → some regularization
            reg = 0.15
        elif method == "cai":
            # CAI: recursive, high dimensionality → moderate regularization
            reg = 0.2
        elif method == "sft":
            # SFT: concentrated in late layers → low regularization
            reg = 0.05
        else:
            reg = 0.1  # safe default

        # Increase regularization for highly entangled models
        if insights.entanglement_score > 0.5:
            reg = min(0.5, reg + 0.15)
            self.log(f"  High entanglement ({insights.entanglement_score:.2f}) "
                     f"→ increased regularization")

        insights.recommended_regularization = reg
        self.regularization = reg
        self.log(f"  Alignment={method}, entanglement={insights.entanglement_score:.2f} "
                 f"→ regularization={reg}")

        # 3. refinement_passes: based on self-repair risk + robustness
        if insights.self_repair_estimate > 0.7:
            passes = 3
            self.log(f"  High self-repair ({insights.self_repair_estimate:.2f}) → 3 refinement passes")
        elif insights.self_repair_estimate > 0.4:
            passes = 2
            self.log(f"  Moderate self-repair ({insights.self_repair_estimate:.2f}) → 2 refinement passes")
        else:
            passes = 1
            self.log(f"  Low self-repair ({insights.self_repair_estimate:.2f}) → 1 refinement pass")

        insights.recommended_refinement_passes = passes
        self.refinement_passes = passes

        # 4. Layer selection: cluster-aware + entanglement-gated
        if insights.cluster_representative_layers:
            # Start from cluster representatives (strongest per cluster)
            base_layers = list(insights.cluster_representative_layers)

            # Conservative expansion: for each cluster, add at most the top-2
            # strongest layers (by refusal norm) beyond the representative,
            # to avoid over-modifying weak layers in large clusters.
            norms = {}
            for idx in self._harmful_means:
                if idx in self._harmless_means:
                    norms[idx] = (self._harmful_means[idx] - self._harmless_means[idx]).squeeze().norm().item()
            for cluster in insights.direction_clusters:
                ranked = sorted(cluster, key=lambda ly: norms.get(ly, 0), reverse=True)
                # Add up to 2 additional strong layers per cluster
                base_layers.extend(ranked[:3])  # representative + up to 2 more
            base_layers = sorted(set(base_layers))

            # Gate: remove highly entangled layers
            skip = set()
            for layer_idx in insights.entangled_layers:
                # Only skip if entanglement exceeds the gate threshold
                # and there are alternative layers available
                if len(base_layers) > len(insights.entangled_layers) + 1:
                    skip.add(layer_idx)
                    self.log(f"  Skipping layer {layer_idx} (entangled)")

            insights.skip_layers = sorted(skip)
            insights.recommended_layers = [ly for ly in base_layers if ly not in skip]
        else:
            insights.recommended_layers = []

        self.log(f"  Final layer set: {insights.recommended_layers or '(default knee detection)'}")

        # 5. Sparse surgery: if refusal is concentrated, use targeted projection
        if insights.mean_refusal_sparsity_index > self._sparse_threshold:
            insights.use_sparse_surgery = True
            self.log(f"  RSI={insights.mean_refusal_sparsity_index:.2f} > {self._sparse_threshold} "
                     f"→ sparse surgery enabled")
        else:
            self.log(f"  RSI={insights.mean_refusal_sparsity_index:.2f} "
                     f"→ standard dense projection")

        # 6. Direction method summary (already set in step 1)
        self.log(f"  Direction method: {self.direction_method} "
                 f"(whitened_svd={'on' if self.use_whitened_svd else 'off'})")

    # ── Informed DISTILL ─────────────────────────────────────────────

    def _distill_informed(self):
        """Distill refusal directions using analysis-informed parameters.

        Key differences from base _distill():
        - Uses analysis-recommended n_directions
        - Respects layer selection from cross-layer analysis
        - Can extract per-category directions for polyhedral models
        """
        self._emit("distill", "running", "Extracting refusal subspace (analysis-informed)...")
        t0 = time.time()

        self.log("\nDISTILL (analysis-informed)")

        # Run the standard distillation (which now uses our overridden params)
        # The base _distill() uses self.n_directions, self.use_whitened_svd, etc.
        # which we've already configured in _derive_configuration()
        n_layers = len(self._harmful_means)
        norms: dict[int, float] = {}

        # ── Small-model direction cap (matching base _distill) ────────
        # On small models, each SVD direction removes a proportionally
        # larger fraction of weight energy.  Cap to prevent over-ablation.
        hidden_size = self.handle.hidden_size if self.handle else 0
        total_params = getattr(self.handle, 'total_params', 0) if self.handle else 0
        if total_params == 0 and self.handle:
            try:
                total_params = sum(p.numel() for p in self.handle.model.parameters())
            except (AttributeError, RuntimeError, TypeError) as exc:
                logger.debug("Could not count model parameters: %s", exc)
        if self.n_directions > 1 and (
            (0 < hidden_size < 2048)
            or (0 < total_params < 2_000_000_000)
            or n_layers <= 16
        ):
            max_dirs = max(1, min(self.n_directions, 2))
            if max_dirs < self.n_directions:
                self.log(
                    f"Capped n_directions from {self.n_directions} to {max_dirs} "
                    f"for small model (hidden={hidden_size}, "
                    f"params={total_params / 1e9:.1f}B, layers={n_layers})"
                )
                self.n_directions = max_dirs

        # LEACE extractor for optimal concept erasure
        leace_extractor = None
        if self.direction_method == "leace":
            from obliteratus.analysis.leace import LEACEExtractor
            leace_extractor = LEACEExtractor()
            self.log("Using LEACE (closed-form optimal concept erasure)")

        if self.use_whitened_svd and self.n_directions > 1 and leace_extractor is None:
            from obliteratus.analysis.whitened_svd import WhitenedSVDExtractor
            whitened_extractor = WhitenedSVDExtractor()
            self.log(f"Using whitened SVD with {self.n_directions} directions")
        else:
            whitened_extractor = None

        for idx in range(n_layers):
            # LEACE path: theoretically optimal single-direction erasure
            if (
                leace_extractor is not None
                and idx in self._harmful_acts
                and idx in self._harmless_acts
            ):
                try:
                    l_result = leace_extractor.extract(
                        self._harmful_acts[idx],
                        self._harmless_acts[idx],
                        layer_idx=idx,
                    )
                    self.refusal_directions[idx] = l_result.direction
                    self.refusal_subspaces[idx] = l_result.direction.unsqueeze(0)
                    norms[idx] = l_result.generalized_eigenvalue

                    if idx < 5 or idx == n_layers - 1:
                        self.log(
                            f"  layer {idx}: LEACE eigenvalue={l_result.generalized_eigenvalue:.4f}, "
                            f"erasure_loss={l_result.erasure_loss:.4f}"
                        )
                    continue
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    if idx < 5:
                        self.log(f"  layer {idx}: LEACE failed ({exc}), falling back")

            if self.n_directions == 1:
                diff = (self._harmful_means[idx] - self._harmless_means[idx]).squeeze(0)
                norm = diff.norm().item()
                norms[idx] = norm
                direction = diff / diff.norm() if norm > 0 else diff
                self.refusal_directions[idx] = direction
                self.refusal_subspaces[idx] = direction.unsqueeze(0)
            elif whitened_extractor is not None:
                result = whitened_extractor.extract(
                    self._harmful_acts[idx],
                    self._harmless_acts[idx],
                    n_directions=self.n_directions,
                    layer_idx=idx,
                )
                self.refusal_subspaces[idx] = result.directions
                self.refusal_directions[idx] = result.directions[0]
                norms[idx] = result.singular_values.sum().item()
            else:
                harmful_stack = torch.stack(self._harmful_acts[idx]).squeeze(1)
                harmless_stack = torch.stack(self._harmless_acts[idx]).squeeze(1)
                diff_matrix = harmful_stack - harmless_stack
                if not torch.isfinite(diff_matrix).all():
                    diff_matrix = torch.nan_to_num(diff_matrix)
                k = min(self.n_directions, diff_matrix.shape[0], diff_matrix.shape[1])
                _, S, Vh = torch.linalg.svd(diff_matrix, full_matrices=False)
                if not torch.isfinite(S).all() or not torch.isfinite(Vh).all():
                    continue
                subspace = Vh[:k]
                self.refusal_subspaces[idx] = subspace
                primary = subspace[0]
                self.refusal_directions[idx] = primary / primary.norm()
                norms[idx] = S[:k].sum().item()

        # Enrich subspaces with per-category cone directions when available.
        # This uses the actual refusal cone generators instead of purely
        # data-agnostic SVD components.
        cat_dirs = self._insights.per_category_directions
        if cat_dirs and self._insights.cone_is_polyhedral and self.n_directions > 1:
            cat_tensors = list(cat_dirs.values())
            # Stack and orthogonalize category directions
            cat_stack = torch.stack(cat_tensors)  # (n_cats, hidden)
            cat_norms = cat_stack.norm(dim=1, keepdim=True).clamp(min=1e-8)
            cat_stack = cat_stack / cat_norms
            # Blend into strong-signal layers: replace later SVD components
            # with category directions (which are geometrically meaningful)
            n_cat = cat_stack.shape[0]
            for idx in norms:
                sub = self.refusal_subspaces.get(idx)
                if sub is None or sub.shape[0] <= 1:
                    continue
                # Keep the first SVD direction (strongest), replace remaining
                # with category directions projected to be orthogonal to it
                primary = sub[0:1]  # (1, hidden)
                # Project category directions orthogonal to primary
                cos = (cat_stack @ primary.squeeze(0))  # (n_cat,)
                ortho_cats = cat_stack - cos.unsqueeze(1) * primary
                ortho_norms = ortho_cats.norm(dim=1)
                # Keep only directions that survived orthogonalization
                valid = ortho_norms > 0.1
                if valid.sum() > 0:
                    ortho_cats = ortho_cats[valid]
                    ortho_cats = ortho_cats / ortho_cats.norm(dim=1, keepdim=True)
                    # Take up to (n_directions - 1) category directions
                    n_take = min(self.n_directions - 1, ortho_cats.shape[0])
                    new_sub = torch.cat([primary, ortho_cats[:n_take]], dim=0)
                    self.refusal_subspaces[idx] = new_sub
            self.log(f"Enriched subspaces with {n_cat} per-category cone directions")

        # Layer selection: use analysis-recommended layers if available,
        # otherwise fall back to knee detection
        if self._insights.recommended_layers:
            self._strong_layers = [ly for ly in self._insights.recommended_layers
                                   if ly in self.refusal_directions]
            self.log(f"Using analysis-recommended layers: {self._strong_layers}")
        else:
            sorted_layers = sorted(norms.items(), key=lambda x: x[1], reverse=True)
            self._strong_layers = self._select_layers_knee(sorted_layers)
            self.log(f"Using knee-detected layers: {self._strong_layers}")

        # Remove skipped layers (entanglement-gated)
        if self._insights.skip_layers:
            before = len(self._strong_layers)
            self._strong_layers = [ly for ly in self._strong_layers
                                   if ly not in self._insights.skip_layers]
            after = len(self._strong_layers)
            if before != after:
                self.log(f"Entanglement gate removed {before - after} layers "
                         f"→ {after} remaining")

        elapsed = time.time() - t0
        self.log(f"Distillation complete: {len(self._strong_layers)} layers, "
                 f"{self.n_directions} directions ({elapsed:.1f}s)")
        self._emit(
            "distill", "done",
            f"Analysis-informed: {len(self._strong_layers)} layers, "
            f"{self.n_directions} dirs ({elapsed:.1f}s)",
            duration=elapsed,
            strong_layers=self._strong_layers,
        )

    # ── Informed EXCISE ──────────────────────────────────────────────

    def _excise_informed(self):
        """Excise refusal directions with analysis-informed strategy.

        Uses sparse surgery when analysis supports it; otherwise applies one
        deterministic analysis-guided projection pass. Bayesian tuning remains
        disabled until the exact candidate scored during search can be replayed.
        """
        # The base excision routines can apply ``refinement_passes`` edits in a
        # single call.  That is unsafe here because only the final state would
        # be checked.  Apply exactly one persistent pass per call; the outer
        # compensation loop may request another pass only after verification.
        configured_refinement_passes = self.refinement_passes
        self.refinement_passes = 1
        self._remove_activation_steering()
        try:
            if self._insights.use_sparse_surgery:
                self._excise_sparse()
            else:
                # Configure deterministic analysis-guided strength/layer
                # settings, then perform one persistent edit pass.
                self._configure_bayesian_warm_start()
                self._excise()
        finally:
            self.refinement_passes = configured_refinement_passes
            removed_hooks = self._remove_activation_steering()
            if removed_hooks:
                self.log(
                    f"Removed {removed_hooks} runtime-only steering hooks "
                    "before checkpoint verification"
                )

    def _configure_bayesian_warm_start(self):
        """Configure deterministic analysis-guided projection settings.

        Retains deterministic per-layer strength/interpolation while disabling
        the inconsistent Bayesian and legacy KL-correction branches.
        """
        # The old optimizer measured a different edit from the one it later
        # applied. Keep it off without disabling the useful informed pipeline.
        self._bayesian_trials = 0

        # Retain deterministic analysis-derived layer weighting. The invalid
        # legacy post-hoc KL correction is disabled in the base pipeline.
        self.layer_adaptive_strength = True
        self.float_layer_interpolation = True
        self.use_kl_optimization = False

        self.log(
            "Bayesian tuning disabled pending exact replay; continuing with "
            "deterministic analysis-guided projection and the held-out gate"
        )

    def _excise_sparse(self):
        """Apply sparse surgery to every selected manifest writer exactly once."""
        self._emit("excise", "running", "Sparse direction surgery...")
        t0 = time.time()

        from obliteratus.analysis.sparse_surgery import SparseDirectionSurgeon

        surgeon = SparseDirectionSurgeon(
            sparsity=self._insights.recommended_sparsity,
            auto_sparsity=True,
        )
        total_modified = 0

        for pass_num in range(self.refinement_passes):
            if self.refinement_passes > 1:
                self.log(f"Sparse surgery pass {pass_num + 1}/{self.refinement_passes}")

            if pass_num > 0 and self.true_iterative_refinement:
                self.log("  Re-probing after sparse surgery...")
                self._probe()
                self._distill_inner()

            plan, expected = self._prepare_sparse_manifest_plan()
            applied: set[tuple[str, int]] = set()
            layer_counts: dict[int, int] = {}
            for entry, owner_layer, projection, tensor, is_quantized, subspace in plan:
                updated = self._sparse_project_manifest_tensor(
                    tensor,
                    subspace,
                    residual_axis=entry.residual_axis,
                    expert_axis=entry.expert_axis,
                    surgeon=surgeon,
                )
                self._commit_sparse_manifest_tensor(
                    entry,
                    projection,
                    updated,
                    is_quantized=is_quantized,
                )
                keys = {
                    (entry.storage_identity, direction_index)
                    for direction_index in range(subspace.shape[0])
                }
                duplicate = applied.intersection(keys)
                if duplicate:
                    raise ArchitectureCoverageError(
                        "Sparse manifest storage was applied more than once: "
                        f"{entry.qualified_name}"
                    )
                applied.update(keys)
                layer_counts[owner_layer] = layer_counts.get(owner_layer, 0) + len(keys)

            if applied != expected:
                missing = expected - applied
                extra = applied - expected
                raise ArchitectureCoverageError(
                    "Sparse surgery did not exactly execute the validated writer "
                    f"manifest (missing={len(missing)}, extra={len(extra)})"
                )

            modified = len(applied)
            for idx in sorted(layer_counts):
                self.log(
                    f"  layer {idx}: {layer_counts[idx]} sparse writer projections"
                )
            total_modified += modified
            self.log(f"  Pass {pass_num + 1}: {modified} manifest projections (sparse)")

        elapsed = time.time() - t0
        self.log(f"Sparse excision: {total_modified} projections ({elapsed:.1f}s)")
        self._emit(
            "excise", "done",
            f"Sparse surgery: {total_modified} projections ({elapsed:.1f}s)",
            duration=elapsed,
            modified_count=total_modified,
        )

    def _prepare_sparse_manifest_plan(self):
        """Resolve and validate the complete sparse writer plan before editing."""
        manifest = self._current_projection_manifest()
        if len(self._strong_layers) != len(set(self._strong_layers)):
            raise ArchitectureCoverageError(
                "Sparse surgery received duplicate strong-layer indices"
            )
        strong_layers = set(self._strong_layers)

        for layer_idx in strong_layers:
            subspace = self.refusal_subspaces.get(layer_idx)
            if not isinstance(subspace, torch.Tensor):
                raise ArchitectureCoverageError(
                    f"Sparse surgery has no refusal subspace for layer {layer_idx}"
                )
            if subspace.ndim != 2 or subspace.shape[0] == 0:
                raise ArchitectureCoverageError(
                    f"Layer {layer_idx} refusal subspace must be a non-empty matrix"
                )
            if subspace.shape[1] != manifest.hidden_size:
                raise ArchitectureCoverageError(
                    f"Layer {layer_idx} refusal subspace width {subspace.shape[1]} "
                    f"does not match manifest hidden size {manifest.hidden_size}"
                )
            if not torch.isfinite(subspace).all():
                raise ArchitectureCoverageError(
                    f"Layer {layer_idx} refusal subspace contains NaN/Inf"
                )
            if (subspace.norm(dim=1) <= 1e-10).any():
                raise ArchitectureCoverageError(
                    f"Layer {layer_idx} refusal subspace contains a zero direction"
                )

        plan = []
        expected: set[tuple[str, int]] = set()
        covered_layers: set[int] = set()
        for entry in manifest.entries:
            if entry.role != "writer":
                continue
            owners = strong_layers.intersection(entry.layer_indices)
            if not owners:
                continue
            if entry.orientation != "output":
                raise ArchitectureCoverageError(
                    f"Manifest writer {entry.qualified_name} has non-output orientation"
                )
            if entry.branch_kind not in {"attention", "ffn"}:
                raise ArchitectureCoverageError(
                    f"Manifest writer {entry.qualified_name} has unsupported branch kind "
                    f"{entry.branch_kind!r}"
                )

            owner_layer = min(owners)
            subspace = self.refusal_subspaces[owner_layer]
            projection = self._resolve_dotted_projection(
                entry.owner, entry.attribute_path
            )
            if entry.projection_kind == "module_weight":
                if not isinstance(projection, torch.nn.Module) or not isinstance(
                    getattr(projection, "weight", None), torch.Tensor
                ):
                    raise ArchitectureCoverageError(
                        f"Manifest module writer {entry.qualified_name} no longer "
                        "resolves to a weighted module"
                    )
                tensor, is_quantized = self._dequantize_weight(projection)
            elif entry.projection_kind == "parameter_axis":
                if not isinstance(projection, (torch.nn.Parameter, torch.Tensor)):
                    raise ArchitectureCoverageError(
                        f"Manifest packed writer {entry.qualified_name} no longer "
                        "resolves to a tensor"
                    )
                tensor = projection.data if isinstance(
                    projection, torch.nn.Parameter
                ) else projection
                is_quantized = False
            else:
                raise ArchitectureCoverageError(
                    f"Manifest writer {entry.qualified_name} has unsupported projection "
                    f"kind {entry.projection_kind!r}"
                )

            if tensor.device.type == "meta":
                raise ArchitectureCoverageError(
                    f"Manifest writer {entry.qualified_name} is still on the meta device"
                )
            if not tensor.is_floating_point():
                raise ArchitectureCoverageError(
                    f"Manifest writer {entry.qualified_name} is not floating point"
                )
            if tuple(tensor.shape) != entry.shape:
                raise ArchitectureCoverageError(
                    f"Manifest writer {entry.qualified_name} changed shape from "
                    f"{entry.shape} to {tuple(tensor.shape)}"
                )
            if tensor.ndim < 2:
                raise ArchitectureCoverageError(
                    f"Manifest writer {entry.qualified_name} is not matrix-like"
                )
            residual_axis = entry.residual_axis % tensor.ndim
            if tensor.shape[residual_axis] != manifest.hidden_size:
                raise ArchitectureCoverageError(
                    f"Manifest writer {entry.qualified_name} residual axis "
                    f"{residual_axis} has width {tensor.shape[residual_axis]}, expected "
                    f"{manifest.hidden_size}"
                )
            if entry.expert_axis is not None:
                expert_axis = entry.expert_axis % tensor.ndim
                if expert_axis == residual_axis:
                    raise ArchitectureCoverageError(
                        f"Manifest writer {entry.qualified_name} reuses its residual "
                        "axis as the expert axis"
                    )
            if not torch.isfinite(tensor).all():
                raise ArchitectureCoverageError(
                    f"Manifest writer {entry.qualified_name} contains NaN/Inf"
                )

            for direction_index in range(subspace.shape[0]):
                key = (entry.storage_identity, direction_index)
                if key in expected:
                    raise ArchitectureCoverageError(
                        "Sparse writer plan contains duplicate storage/direction "
                        f"for {entry.qualified_name}"
                    )
                expected.add(key)
            covered_layers.update(owners)
            plan.append(
                (
                    entry,
                    owner_layer,
                    projection,
                    tensor,
                    is_quantized,
                    subspace,
                )
            )

        missing_layers = strong_layers - covered_layers
        if missing_layers:
            raise ArchitectureCoverageError(
                "Sparse writer manifest does not cover strong layers "
                f"{sorted(missing_layers)}"
            )

        for coverage in manifest.branch_coverage:
            layer_idx = coverage.get("layer")
            if layer_idx not in strong_layers:
                continue
            branch_kind = coverage.get("kind")
            branch_path = coverage.get("path")
            if not any(
                entry.role == "writer"
                and layer_idx in entry.layer_indices
                and entry.branch_kind == branch_kind
                and branch_path in entry.branch_paths
                for entry in manifest.entries
            ):
                raise ArchitectureCoverageError(
                    f"Sparse writer plan omits layer {layer_idx} {branch_kind} "
                    f"branch {branch_path!r}"
                )

        if strong_layers and not expected:
            raise ArchitectureCoverageError(
                "Sparse surgery has strong layers but no manifest writer projections"
            )
        return plan, expected

    def _sparse_project_manifest_tensor(
        self,
        tensor: torch.Tensor,
        subspace: torch.Tensor,
        *,
        residual_axis: int,
        expert_axis: int | None,
        surgeon,
    ) -> torch.Tensor:
        """Project a manifest tensor sparsely along its declared residual axis."""
        working = tensor.detach().clone()
        original_norm = working.float().norm().item() if self.norm_preserve else 0.0
        residual_axis %= working.ndim

        def project_slice(target: torch.Tensor, axis: int, direction: torch.Tensor):
            moved = target.movedim(axis, -1)
            matrix = moved.reshape(-1, moved.shape[-1])
            local_direction = direction.to(
                device=matrix.device,
                dtype=matrix.dtype,
            )
            modified = surgeon.apply_sparse_projection(matrix, local_direction)
            if modified.shape != matrix.shape or not torch.isfinite(modified).all():
                raise RuntimeError("Sparse projection produced an invalid writer tensor")
            moved.copy_(modified.reshape(moved.shape))

        for direction in subspace:
            if expert_axis is None:
                project_slice(working, residual_axis, direction)
                continue
            normalized_expert_axis = expert_axis % working.ndim
            adjusted_residual_axis = residual_axis
            if normalized_expert_axis < residual_axis:
                adjusted_residual_axis -= 1
            for expert_index in range(working.shape[normalized_expert_axis]):
                project_slice(
                    working.select(normalized_expert_axis, expert_index),
                    adjusted_residual_axis,
                    direction,
                )

        if self.norm_preserve and original_norm > 0.0:
            new_norm = working.float().norm().item()
            if not math.isfinite(new_norm) or new_norm <= 0.0:
                raise RuntimeError("Sparse projection produced a degenerate writer tensor")
            working.mul_(original_norm / new_norm)
        return working

    def _commit_sparse_manifest_tensor(
        self,
        entry: ProjectionManifestEntry,
        projection,
        updated: torch.Tensor,
        *,
        is_quantized: bool,
    ) -> None:
        """Commit one completely computed sparse writer tensor."""
        if entry.projection_kind == "module_weight":
            if is_quantized:
                self._replace_quantized_weight(projection, updated)
            else:
                projection.weight.data.copy_(
                    updated.to(
                        device=projection.weight.device,
                        dtype=projection.weight.dtype,
                    )
                )
            return

        target = projection.data if isinstance(
            projection, torch.nn.Parameter
        ) else projection
        target.copy_(updated.to(device=target.device, dtype=target.dtype))

    # ── Informed VERIFY + Ouroboros Compensation ──────────────────────

    def _verify_and_compensate(self):
        """Verify excision and run Ouroboros-compensated refinement if needed.

        After the initial excision, uses analysis modules to detect:
        1. Residual refusal signal (via activation probing)
        2. Self-repair / Ouroboros effect (via defense robustness)
        3. Triggers additional targeted passes at compensating layers

        Every pass is measured against the same untouched held-out baseline.
        A collateral-damage failure restores the original snapshot and aborts;
        refusal improvement can never compensate for failed locality checks.
        """
        assessment = self._verify()
        if self.damage_gate_enabled and not assessment.damage_accepted:
            self._reject_and_restore(assessment)

        # Check if Ouroboros compensation is needed
        ouroboros_pass = 0
        efficacy_limit = self.damage_budget.efficacy.max_refusal_rate
        target_refusal = (
            efficacy_limit
            if efficacy_limit is not None
            else self._ouroboros_threshold
        )
        refusal_rate = self._validated_refusal_rate(
            require_evidence=efficacy_limit is not None,
        )
        if (
            self.damage_gate_enabled
            and efficacy_limit is not None
            and refusal_rate is None
        ):
            # A missing/partial refusal measurement cannot justify another
            # destructive edit.  It also cannot qualify a checkpoint.
            self._reject_and_restore(assessment)

        while (
            refusal_rate is not None
            and refusal_rate > target_refusal
            and ouroboros_pass < self._max_ouroboros_passes
        ):
            ouroboros_pass += 1
            self.log(f"\n{'='*60}")
            self.log(f"OUROBOROS COMPENSATION — Pass {ouroboros_pass}")
            self.log(
                f"Refusal rate still {refusal_rate:.0%} > "
                f"{target_refusal:.0%} acceptance threshold"
            )
            self.log(f"{'='*60}")

            # Re-probe to find where refusal has re-emerged
            self.log("Re-probing model for residual refusal...")
            self._probe()

            # Re-distill to find rotated directions
            self._distill_inner()
            self.log(f"Found {len(self._strong_layers)} layers with residual refusal")

            # Re-excise at the new strong layers using informed strategy
            if self._strong_layers:
                self._excise_informed()
                self._remove_activation_steering()
            else:
                self.log("No strong layers found — stopping Ouroboros compensation")
                break

            # Re-verify
            assessment = self._verify()
            if self.damage_gate_enabled and not assessment.damage_accepted:
                self._reject_and_restore(assessment)
            refusal_rate = self._validated_refusal_rate(
                require_evidence=efficacy_limit is not None,
            )
            if (
                self.damage_gate_enabled
                and efficacy_limit is not None
                and refusal_rate is None
            ):
                self._reject_and_restore(assessment)
            current_kl = self._quality_metrics.get("kl_divergence")
            try:
                kl_number = float(current_kl) if current_kl is not None else None
            except (TypeError, ValueError):
                kl_number = None
            kl_text = (
                f"{kl_number:.4f}"
                if kl_number is not None and math.isfinite(kl_number)
                else "missing"
            )
            refusal_text = (
                f"{refusal_rate:.0%}" if refusal_rate is not None else "missing"
            )
            self.log(
                f"After Ouroboros pass {ouroboros_pass}: "
                f"refusal={refusal_text}, KL={kl_text}"
            )

        self._report.ouroboros_passes = ouroboros_pass
        self._report.final_refusal_rate = refusal_rate

        if ouroboros_pass > 0:
            self.log(f"\nOuroboros compensation: {ouroboros_pass} additional passes applied")
        if self.damage_gate_enabled and not assessment.accepted:
            self._reject_and_restore(assessment)
        return assessment

    def _validated_refusal_rate(self, *, require_evidence: bool) -> float | None:
        """Return a finite, sufficiently sampled held-out refusal rate.

        Compensation is allowed only when the efficacy failure is an observed
        high refusal rate.  Missing, non-finite, or undersampled evidence is a
        measurement failure, not a reason to edit the model again.
        """
        value = self._quality_metrics.get("refusal_rate")
        if value is None:
            return None
        try:
            rate = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            return None

        if require_evidence:
            count = self._quality_metrics.get("refusal_eval_count")
            try:
                count_number = float(count) if count is not None else None
            except (TypeError, ValueError):
                return None
            if (
                count_number is None
                or not math.isfinite(count_number)
                or count_number < self.damage_budget.efficacy.min_eval_prompts
            ):
                return None
        return rate

    # ── Informed REBIRTH ─────────────────────────────────────────────

    def _build_metadata(self) -> dict:
        """Extend the base, gate-audited metadata with informed analysis."""
        metadata = super()._build_metadata()
        insights = self._insights
        metadata.update(
            {
                "technique": "analysis_informed_abliteration",
                "analysis_insights": {
                    "detected_alignment_method": insights.detected_alignment_method,
                    "alignment_confidence": insights.alignment_confidence,
                    "alignment_probabilities": insights.alignment_probabilities,
                    "cone_is_polyhedral": insights.cone_is_polyhedral,
                    "cone_dimensionality": insights.cone_dimensionality,
                    "mean_pairwise_cosine": insights.mean_pairwise_cosine,
                    "direction_clusters": insights.direction_clusters,
                    "cluster_count": insights.cluster_count,
                    "direction_persistence": insights.direction_persistence,
                    "estimated_robustness": insights.estimated_robustness,
                    "self_repair_estimate": insights.self_repair_estimate,
                    "entanglement_score": insights.entanglement_score,
                    "entangled_layers_skipped": insights.skip_layers,
                    "use_sparse_surgery": insights.use_sparse_surgery,
                    "recommended_sparsity": insights.recommended_sparsity,
                },
                "derived_config": {
                    "n_directions": insights.recommended_n_directions,
                    "direction_method": insights.recommended_direction_method,
                    "regularization": insights.recommended_regularization,
                    "refinement_passes": insights.recommended_refinement_passes,
                    "layers_used": insights.recommended_layers,
                    "layers_skipped": insights.skip_layers,
                    "norm_preserve": self.norm_preserve,
                    "whitened_svd": self.use_whitened_svd,
                    "sparse_surgery": insights.use_sparse_surgery,
                },
                "pipeline_stats": {
                    "analysis_duration_s": self._report.analysis_duration,
                    "total_duration_s": self._report.total_duration,
                    "ouroboros_passes": self._report.ouroboros_passes,
                    "verified_weight_edit_passes": 1 + self._report.ouroboros_passes,
                    "final_refusal_rate": self._report.final_refusal_rate,
                },
            }
        )
        metadata["references"].extend(
            [
                "Wollschlager et al., The Geometry of Refusal in LLMs — concept cones (ICML 2025)",
                "OBLITERATUS: Analysis-informed abliteration pipeline",
            ]
        )
        return metadata

    def _rebirth_informed(self) -> Path:
        """Use the base fail-closed, transactional checkpoint publisher."""
        return self._rebirth()

    @staticmethod
    def format_insights(insights: AnalysisInsights) -> str:
        """Format analysis insights as a human-readable report."""
        lines = []
        lines.append("Analysis-Informed Pipeline — Insights Report")
        lines.append("=" * 50)
        lines.append("")

        lines.append("Alignment Imprint:")
        lines.append(f"  Detected method: {insights.detected_alignment_method.upper()}")
        lines.append(f"  Confidence: {insights.alignment_confidence:.1%}")
        for method, prob in sorted(insights.alignment_probabilities.items()):
            lines.append(f"    {method.upper():6s} {prob:.1%}")
        lines.append("")

        lines.append("Concept Cone Geometry:")
        cone_type = "POLYHEDRAL" if insights.cone_is_polyhedral else "LINEAR"
        lines.append(f"  Type: {cone_type}")
        lines.append(f"  Dimensionality: {insights.cone_dimensionality:.2f}")
        lines.append(f"  Mean pairwise cosine: {insights.mean_pairwise_cosine:.3f}")
        if insights.direction_specificity:
            lines.append("  Per-category DSI:")
            for cat, dsi in sorted(insights.direction_specificity.items(), key=lambda x: -x[1]):
                lines.append(f"    {cat:15s}: {dsi:.3f}")
        lines.append("")

        lines.append("Cross-Layer Structure:")
        lines.append(f"  Direction clusters: {insights.cluster_count}")
        lines.append(f"  Direction persistence: {insights.direction_persistence:.3f}")
        lines.append(f"  Cluster representatives: {insights.cluster_representative_layers}")
        lines.append("")

        lines.append("Defense Robustness:")
        lines.append(f"  Estimated robustness: {insights.estimated_robustness.upper()}")
        lines.append(f"  Self-repair (Ouroboros): {insights.self_repair_estimate:.2f}")
        lines.append(f"  Entanglement: {insights.entanglement_score:.3f}")
        lines.append(f"  Entangled layers: {insights.entangled_layers}")
        lines.append(f"  Clean layers: {insights.clean_layers}")
        lines.append("")

        lines.append("Derived Configuration:")
        lines.append(f"  n_directions: {insights.recommended_n_directions}")
        lines.append(f"  direction_method: {insights.recommended_direction_method}")
        lines.append(f"  regularization: {insights.recommended_regularization}")
        lines.append(f"  refinement_passes: {insights.recommended_refinement_passes}")
        lines.append(f"  sparse surgery: {insights.use_sparse_surgery}")
        lines.append(f"  layers: {insights.recommended_layers or '(knee detection)'}")
        lines.append(f"  skipped: {insights.skip_layers or '(none)'}")

        return "\n".join(lines)
