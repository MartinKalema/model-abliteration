"""Novel analysis techniques for mechanistic interpretability of refusal."""

from obliteratus.analysis.activation_patching import ActivationPatcher
from obliteratus.analysis.activation_probing import ActivationProbe
from obliteratus.analysis.alignment_imprint import AlignmentImprintDetector
from obliteratus.analysis.anti_ouroboros import AntiOuroborosProber
from obliteratus.analysis.bayesian_kernel_projection import BayesianKernelProjection
from obliteratus.analysis.causal_tracing import CausalRefusalTracer
from obliteratus.analysis.concept_geometry import (
    CategoryDirectionDispersionAnalyzer,
    ConceptConeAnalyzer,
)
from obliteratus.analysis.conditional_abliteration import ConditionalAbliterator
from obliteratus.analysis.cot_preservation import (
    DEFAULT_COT_PRESERVATION_EXAMPLES,
    CoTPreservationError,
    CoTPreservationExample,
    CoTPreservationExampleResult,
    CoTPreservationReport,
    CoTScoreSnapshot,
    SegmentCrossEntropy,
    compare_cot_score_snapshots,
    evaluate_cot_preservation,
    normalize_final_answer,
    score_cot_references,
)
from obliteratus.analysis.cross_layer import CrossLayerAlignmentAnalyzer
from obliteratus.analysis.cross_model_transfer import TransferAnalyzer
from obliteratus.analysis.defense_robustness import DefenseRobustnessEvaluator
from obliteratus.analysis.gabliteration import (
    GabliterationError,
    GabliterationReplayPlan,
    GabliterationSearchConfig,
    GabliterationSearchResult,
    GabliterationValidationError,
    HiddenStateBatch,
    apply_gabliteration_replay,
    mean_separation_source_layer,
    paper_adaptive_layer_scales,
    ridge_subspace_update,
    run_gabliteration_search,
    shuffle_stabilized_svd_subspace,
)
from obliteratus.analysis.interventions import (
    DirectionalIntervention,
    InterventionError,
    ablate_direction,
    add_direction,
    run_with_directional_ablation,
    run_with_directional_addition,
)
from obliteratus.analysis.kl_preservation import (
    CandidateKLResult,
    CausalLMBatch,
    KLCandidateSelection,
    KLDirection,
    KLPreservationError,
    KLPreservationMetrics,
    KLPreservationThresholds,
    evaluate_kl_preservation,
    select_kl_preserving_candidate,
)
from obliteratus.analysis.leace import LEACEExtractor
from obliteratus.analysis.linear_eraser import ResidualEraser
from obliteratus.analysis.logit_lens import RefusalLogitLens
from obliteratus.analysis.multi_token_position import MultiTokenPositionAnalyzer
from obliteratus.analysis.probing_classifiers import LinearRefusalProbe
from obliteratus.analysis.rdo import (
    RDOConfig,
    RDOError,
    RDOEvidence,
    RDOEvidenceSummary,
    RDOGeneratedExample,
    RDOLossEvidence,
    RDOPromptSplit,
    RDOResult,
    RDOSnapshotEvidence,
    RDOTargetSequence,
    generate_rdo_evidence,
    optimize_rdo_direction,
    run_rdo,
)
from obliteratus.analysis.residual_stream import ResidualStreamDecomposer
from obliteratus.analysis.riemannian_manifold import RiemannianManifoldAnalyzer
from obliteratus.analysis.sae_abliteration import (
    SAEDecompositionPipeline,
    SparseAutoencoder,
    identify_refusal_features,
    train_sae,
)
from obliteratus.analysis.som_directions import (
    SOMDirectionExtractor,
    SOMDirectionResult,
)
from obliteratus.analysis.som_paper import (
    HarmBenchJudgeAdapter,
    SOMBehaviorExample,
    SOMBehaviorJudge,
    SOMCheckpointError,
    SOMCompletionGenerator,
    SOMDirectionPool,
    SOMEvidenceError,
    SOMEvidenceSplits,
    SOMGeneratorEvidence,
    SOMJudgeEvidence,
    SOMPaperError,
    SOMPaperSearchResult,
    SOMProjectionTarget,
    SOMReplayError,
    SOMRollbackError,
    SOMSearchConfig,
    SOMSubsetSearchResult,
    SOMTrainingConfig,
    SOMWinnerReplay,
    replay_som_winner,
    run_paper_som_search,
    search_som_direction_subsets,
    train_paper_som_directions,
)
from obliteratus.analysis.sparse_surgery import SparseDirectionSurgeon
from obliteratus.analysis.spectral_certification import (
    CertificationLevel,
    SpectralCertifier,
)
from obliteratus.analysis.steering_vectors import (
    SteeringHookManager,
    SteeringVectorFactory,
)
from obliteratus.analysis.tuned_lens import RefusalTunedLens, TunedLensTrainer
from obliteratus.analysis.wasserstein_optimal import WassersteinOptimalExtractor
from obliteratus.analysis.wasserstein_transfer import WassersteinRefusalTransfer
from obliteratus.analysis.whitened_svd import WhitenedSVDExtractor

__all__ = [
    "DEFAULT_COT_PRESERVATION_EXAMPLES",
    "ActivationPatcher",
    "ActivationProbe",
    "AlignmentImprintDetector",
    "AntiOuroborosProber",
    "BayesianKernelProjection",
    "CandidateKLResult",
    "CategoryDirectionDispersionAnalyzer",
    "CausalLMBatch",
    "CausalRefusalTracer",
    "CertificationLevel",
    "CoTPreservationError",
    "CoTPreservationExample",
    "CoTPreservationExampleResult",
    "CoTPreservationReport",
    "CoTScoreSnapshot",
    "ConceptConeAnalyzer",
    "ConditionalAbliterator",
    "CrossLayerAlignmentAnalyzer",
    "DefenseRobustnessEvaluator",
    "DirectionalIntervention",
    "GabliterationError",
    "GabliterationReplayPlan",
    "GabliterationSearchConfig",
    "GabliterationSearchResult",
    "GabliterationValidationError",
    "HarmBenchJudgeAdapter",
    "HiddenStateBatch",
    "InterventionError",
    "KLCandidateSelection",
    "KLDirection",
    "KLPreservationError",
    "KLPreservationMetrics",
    "KLPreservationThresholds",
    "LEACEExtractor",
    "LinearRefusalProbe",
    "MultiTokenPositionAnalyzer",
    "RDOConfig",
    "RDOError",
    "RDOEvidence",
    "RDOEvidenceSummary",
    "RDOGeneratedExample",
    "RDOLossEvidence",
    "RDOPromptSplit",
    "RDOResult",
    "RDOSnapshotEvidence",
    "RDOTargetSequence",
    "RefusalLogitLens",
    "RefusalTunedLens",
    "ResidualEraser",
    "ResidualStreamDecomposer",
    "RiemannianManifoldAnalyzer",
    "SAEDecompositionPipeline",
    "SOMBehaviorExample",
    "SOMBehaviorJudge",
    "SOMCheckpointError",
    "SOMCompletionGenerator",
    "SOMDirectionExtractor",
    "SOMDirectionPool",
    "SOMDirectionResult",
    "SOMEvidenceError",
    "SOMEvidenceSplits",
    "SOMGeneratorEvidence",
    "SOMJudgeEvidence",
    "SOMPaperError",
    "SOMPaperSearchResult",
    "SOMProjectionTarget",
    "SOMReplayError",
    "SOMRollbackError",
    "SOMSearchConfig",
    "SOMSubsetSearchResult",
    "SOMTrainingConfig",
    "SOMWinnerReplay",
    "SegmentCrossEntropy",
    "SparseAutoencoder",
    "SparseDirectionSurgeon",
    "SpectralCertifier",
    "SteeringHookManager",
    "SteeringVectorFactory",
    "TransferAnalyzer",
    "TunedLensTrainer",
    "WassersteinOptimalExtractor",
    "WassersteinRefusalTransfer",
    "WhitenedSVDExtractor",
    "ablate_direction",
    "add_direction",
    "apply_gabliteration_replay",
    "compare_cot_score_snapshots",
    "evaluate_cot_preservation",
    "evaluate_kl_preservation",
    "generate_rdo_evidence",
    "identify_refusal_features",
    "mean_separation_source_layer",
    "normalize_final_answer",
    "optimize_rdo_direction",
    "paper_adaptive_layer_scales",
    "replay_som_winner",
    "ridge_subspace_update",
    "run_gabliteration_search",
    "run_paper_som_search",
    "run_rdo",
    "run_with_directional_ablation",
    "run_with_directional_addition",
    "score_cot_references",
    "search_som_direction_subsets",
    "select_kl_preserving_candidate",
    "shuffle_stabilized_svd_subspace",
    "train_paper_som_directions",
    "train_sae",
]
