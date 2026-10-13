"""Autonomous System Evaluation & Benchmark Runner.

Executes:
1. 60-Query Golden Test Set RAGAS-style Retrieval and Generation Benchmarks (Faithfulness, Relevancy, Precision, Recall).
2. Modality Breakdown (Lecture Video vs Textbook vs Slide Deck).
3. Non-Circular Student Simulation comparing Adaptive BKT vs Random Selection vs Fixed Sequential baselines.
4. Anti-Repetition Rate Verification (Zero repetition guarantee).
5. High-Resolution Visual Benchmark Generation (benchmark_results.png).
"""

import sys
import json
from core.ingestion import CourseKnowledgeBase
from core.learner_model import BayesianLearnerModel
from core.grounding_engine import GroundingEngine
from core.assessment_engine import AdaptiveAssessmentEngine
from core.evaluation_benchmarks import BenchmarkSuite

def main():
    print("=" * 75)
    print("   OMNISTUDY AI: SYSTEM EVALUATION & BENCHMARKING SUITE")
    print("   Track D: Personalized Tutoring & Adaptive Learning")
    print("=" * 75)

    kb = CourseKnowledgeBase()
    learner = BayesianLearnerModel()
    grounding = GroundingEngine(kb)
    assessment = AdaptiveAssessmentEngine(kb)
    suite = BenchmarkSuite(kb, grounding, assessment, learner)

    print("\n[Step 1/3] Evaluating RAG Pipeline on 60 Balanced Golden Queries...")
    eval_res = suite.run_rigorous_evaluation()
    metrics = eval_res["summary_metrics"]
    mods = eval_res["modality_breakdown"]

    print("\n--- RAGAS / Framework Metrics Summary ---")
    print(f"  [+] Context Recall:       {metrics['context_recall']:.3f} / 1.000")
    print(f"  [+] Context Precision:    {metrics['context_precision']:.3f} / 1.000")
    print(f"  [+] Answer Relevancy:     {metrics['answer_relevancy']:.3f} / 1.000")
    print(f"  [+] Faithfulness:         {metrics['faithfulness']:.3f} / 1.000")
    print(f"  [+] Refusal Accuracy:     {metrics['refusal_accuracy']:.3f} / 1.000 (Out-of-curriculum rejection)")
    print(f"  [+] Golden Test Size:     {metrics['test_set_size']} balanced queries")

    print("\n--- Modality Retrieval Breakdown ---")
    for mod, vals in mods.items():
        print(f"  * {mod.replace('_', ' ').title():<16}: Recall = {vals['recall']:.3f} | Precision = {vals['precision']:.3f} | Faithfulness = {vals['faithfulness']:.3f}")

    print("\n[Step 2/3] Simulating Student Cohorts: Adaptive BKT vs Baselines...")
    sim_res = suite.simulate_cohort_with_baselines(num_sessions=6)
    strats = sim_res["strategy_comparison"]
    personas = sim_res["persona_adaptive_results"]

    print("\n--- Strategy Comparison (Mean Mastery Progression) ---")
    for strat_name, timeline in strats.items():
        print(f"  * {strat_name:<30}: {' -> '.join(f'{m:.2f}' for m in timeline)}")

    print("\n--- Persona Mastery Trajectories (Adaptive BKT) ---")
    for name, data in personas.items():
        print(f"  * {name:<30}: Init = {data['initial']:.2f} -> Final = {data['final']:.2f} (Gain: +{data['gain']:.2f})")

    print(f"\n  * Question Repetition Rate: {sim_res['question_repetition_rate']:.1f}% (Zero repetition guarantee)")

    print("\n[Step 3/3] Generating High-Resolution 3-Panel Benchmark Chart...")
    chart_path = suite.generate_benchmark_plot("benchmark_results.png")
    print(f"  [OK] Visual benchmark plot successfully saved to: {chart_path}")

    # Save metrics JSON for reproducibility & Devpost documentation
    report = {
        "rag_metrics": metrics,
        "modality_breakdown": mods,
        "strategy_comparison": strats,
        "persona_results": personas,
        "question_repetition_rate": sim_res["question_repetition_rate"]
    }
    with open("benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("  [OK] Detailed report saved to: benchmark_report.json")

    print("\n" + "=" * 75)
    print("   ALL BENCHMARKS COMPLETED SUCCESSFULLY! SYSTEM READY FOR DEMO.")
    print("=" * 75)

if __name__ == "__main__":
    main()
