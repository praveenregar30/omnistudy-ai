"""Autonomous System Evaluation & Benchmark Runner.

Executes:
1. Golden Test Set RAGAS-style Retrieval and Generation Benchmarks.
2. Multi-Session Simulated Student Cohort (BKT Mastery Gains).
3. Question Repetition Rate Verification (Anti-Repetition Engine).
4. High-Resolution Visual Chart Generation (benchmark_results.png).
"""

import sys
import json
from core.ingestion import CourseKnowledgeBase
from core.learner_model import BayesianLearnerModel
from core.grounding_engine import GroundingEngine
from core.assessment_engine import AdaptiveAssessmentEngine
from core.evaluation_benchmarks import BenchmarkSuite

def main():
    print("=" * 70)
    print("   OMNISTUDY AI: SYSTEM EVALUATION & BENCHMARKING SUITE")
    print("   Track D: Personalized Tutoring & Adaptive Learning")
    print("=" * 70)

    kb = CourseKnowledgeBase()
    learner = BayesianLearnerModel()
    grounding = GroundingEngine(kb)
    assessment = AdaptiveAssessmentEngine(kb)
    suite = BenchmarkSuite(kb, grounding, assessment, learner)

    print("\n[Step 1/3] Evaluating RAG Pipeline (Faithfulness, Relevancy, Precision, Recall)...")
    eval_res = suite.evaluate_retrieval_and_generation()
    metrics = eval_res["summary_metrics"]

    print("\n--- RAGAS / Framework Metrics Summary ---")
    print(f"  [+] Faithfulness:         {metrics['faithfulness']:.3f} / 1.000")
    print(f"  [+] Answer Relevancy:     {metrics['answer_relevancy']:.3f} / 1.000")
    print(f"  [+] Context Precision:    {metrics['context_precision']:.3f} / 1.000")
    print(f"  [+] Context Recall:       {metrics['context_recall']:.3f} / 1.000")
    print(f"  [+] Refusal Accuracy:     {metrics['refusal_accuracy']:.3f} / 1.000 (Out-of-curriculum rejection)")
    print(f"  [+] Overall RAG Score:    {metrics['overall_score']:.3f} / 1.000")

    print("\n[Step 2/3] Simulating Multi-Session Student Cohort (Personalization & BKT)...")
    sim_res = suite.simulate_student_cohort(num_sessions=6, questions_per_session=3)
    cohort = sim_res["cohort_results"]

    print("\n--- Student Cohort Mastery Progression ---")
    for name, data in cohort.items():
        print(f"  * {name}:")
        print(f"      Initial Mastery P(L0): {data['initial_mastery']:.2f}")
        print(f"      Final Mastery P(Lt):   {data['final_mastery']:.2f}")
        print(f"      Net Mastery Gain:      +{data['mastery_gain']:.2f}")
        print(f"      Trajectory:            {' -> '.join(str(m) for m in data['mastery_timeline'])}")

    print(f"\n  * Question Repetition Rate: {sim_res['question_repetition_rate']:.1f}% (Zero repetition guarantee)")

    print("\n[Step 3/3] Generating High-Resolution Benchmark Plots...")
    chart_path = suite.generate_benchmark_plot("benchmark_results.png")
    print(f"  [OK] Visual benchmark plot successfully saved to: {chart_path}")

    # Save metrics JSON for reproducibility
    report = {
        "rag_metrics": metrics,
        "cohort_simulation": cohort,
        "question_repetition_rate": sim_res["question_repetition_rate"]
    }
    with open("benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("  [OK] Detailed report saved to: benchmark_report.json")

    print("\n" + "=" * 70)
    print("   ALL BENCHMARKS COMPLETED SUCCESSFULLY! SYSTEM READY FOR DEMO.")
    print("=" * 70)

if __name__ == "__main__":
    main()
