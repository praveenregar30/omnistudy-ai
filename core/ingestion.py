"""Multimodal Ingestion Pipeline & Concept Knowledge Graph Builder.

Supports ingesting lecture videos (timestamped), textbooks (paginated), and
slide decks (slide-numbered with diagram/figure extraction), and constructs
a topic-concept-prerequisite DAG.
"""

from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class DiagramMeta:
    """Metadata describing figures, plots, or diagrams in materials."""
    figure_id: str
    caption: str
    diagram_type: str  # e.g., 'loss_curve', 'architecture_diagram', 'scatter_plot', 'table'
    context_text: str

@dataclass
class ContentUnit:
    """Atomic multimodal chunk tagged with source origin and concept metadata."""
    unit_id: str
    source_type: str  # 'lecture_video', 'textbook', 'slide_deck'
    source_title: str
    location: str     # e.g. 'Video @ 04:15 - 08:30', 'Textbook p. 42-43', 'Slide 14'
    timestamp_seconds: Optional[int] = None
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    topic: str = ""
    subtopic: str = ""
    prerequisites: List[str] = field(default_factory=list)
    content_text: str = ""
    diagram: Optional[DiagramMeta] = None
    key_formulas: List[str] = field(default_factory=list)

@dataclass
class ConceptNode:
    """Node in the Course Prerequisite DAG."""
    concept_id: str
    name: str
    topic: str
    description: str
    prerequisites: List[str] = field(default_factory=list)
    source_units: List[str] = field(default_factory=list)

class CourseKnowledgeBase:
    """Unified Multimodal Course Store with DAG graph and cross-modal retrieval."""

    def __init__(self, course_name: str = "CS 229: Machine Learning & Pattern Recognition"):
        self.course_name = course_name
        self.units: Dict[str, ContentUnit] = {}
        self.concepts: Dict[str, ConceptNode] = {}
        self.topics: Dict[str, List[str]] = {}
        self._load_default_curriculum()

    def add_unit(self, unit: ContentUnit) -> None:
        self.units[unit.unit_id] = unit
        if unit.topic not in self.topics:
            self.topics[unit.topic] = []
        if unit.subtopic and unit.subtopic not in self.topics[unit.topic]:
            self.topics[unit.topic].append(unit.subtopic)

    def add_concept(self, concept: ConceptNode) -> None:
        self.concepts[concept.concept_id] = concept

    def get_prerequisite_dag(self) -> Dict[str, Any]:
        """Returns nodes and edges for visualization and prerequisite tracking."""
        nodes = []
        edges = []
        for cid, node in self.concepts.items():
            nodes.append({
                "id": cid,
                "label": node.name,
                "topic": node.topic,
                "desc": node.description
            })
            for prereq in node.prerequisites:
                edges.append({
                    "from": prereq,
                    "to": cid
                })
        return {"nodes": nodes, "edges": edges}

    def search(self, query: str, top_k: int = 4, topic_filter: Optional[str] = None) -> List[ContentUnit]:
        """Multi-criteria semantic & lexical retrieval over all course modalities."""
        query_terms = set(re.findall(r"\w+", query.lower()))
        scores = []
        
        for unit in self.units.values():
            if topic_filter and unit.topic != topic_filter:
                continue
            
            # Combine unit text, diagrams, formulas for comprehensive matching
            combined = f"{unit.topic} {unit.subtopic} {unit.content_text} "
            if unit.diagram:
                combined += f"{unit.diagram.caption} {unit.diagram.context_text} "
            combined += " ".join(unit.key_formulas)
            
            unit_terms = set(re.findall(r"\w+", combined.lower()))
            overlap = len(query_terms.intersection(unit_terms))
            
            # Boost matches on subtopic and topic
            if any(term in unit.topic.lower() for term in query_terms):
                overlap += 3
            if any(term in unit.subtopic.lower() for term in query_terms):
                overlap += 4
            
            # Boost if formula query
            if any(f in query for f in ["loss", "gradient", "sigmoid", "matrix", "derivative", "mse"]):
                if unit.key_formulas:
                    overlap += 2
                    
            if overlap > 0:
                scores.append((overlap, unit))
                
        scores.sort(key=lambda x: x[0], reverse=True)
        return [unit for _, unit in scores[:top_k]]

    def _load_default_curriculum(self) -> None:
        """Seed rich multimodal curriculum with lectures, textbooks, and slides."""
        # Concept nodes & Prerequisite hierarchy
        c_nodes = [
            ConceptNode("c_linreg", "Linear Regression & MSE", "Supervised Learning", 
                        "Ordinary Least Squares formulation, hypothesis function, and Mean Squared Error cost.",
                        prerequisites=[]),
            ConceptNode("c_grad_desc", "Gradient Descent Optimization", "Supervised Learning",
                        "Batch, Stochastic, and Mini-batch gradient descent update rules and learning rate alpha.",
                        prerequisites=["c_linreg"]),
            ConceptNode("c_logreg", "Logistic Regression & Sigmoid", "Classification",
                        "Binary classification, sigmoid activation mapping, and log-loss cost function.",
                        prerequisites=["c_linreg", "c_grad_desc"]),
            ConceptNode("c_eval_metrics", "Classification Metrics & Confusion Matrix", "Classification",
                        "Precision, Recall, F1-Score, ROC-AUC, and confusion matrix analysis.",
                        prerequisites=["c_logreg"]),
            ConceptNode("c_bias_var", "Bias-Variance Tradeoff", "Model Evaluation & Regularization",
                        "Decomposition of expected error into bias squared, variance, and irreducible noise.",
                        prerequisites=["c_linreg", "c_logreg"]),
            ConceptNode("c_regularization", "L1 Lasso & L2 Ridge Regularization", "Model Evaluation & Regularization",
                        "Penalty terms on weights to prevent overfitting, weight decay, and sparsity.",
                        prerequisites=["c_bias_var"]),
            ConceptNode("c_neural_nets", "Multilayer Perceptrons & Backpropagation", "Deep Learning",
                        "Feedforward architecture, non-linear activation functions (ReLU, Sigmoid), and chain-rule backpropagation.",
                        prerequisites=["c_logreg", "c_grad_desc"]),
            ConceptNode("c_kmeans", "K-Means Clustering", "Unsupervised Learning",
                        "Expectation-Maximization partition heuristic, centroids update, and inertia/elbow method.",
                        prerequisites=[])
        ]
        for c in c_nodes:
            self.add_concept(c)

        # Content Units (Lectures, Textbooks, Slides)
        sample_units = [
            # 1. Linear Regression
            ContentUnit(
                unit_id="unit_vid_01",
                source_type="lecture_video",
                source_title="Lecture 1: Intro to Supervised Learning & Linear Regression",
                location="Video @ 06:12 - 11:45",
                timestamp_seconds=372,
                topic="Supervised Learning",
                subtopic="Linear Regression & MSE",
                prerequisites=[],
                content_text="Professor explains the hypothesis function h_theta(x) = theta^T x. In linear regression, our goal is to fit a continuous hyperplane through training data points. We measure error using the Mean Squared Error (MSE) cost function J(theta) = 1/(2m) * sum((h_theta(x_i) - y_i)^2). The factor of 1/2 is intentionally introduced so that the derivative neatly cancels the exponent 2.",
                key_formulas=["h_θ(x) = θ^T x", "J(θ) = 1/(2m) ∑_{i=1}^m (h_θ(x^{(i)}) - y^{(i)})^2"],
                diagram=DiagramMeta("fig_linreg_fit", "Scatter plot with fitted linear regression line showing residual error lines (vertical offsets)", "scatter_plot", "Visualizing residual error differences between predicted y-hat and observed ground truth y.")
            ),
            ContentUnit(
                unit_id="unit_slide_01",
                source_type="slide_deck",
                source_title="Lecture 01 Slides: Linear Models",
                location="Slide 8",
                slide_number=8,
                topic="Supervised Learning",
                subtopic="Linear Regression & MSE",
                prerequisites=[],
                content_text="Slide 8: Cost Function Contour Plot. For a 2-parameter model (theta_0, theta_1), J(theta) forms an elliptical bowl (paraboloid). Each elliptical contour represents equal cost. The global minimum lies at the center.",
                diagram=DiagramMeta("fig_contour_mse", "Elliptical contour plot of J(theta_0, theta_1) with concentric ellipses and global minimum point", "contour_plot", "Demonstrating convexity of the linear regression MSE cost surface, guaranteeing no local minima.")
            ),
            # 2. Gradient Descent
            ContentUnit(
                unit_id="unit_vid_02",
                source_type="lecture_video",
                source_title="Lecture 2: Gradient Descent Optimization",
                location="Video @ 04:30 - 09:15",
                timestamp_seconds=270,
                topic="Supervised Learning",
                subtopic="Gradient Descent Optimization",
                prerequisites=["c_linreg"],
                content_text="The gradient descent update rule is: theta_j := theta_j - alpha * (d/d theta_j) J(theta). For linear regression MSE, the partial derivative is (1/m) * sum((h_theta(x) - y) * x_j). Crucial detail: all parameters theta_0, theta_1, ... theta_n must be updated simultaneously at each iteration step! If learning rate alpha is too small, convergence is slow; if alpha is too large, gradient descent can overshoot and diverge.",
                key_formulas=["θ_j := θ_j - α * (1/m) ∑_{i=1}^m (h_θ(x^{(i)}) - y^{(i)}) x_j^{(i)}"],
                diagram=DiagramMeta("fig_alpha_divergence", "Comparison of convergence trajectories for small, optimal, and excessive learning rate alpha", "trajectories", "Showing oscillating overshoot when alpha exceeds 2 / lambda_max of the Hessian.")
            ),
            ContentUnit(
                unit_id="unit_book_01",
                source_type="textbook",
                source_title="Bishop - Pattern Recognition and Machine Learning",
                location="Textbook p. 142-144",
                page_number=142,
                topic="Supervised Learning",
                subtopic="Gradient Descent Optimization",
                prerequisites=["c_linreg"],
                content_text="Section 3.1.3: Batch vs. Stochastic Gradient Descent (SGD). Batch Gradient Descent computes the gradient over the entire training set of size m at each step, ensuring monotonic decrease in cost on convex surfaces but computationally expensive for large datasets. Stochastic Gradient Descent computes updates based on a single sample, introducing stochastic fluctuations that can help escape saddle points.",
                key_formulas=["SGD: θ := θ - α * ∇ J_i(θ)"]
            ),
            # 3. Logistic Regression & Classification
            ContentUnit(
                unit_id="unit_book_02",
                source_type="textbook",
                source_title="Murphy - Probabilistic Machine Learning",
                location="Textbook p. 288-291",
                page_number=288,
                topic="Classification",
                subtopic="Logistic Regression & Sigmoid",
                prerequisites=["c_linreg", "c_grad_desc"],
                content_text="Chapter 10: Binary Logistic Regression. Rather than modeling continuous values, we model the posterior class probability P(y=1|x) = sigma(w^T x). The sigmoid function is defined as sigma(z) = 1 / (1 + exp(-z)). It maps any real number in (-inf, +inf) strictly into (0, 1). We cannot use MSE for logistic regression because the resulting cost function would be non-convex with numerous local minima. Instead, we derive the Negative Log-Likelihood (Binary Cross-Entropy Loss): J(w) = -1/m * sum( y*log(y_hat) + (1-y)*log(1-y_hat) ).",
                key_formulas=[r"\sigma(z) = 1 / (1 + e^{-z})", r"J(w) = - 1/m \sum [y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})]"],
                diagram=DiagramMeta("fig_sigmoid_curve", "Sigmoid activation curve S-shape with asymptote at 0 and 1, crossing 0.5 at z=0", "activation_curve", "Illustrates continuous probability mapping and saturation regions where gradient diminishes.")
            ),
            ContentUnit(
                unit_id="unit_slide_02",
                source_type="slide_deck",
                source_title="Lecture 03 Slides: Classification & Evaluation",
                location="Slide 16",
                slide_number=16,
                topic="Classification",
                subtopic="Classification Metrics & Confusion Matrix",
                prerequisites=["c_logreg"],
                content_text="Slide 16: Evaluation Metrics for Imbalanced Datasets. Accuracy can be deceiving when classes are imbalanced. We compute: Precision = TP / (TP + FP) [quantifies exactness]; Recall = TP / (TP + FN) [quantifies completeness / sensitivity]; F1-Score = 2 * (Precision * Recall) / (Precision + Recall) [harmonic mean]. The ROC curve plots True Positive Rate vs. False Positive Rate across all decision thresholds.",
                key_formulas=["Precision = TP / (TP + FP)", "Recall = TP / (TP + FN)", "F1 = 2 * (P * R) / (P + R)"],
                diagram=DiagramMeta("fig_confusion_matrix", "2x2 Confusion matrix diagram labeling TP, FP, FN, TN with margin calculations", "table_matrix", "Visual reference for type I (false positive) and type II (false negative) error distributions.")
            ),
            # 4. Bias-Variance & Regularization
            ContentUnit(
                unit_id="unit_vid_03",
                source_type="lecture_video",
                source_title="Lecture 4: Bias-Variance Decomposition & Regularization",
                location="Video @ 12:40 - 18:20",
                timestamp_seconds=760,
                topic="Model Evaluation & Regularization",
                subtopic="Bias-Variance Tradeoff",
                prerequisites=["c_linreg", "c_logreg"],
                content_text="Total expected generalization error decomposes into three components: Error = Bias^2 + Variance + Irreducible Noise. High Bias corresponds to underfitting (the model makes strong oversimplifying assumptions, failing to capture underlying patterns, showing high training error). High Variance corresponds to overfitting (the model is too complex, fitting noise in the training set, showing low training error but high test error).",
                key_formulas=[r"E[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2"],
                diagram=DiagramMeta("fig_bias_variance_u_curve", "U-shaped curve showing training error decreasing monotonically while validation error reaches a minimum then increases due to variance", "curves", "Visualizing sweet spot of model complexity balancing underfitting and overfitting.")
            ),
            ContentUnit(
                unit_id="unit_book_03",
                source_type="textbook",
                source_title="Hastie et al. - Elements of Statistical Learning",
                location="Textbook p. 66-70",
                page_number=66,
                topic="Model Evaluation & Regularization",
                subtopic="L1 Lasso & L2 Ridge Regularization",
                prerequisites=["c_bias_var"],
                content_text="Chapter 3.4: Shrinkage Methods. Ridge Regression (L2) adds lambda * sum(theta_j^2) penalty. It shrinks weights continuously towards zero but never forces them to exact zero. Lasso Regression (L1) adds lambda * sum(|theta_j|) penalty. Because the L1 constraint region is a diamond with sharp vertices on the coordinate axes, the elliptical contours of the unconstrained cost often intersect at a vertex, setting non-informative feature weights strictly to zero, effectively performing automatic feature selection.",
                key_formulas=[r"Ridge: J_{ridge}(\theta) = J(\theta) + \lambda \sum_{j=1}^n \theta_j^2", r"Lasso: J_{lasso}(\theta) = J(\theta) + \lambda \sum_{j=1}^n |\theta_j|"],
                diagram=DiagramMeta("fig_l1_l2_geometry", "Geometric comparison of L1 diamond vs L2 circle constraint boundaries intersecting loss contours", "geometry", "Visual explanation for why L1 Lasso yields sparse weight vectors.")
            ),
            # 5. Deep Learning & Neural Nets
            ContentUnit(
                unit_id="unit_vid_04",
                source_type="lecture_video",
                source_title="Lecture 5: Neural Networks & Backpropagation",
                location="Video @ 15:10 - 22:30",
                timestamp_seconds=910,
                topic="Deep Learning",
                subtopic="Multilayer Perceptrons & Backpropagation",
                prerequisites=["c_logreg", "c_grad_desc"],
                content_text="Multilayer Perceptrons (MLPs) overcome the linear separability limitation of single perceptrons (such as the XOR problem) by stacking hidden layers with non-linear activation functions. Common activations include ReLU (max(0, z)), Leaky ReLU, and Sigmoid. Backpropagation uses the multivariate calculus chain rule to compute gradients of the loss with respect to every weight parameter in reverse topological order, from output layer to input layer.",
                key_formulas=[r"a^{[l]} = g(W^{[l]} a^{[l-1]} + b^{[l]})", r"\delta^{[l]} = (W^{[l+1]T} \delta^{[l+1]}) \odot g'(z^{[l]})"],
                diagram=DiagramMeta("fig_mlp_flow", "Multilayer network diagram with input, hidden, and output layers showing forward activation flow and backward error flow", "network_graph", "Visualizing forward pass computation of activations followed by backward pass gradient propagation.")
            ),
            # 6. Unsupervised Learning
            ContentUnit(
                unit_id="unit_slide_03",
                source_type="slide_deck",
                source_title="Lecture 06 Slides: Unsupervised Clustering",
                location="Slide 11",
                slide_number=11,
                topic="Unsupervised Learning",
                subtopic="K-Means Clustering",
                prerequisites=[],
                content_text="Slide 11: K-Means Algorithm. 1. Randomly initialize K cluster centroids mu_1, ... mu_K. 2. Assignment Step: assign each training sample x_i to the nearest centroid argmin_k ||x_i - mu_k||^2. 3. Update Step: compute new centroid mu_k as the mean of all points assigned to cluster k. Repeat until centroids stop shifting. The objective minimizes inertia (Within-Cluster Sum of Squares). Optimal K can be chosen using the Elbow Method or Silhouette Score.",
                key_formulas=[r"J(c, \mu) = \sum_{i=1}^m ||x^{(i)} - \mu_{c^{(i)}}||^2"],
                diagram=DiagramMeta("fig_kmeans_elbow", "Elbow method plot showing sharp drop in inertia from K=1 to K=3 followed by diminishing returns", "elbow_plot", "Determining optimal number of clusters K based on inertia inflection point.")
            )
        ]
        for u in sample_units:
            self.add_unit(u)

    def ingest_custom_document(self, title: str, text: str, source_type: str, topic: str, subtopic: str, location: str) -> ContentUnit:
        """Dynamically ingests user uploaded files and adds them into the live knowledge base."""
        uid = f"custom_{len(self.units) + 1}_{re.sub(r'[^a-zA-Z0-9]', '', title)[:8]}"
        unit = ContentUnit(
            unit_id=uid,
            source_type=source_type,
            source_title=title,
            location=location,
            topic=topic or "General Course Materials",
            subtopic=subtopic or title,
            content_text=text
        )
        self.add_unit(unit)
        return unit
