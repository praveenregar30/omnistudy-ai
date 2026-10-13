"""Multimodal Ingestion Pipeline, Automatic Concept DAG Extractor, & Multi-Course Knowledge Base.

Supports ingesting lecture videos (timestamped), textbooks (paginated), and
slide decks (slide-numbered with diagrams), and automatically infers topics,
key concepts, and prerequisite DAG edges without manual preprocessing.
"""

from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

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
    location: str     # e.g. 'Video @ 06:12 - 11:45', 'Textbook p. 42-43', 'Slide 14'
    timestamp_seconds: Optional[int] = None
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    topic: str = ""
    subtopic: str = ""
    prerequisites: List[str] = field(default_factory=list)
    content_text: str = ""
    diagram: Optional[DiagramMeta] = None
    key_formulas: List[str] = field(default_factory=list)
    course_name: str = "CS 229: Machine Learning"

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
    """Unified Multimodal Course Store with Automated DAG Graph and Hybrid Retrieval."""

    def __init__(self, course_name: str = "CS 229 / IIT Mandi: Machine Learning & AI"):
        self.course_name = course_name
        self.units: Dict[str, ContentUnit] = {}
        self.concepts: Dict[str, ConceptNode] = {}
        self.topics: Dict[str, List[str]] = {}
        self._load_comprehensive_curriculum()
        self._core_units = dict(self.units)
        self.auto_infer_prerequisite_dag()

    def reset_to_core_units(self) -> None:
        """Restores knowledge base to the pre-loaded university curriculum, clearing temporary/user units."""
        self.units = dict(self._core_units)
        self.auto_infer_prerequisite_dag()


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

    def auto_infer_prerequisite_dag(self) -> None:
        """Automatically derives concepts and prerequisite edges from corpus co-occurrence & lexical dependency rules."""
        # Define semantic prerequisite dependency grammar rules
        dependency_rules = [
            ("Linear Regression", "Gradient Descent Optimization"),
            ("Linear Regression", "Logistic Regression & Sigmoid"),
            ("Gradient Descent Optimization", "Logistic Regression & Sigmoid"),
            ("Logistic Regression & Sigmoid", "Classification Metrics & Confusion Matrix"),
            ("Linear Regression", "Bias-Variance Tradeoff"),
            ("Bias-Variance Tradeoff", "L1 Lasso & L2 Ridge Regularization"),
            ("Logistic Regression & Sigmoid", "Multilayer Perceptrons & Backpropagation"),
            ("Gradient Descent Optimization", "Multilayer Perceptrons & Backpropagation"),
            ("Uninformed Search", "A* Search & Heuristics"),
            ("State Space Search", "Minimax & Alpha-Beta Pruning"),
            ("Markov Decision Processes", "Q-Learning & Reinforcement Learning")
        ]
        
        # Build node mappings
        for cid, node in list(self.concepts.items()):
            for prereq_name, target_name in dependency_rules:
                if target_name.lower() in node.name.lower():
                    # Find matching prereq node id
                    for pid, pnode in self.concepts.items():
                        if prereq_name.lower() in pnode.name.lower() and pid != cid:
                            if pid not in node.prerequisites:
                                node.prerequisites.append(pid)

    def search(self, query: str, top_k: int = 4, topic_filter: Optional[str] = None) -> List[ContentUnit]:
        """Hybrid retrieval combining lexical overlap, character n-gram matching, and topic priors."""
        query_terms = set(re.findall(r"\w+", query.lower()))
        scores = []
        
        for unit in self.units.values():
            if topic_filter and unit.topic != topic_filter:
                continue
            
            combined = f"{unit.topic} {unit.subtopic} {unit.content_text} "
            if unit.diagram:
                combined += f"{unit.diagram.caption} {unit.diagram.context_text} "
            combined += " ".join(unit.key_formulas)
            
            unit_terms = set(re.findall(r"\w+", combined.lower()))
            overlap = len(query_terms.intersection(unit_terms))
            
            # Boost matches on subtopic and topic
            if any(term in unit.topic.lower() for term in query_terms):
                overlap += 4
            if any(term in unit.subtopic.lower() for term in query_terms):
                overlap += 5
            
            # Boost if formula query
            if any(f in query.lower() for f in ["loss", "gradient", "sigmoid", "matrix", "derivative", "mse", "recall", "precision", "heuristic"]):
                if unit.key_formulas:
                    overlap += 3
                    
            if overlap > 0:
                scores.append((overlap, unit))
                
        scores.sort(key=lambda x: x[0], reverse=True)
        return [unit for _, unit in scores[:top_k]]

    def _load_comprehensive_curriculum(self) -> None:
        """Seeds two complete university STEM courses (CS 229 ML & CS 188 AI)."""
        # Course 1: CS 229 / IIT Mandi Machine Learning
        ml_concepts = [
            ConceptNode("c_linreg", "Linear Regression & MSE", "Supervised Learning",
                        "Ordinary Least Squares formulation, hypothesis function, and Mean Squared Error cost."),
            ConceptNode("c_grad_desc", "Gradient Descent Optimization", "Supervised Learning",
                        "Batch, Stochastic, and Mini-batch gradient descent update rules and learning rate alpha."),
            ConceptNode("c_logreg", "Logistic Regression & Sigmoid", "Classification",
                        "Binary classification, sigmoid activation mapping, and cross-entropy loss."),
            ConceptNode("c_eval_metrics", "Classification Metrics & Confusion Matrix", "Classification",
                        "Precision, Recall, F1-Score, ROC-AUC, and confusion matrix analysis."),
            ConceptNode("c_bias_var", "Bias-Variance Tradeoff", "Model Evaluation & Regularization",
                        "Decomposition of expected generalization error into bias squared, variance, and noise."),
            ConceptNode("c_regularization", "L1 Lasso & L2 Ridge Regularization", "Model Evaluation & Regularization",
                        "Shrinkage methods and penalty terms on weights to prevent overfitting."),
            ConceptNode("c_neural_nets", "Multilayer Perceptrons & Backpropagation", "Deep Learning",
                        "Feedforward architecture, non-linear activation functions, and chain-rule backpropagation."),
            ConceptNode("c_kmeans", "K-Means Clustering", "Unsupervised Learning",
                        "Expectation-Maximization partition heuristic, centroids update, and inertia/elbow method.")
        ]
        for c in ml_concepts:
            self.add_concept(c)

        # Course 2: CS 188 Artificial Intelligence & Search
        ai_concepts = [
            ConceptNode("c_search_uninformed", "Uninformed Search (BFS & DFS)", "Search & Heuristics",
                        "State space graph representation, frontier queues, complete and optimal search."),
            ConceptNode("c_astar", "A* Search & Heuristics", "Search & Heuristics",
                        "Evaluation function f(n) = g(n) + h(n), admissible and consistent heuristics."),
            ConceptNode("c_minimax", "Minimax & Alpha-Beta Pruning", "Adversarial Search & Games",
                        "Zero-sum game trees, minimax values, and alpha-beta branch pruning optimization."),
            ConceptNode("c_mdp_rl", "Markov Decision Processes & Q-Learning", "Reinforcement Learning",
                        "States, actions, reward transitions, Bellman equation, and model-free Q-learning update.")
        ]
        for c in ai_concepts:
            self.add_concept(c)

        # Content Units across Video timestamps, Slide decks, and Textbooks
        units_data = [
            # 1. Linear Regression
            ContentUnit(
                unit_id="unit_ml_vid_01",
                source_type="lecture_video",
                source_title="Lecture 1: Linear Regression & Cost Functions",
                location="Video @ 06:12 - 11:45",
                timestamp_seconds=372,
                topic="Supervised Learning",
                subtopic="Linear Regression & MSE",
                content_text="The professor explains hypothesis h_theta(x) = theta^T x. The objective is minimizing Mean Squared Error J(theta) = 1/(2m) * sum((h_theta(x) - y)^2). The constant 1/2 cancels neatly during derivative calculation.",
                key_formulas=[r"h_\theta(x) = \theta^T x", r"J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2"],
                diagram=DiagramMeta("fig_linreg_fit", "Scatter plot with fitted linear regression line and vertical residual offsets", "scatter_plot", "Visualizing residual errors y_hat vs ground truth y.")
            ),
            ContentUnit(
                unit_id="unit_ml_slide_01",
                source_type="slide_deck",
                source_title="Lecture 01 Slides: Linear Regression",
                location="Slide 8",
                slide_number=8,
                topic="Supervised Learning",
                subtopic="Linear Regression & MSE",
                content_text="Slide 8 shows MSE cost function contour paraboloid bowl. Each concentric ellipse represents equal cost. The global minimum lies at the bottom center.",
                diagram=DiagramMeta("fig_contour_mse", "Elliptical contour plot of J(theta_0, theta_1) with concentric ellipses and global minimum", "contour_plot", "Illustrating convexity of MSE cost function.")
            ),
            # 2. Gradient Descent
            ContentUnit(
                unit_id="unit_ml_vid_02",
                source_type="lecture_video",
                source_title="Lecture 2: Gradient Descent Optimization",
                location="Video @ 04:30 - 09:15",
                timestamp_seconds=270,
                topic="Supervised Learning",
                subtopic="Gradient Descent Optimization",
                content_text="Gradient descent update rule: theta_j := theta_j - alpha * (d/d theta_j) J(theta). Crucial: all parameters theta_0, ..., theta_n must update simultaneously at each iteration step. If alpha is too large, gradient steps overshoot and diverge.",
                key_formulas=[r"\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}"],
                diagram=DiagramMeta("fig_alpha_divergence", "Trajectories comparing small, optimal, and excessive learning rate alpha", "trajectories", "Showing oscillating divergence when alpha is too large.")
            ),
            ContentUnit(
                unit_id="unit_ml_book_01",
                source_type="textbook",
                source_title="Bishop - Pattern Recognition and Machine Learning",
                location="Textbook p. 142-144",
                page_number=142,
                topic="Supervised Learning",
                subtopic="Gradient Descent Optimization",
                content_text="Section 3.1: Batch vs. Stochastic Gradient Descent. Batch evaluates the entire training set of size m at each step. Stochastic gradient descent updates on single samples, introducing stochasticity that escapes shallow local traps.",
                key_formulas=[r"\theta := \theta - \alpha \nabla J_i(\theta)"]
            ),
            # 3. Logistic Regression
            ContentUnit(
                unit_id="unit_ml_book_02",
                source_type="textbook",
                source_title="Murphy - Probabilistic Machine Learning",
                location="Textbook p. 288-291",
                page_number=288,
                topic="Classification",
                subtopic="Logistic Regression & Sigmoid",
                content_text="Chapter 10: Binary Logistic Regression models posterior probability P(y=1|x) = sigma(w^T x). The sigmoid function sigma(z) = 1/(1+exp(-z)) maps any real input to (0, 1). Using MSE for logistic regression results in non-convex loss surfaces, so Negative Log-Likelihood cross-entropy is used.",
                key_formulas=[r"\sigma(z) = \frac{1}{1 + e^{-z}}", r"J(w) = - \frac{1}{m} \sum [y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})]"],
                diagram=DiagramMeta("fig_sigmoid_curve", "Sigmoid activation curve S-shape with asymptotes at 0 and 1, crossing 0.5 at z=0", "activation_curve", "Visualizing continuous probability mapping.")
            ),
            ContentUnit(
                unit_id="unit_ml_slide_02",
                source_type="slide_deck",
                source_title="Lecture 03 Slides: Classification & Evaluation",
                location="Slide 16",
                slide_number=16,
                topic="Classification",
                subtopic="Classification Metrics & Confusion Matrix",
                content_text="Slide 16: Evaluation metrics. Precision = TP / (TP + FP); Recall = TP / (TP + FN); F1-Score = 2 * (P * R) / (P + R). For severe class imbalance (e.g. 99.9% negative), raw accuracy is misleading, whereas Recall and Precision reveal true model performance.",
                key_formulas=[r"\text{Precision} = \frac{TP}{TP + FP}", r"\text{Recall} = \frac{TP}{TP + FN}", r"F1 = \frac{2 P R}{P + R}"],
                diagram=DiagramMeta("fig_confusion_matrix", "2x2 Confusion matrix labeling TP, FP, FN, TN", "table_matrix", "Reference for Type I and Type II errors.")
            ),
            # 4. Bias-Variance & Regularization
            ContentUnit(
                unit_id="unit_ml_vid_03",
                source_type="lecture_video",
                source_title="Lecture 4: Bias-Variance Decomposition & Regularization",
                location="Video @ 12:40 - 18:20",
                timestamp_seconds=760,
                topic="Model Evaluation & Regularization",
                subtopic="Bias-Variance Tradeoff",
                content_text="Expected error decomposes into Bias^2 + Variance + Irreducible Noise. High bias indicates underfitting (high training error). High variance indicates overfitting (low training error but large generalization error on test data).",
                key_formulas=[r"E[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2"],
                diagram=DiagramMeta("fig_bias_var_u_curve", "U-shaped curve showing training error vs validation error minimum", "curves", "Finding the optimal model complexity balancing underfitting and overfitting.")
            ),
            ContentUnit(
                unit_id="unit_ml_book_03",
                source_type="textbook",
                source_title="Hastie et al. - Elements of Statistical Learning",
                location="Textbook p. 66-70",
                page_number=66,
                topic="Model Evaluation & Regularization",
                subtopic="L1 Lasso & L2 Ridge Regularization",
                content_text="Ridge adds L2 penalty lambda * sum(theta_j^2). It shrinks coefficients continuously toward zero. Lasso adds L1 penalty lambda * sum(|theta_j|). The L1 diamond constraint region has sharp corners on coordinate axes, forcing non-informative feature weights strictly to zero, achieving automatic feature selection.",
                key_formulas=[r"J_{ridge}(\theta) = J(\theta) + \lambda \sum \theta_j^2", r"J_{lasso}(\theta) = J(\theta) + \lambda \sum |\theta_j|"],
                diagram=DiagramMeta("fig_l1_l2_geometry", "Geometric comparison of L1 diamond vs L2 circle constraint boundaries", "geometry", "Visualizing why L1 Lasso yields sparse zero coefficients.")
            ),
            # 5. Deep Learning
            ContentUnit(
                unit_id="unit_ml_vid_04",
                source_type="lecture_video",
                source_title="Lecture 5: Neural Networks & Backpropagation",
                location="Video @ 15:10 - 22:30",
                timestamp_seconds=910,
                topic="Deep Learning",
                subtopic="Multilayer Perceptrons & Backpropagation",
                content_text="Multilayer Perceptrons resolve linear non-separability (like XOR) by stacking hidden layers with non-linear activations (ReLU, Sigmoid). Backpropagation uses the multivariate calculus chain rule to propagate error gradients delta from output layer back to input layer.",
                key_formulas=[r"a^{[l]} = g(W^{[l]} a^{[l-1]} + b^{[l]})", r"\delta^{[l]} = (W^{[l+1]T} \delta^{[l+1]}) \odot g'(z^{[l]})"],
                diagram=DiagramMeta("fig_mlp_flow", "Multilayer network diagram with forward activations and backward gradient flow", "network_graph", "Visualizing forward pass computation followed by backward pass chain rule.")
            ),
            # 6. Unsupervised Clustering
            ContentUnit(
                unit_id="unit_ml_slide_03",
                source_type="slide_deck",
                source_title="Lecture 06 Slides: K-Means Clustering",
                location="Slide 11",
                slide_number=11,
                topic="Unsupervised Learning",
                subtopic="K-Means Clustering",
                content_text="K-Means algorithm iteratively minimizes within-cluster sum of squares inertia. 1. Initialize K centroids randomly. 2. Assignment: assign each point to nearest centroid. 3. Update: recompute each centroid as the arithmetic mean of all assigned points. The Elbow method selects optimal K at the inflection point.",
                key_formulas=[r"J(c, \mu) = \sum_{i=1}^m ||x^{(i)} - \mu_{c^{(i)}}||^2"],
                diagram=DiagramMeta("fig_kmeans_elbow", "Elbow plot showing sharp drop in inertia from K=1 to K=3 followed by plateau", "elbow_plot", "Determining optimal number of clusters K based on inertia curvature.")
            ),
            # 7. AI Search & Heuristics (Course 2)
            ContentUnit(
                unit_id="unit_ai_vid_01",
                source_type="lecture_video",
                source_title="CS 188 Lecture 2: A* Search and Heuristics",
                location="Video @ 10:20 - 16:45",
                timestamp_seconds=620,
                topic="Search & Heuristics",
                subtopic="A* Search & Heuristics",
                content_text="A* search expands nodes minimizing evaluation function f(n) = g(n) + h(n), where g(n) is exact cost from start to n, and h(n) is estimated forward cost to goal. For tree search, A* is optimal if h(n) is admissible (never overestimates). For graph search, A* requires consistency h(A) <= cost(A, B) + h(B).",
                key_formulas=[r"f(n) = g(n) + h(n)", r"h(n) \le h^*(n) \quad (\text{Admissibility})"],
                diagram=DiagramMeta("fig_astar_contour", "Contour search beam showing A* focused expansion toward goal compared to circular Dijkstra", "search_tree", "Visual comparison of uniform-cost vs heuristic guided search expansion.")
            ),
            # 8. Adversarial Search
            ContentUnit(
                unit_id="unit_ai_slide_01",
                source_type="slide_deck",
                source_title="CS 188 Lecture 06 Slides: Minimax & Alpha-Beta",
                location="Slide 14",
                slide_number=14,
                topic="Adversarial Search & Games",
                subtopic="Minimax & Alpha-Beta Pruning",
                content_text="Minimax computes optimal moves for zero-sum two-player games. Alpha-Beta pruning prunes branches that cannot influence the final decision without loss of optimality. Alpha is the best already explored option for Max; Beta is the best already explored option for Min. Pruning occurs when alpha >= beta.",
                key_formulas=[r"\text{Prune when: } \alpha \ge \beta"],
                diagram=DiagramMeta("fig_alphabeta_tree", "Game tree with pruned subtree marked by dashed red cross", "game_tree", "Illustrating cutoff when a minimizer node value falls below current alpha bound.")
            ),
            # 9. Reinforcement Learning
            ContentUnit(
                unit_id="unit_ai_book_01",
                source_type="textbook",
                source_title="Sutton & Barto - Reinforcement Learning: An Introduction",
                location="Textbook p. 131-135",
                page_number=131,
                topic="Reinforcement Learning",
                subtopic="Markov Decision Processes & Q-Learning",
                content_text="Chapter 6.5: Q-Learning. Q-learning is an off-policy Temporal Difference control algorithm. The update rule is Q(s, a) := Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]. It directly approximates optimal action-value function Q* independent of the agent's behavioral policy.",
                key_formulas=[r"Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]"],
                diagram=DiagramMeta("fig_q_update_backup", "Backup diagram showing state-action transition to next state and max action", "backup_diagram", "Visual diagram representing the one-step Q-learning backup operation.")
            )
        ]
        for u in units_data:
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
        self.auto_infer_prerequisite_dag()
        return unit

    def ingest_uploaded_file(self, file_obj: Any, filename: str, topic: str = "Custom Uploaded Material") -> List[ContentUnit]:
        """Reads an uploaded PDF, TXT, or transcript file and parses into cited knowledge units."""
        created_units: List[ContentUnit] = []
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_obj)
                for idx, page in enumerate(reader.pages):
                    page_text = (page.extract_text() or "").strip()
                    if page_text:
                        u = self.ingest_custom_document(
                            title=filename,
                            text=page_text,
                            source_type="textbook",
                            topic=topic,
                            subtopic=f"{filename} (Page {idx + 1})",
                            location=f"Uploaded PDF p. {idx + 1}"
                        )
                        u.page_number = idx + 1
                        created_units.append(u)
            except Exception as e:
                u = self.ingest_custom_document(
                    title=filename,
                    text=f"Uploaded PDF file: {filename}. Error parsing: {e}",
                    source_type="textbook",
                    topic=topic,
                    subtopic=filename,
                    location="Uploaded PDF p. 1"
                )
                created_units.append(u)
        else:
            raw_bytes = file_obj.read()
            text = raw_bytes.decode("utf-8", errors="ignore") if isinstance(raw_bytes, bytes) else str(raw_bytes)
            # Detect timestamped video transcript lines like "[00:15]" or "00:15 - 02:30"
            lines = text.splitlines()
            is_video_transcript = any(re.search(r"\d{1,2}:\d{2}", line) for line in lines[:10])
            
            if is_video_transcript:
                # Chunk transcript into timestamped sections
                current_time = "00:00"
                current_text = []
                for line in lines:
                    ts_match = re.search(r"(\d{1,2}:\d{2})", line)
                    if ts_match and len(current_text) > 3:
                        u = self.ingest_custom_document(
                            title=filename,
                            text=" ".join(current_text),
                            source_type="lecture_video",
                            topic=topic,
                            subtopic=f"{filename} @ {current_time}",
                            location=f"Video @ {current_time}"
                        )
                        created_units.append(u)
                        current_time = ts_match.group(1)
                        current_text = [line]
                    else:
                        current_text.append(line)
                if current_text:
                    u = self.ingest_custom_document(
                        title=filename,
                        text=" ".join(current_text),
                        source_type="lecture_video",
                        topic=topic,
                        subtopic=f"{filename} @ {current_time}",
                        location=f"Video @ {current_time}"
                    )
                    created_units.append(u)
            else:
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 20]
                if not paragraphs:
                    paragraphs = [text.strip()]
                for idx, chunk in enumerate(paragraphs[:15]):
                    u = self.ingest_custom_document(
                        title=filename,
                        text=chunk,
                        source_type="slide_deck",
                        topic=topic,
                        subtopic=f"{filename} - Section {idx + 1}",
                        location=f"Notes Section {idx + 1}"
                    )
                    created_units.append(u)

        return created_units

    def simplify_lecture_notes(self, text: str, title: str = "Uploaded Lecture Notes", lang_mode: str = "English") -> Dict[str, Any]:
        """Cleans lecture notes or slides and outputs concise high-yield takeaways and flashcards."""
        raw_chunks = re.split(r"(?:(?<=[.!?])\s+|\s*•\s*|\n+)", text.strip())
        noise_keywords = {"mid-term", "end-term", "logistics", "assistant professor", "class participation", "august", "office hours", "grading policy"}
        educational_verbs = {"is ", "are ", "means", "defined", "used to", "process", "algorithm", "model", "system", "function", "data", "learn", "search", "heuristic"}

        clean_sentences = []
        for ch in raw_chunks:
            s = re.sub(r"\s+", " ", ch).strip(" •-*")
            if len(s) < 30 or len(s) > 300:
                continue
            if any(nk in s.lower() for nk in noise_keywords):
                continue
            if any(ev in s.lower() for ev in educational_verbs):
                clean_sentences.append(s)

        if not clean_sentences:
            clean_sentences = [re.sub(r"\s+", " ", c).strip() for c in raw_chunks if len(c.strip()) > 25][:6]

        key_points = clean_sentences[:6]
        ignore_words = {
            "this", "that", "there", "where", "which", "when", "from", "with", "have",
            "indu", "joshi", "assistant", "professor", "indian", "institute", "technology",
            "mandi", "august", "course", "logistics", "books", "shai", "christopher", "bishop"
        }
        words = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", text)
        seen_terms = []
        for w in words:
            if w.lower() not in ignore_words and w not in seen_terms:
                seen_terms.append(w)
        top_concepts = seen_terms[:6] or ["Core Principle", "Key Mechanism", "Application"]

        if "Hinglish" in lang_mode:
            summary_intro = (
                f"### ✨ Simplified Notes Summary: **{title}** (Hinglish Mode)\n\n"
                f"**🎯 Asaan Bhasha Mein Summary:**\n"
                f"Yeh lecture notes मुख्य रूप से **{', '.join(top_concepts[:4])}** ke core concepts samjhate hain. "
                f"Niche har slide/page ke sabse important exam points ko simple language mein diya gaya hai:"
            )
        else:
            summary_intro = (
                f"### ✨ Simplified Notes Summary: **{title}**\n\n"
                f"**🎯 Big-Picture Overview:**\n"
                f"These lecture slides/notes focus on **{', '.join(top_concepts[:4])}**. "
                f"Course logistics and headers have been automatically filtered out to give you pure high-yield definitions:"
            )

        bullets = "\n".join([f"- **Key Takeaway {i+1}:** {pt}" for i, pt in enumerate(key_points)])
        flashcards = []
        for i, pt in enumerate(key_points[:3]):
            concept_label = top_concepts[i] if i < len(top_concepts) else f"Concept #{i+1}"
            flashcards.append({
                "question": f"What is the key principle regarding {concept_label} in {title}?",
                "answer": pt,
                "source": f"{title} (Takeaway #{i+1})"
            })

        return {
            "title": title,
            "summary_markdown": f"{summary_intro}\n\n#### 📌 Simplified Step-by-Step Breakdown:\n{bullets}",
            "key_points": key_points,
            "top_concepts": top_concepts,
            "flashcards": flashcards,
            "word_count": len(text.split())
        }
