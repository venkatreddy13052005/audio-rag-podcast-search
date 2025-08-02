"""
Evaluation module for measuring retrieval performance and system metrics.
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


class EvaluationMetrics:
    """Class for computing various evaluation metrics for the RAG system."""
    
    def __init__(self):
        """Initialize evaluation metrics."""
        self.search_times = []
        self.relevance_scores = []
        self.precision_at_k_scores = []
        self.recall_scores = []
    
    def measure_search_latency(self, search_function, query: str, *args, **kwargs) -> Tuple[Any, float]:
        """
        Measure the latency of a search function.
        
        Args:
            search_function: Function to measure
            query: Search query
            *args, **kwargs: Additional arguments for the search function
            
        Returns:
            Tuple of (search_results, latency_seconds)
        """
        start_time = time.time()
        results = search_function(query, *args, **kwargs)
        end_time = time.time()
        
        latency = end_time - start_time
        self.search_times.append(latency)
        
        return results, latency
    
    def calculate_relevance_score(self, query: str, results: List[Dict], 
                                embedding_model) -> List[float]:
        """
        Calculate relevance scores between query and search results.
        
        Args:
            query: Original search query
            results: List of search results
            embedding_model: Model to generate embeddings
            
        Returns:
            List of relevance scores (0-1)
        """
        if not results:
            return []
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query])
        
        # Generate result embeddings
        result_texts = [result["text"] for result in results]
        result_embeddings = embedding_model.encode(result_texts)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, result_embeddings)[0]
        
        self.relevance_scores.extend(similarities)
        return similarities.tolist()
    
    def calculate_precision_at_k(self, relevant_results: List[bool], k: int = 10) -> float:
        """
        Calculate Precision@K metric.
        
        Args:
            relevant_results: Boolean list indicating relevance of each result
            k: Number of top results to consider
            
        Returns:
            Precision@K score
        """
        if not relevant_results or k <= 0:
            return 0.0
        
        top_k_results = relevant_results[:k]
        precision = sum(top_k_results) / len(top_k_results)
        
        self.precision_at_k_scores.append(precision)
        return precision
    
    def calculate_recall_at_k(self, relevant_results: List[bool], 
                            total_relevant: int, k: int = 10) -> float:
        """
        Calculate Recall@K metric.
        
        Args:
            relevant_results: Boolean list indicating relevance of each result
            total_relevant: Total number of relevant documents in the collection
            k: Number of top results to consider
            
        Returns:
            Recall@K score
        """
        if not relevant_results or k <= 0 or total_relevant <= 0:
            return 0.0
        
        top_k_results = relevant_results[:k]
        recall = sum(top_k_results) / total_relevant
        
        self.recall_scores.append(recall)
        return recall
    
    def calculate_mrr(self, relevant_results: List[List[bool]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            relevant_results: List of boolean lists for multiple queries
            
        Returns:
            MRR score
        """
        if not relevant_results:
            return 0.0
        
        reciprocal_ranks = []
        
        for query_results in relevant_results:
            # Find the rank of the first relevant result
            first_relevant_rank = None
            for i, is_relevant in enumerate(query_results):
                if is_relevant:
                    first_relevant_rank = i + 1  # Rank is 1-indexed
                    break
            
            if first_relevant_rank:
                reciprocal_ranks.append(1.0 / first_relevant_rank)
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks)
    
    def calculate_ndcg(self, relevance_scores: List[float], k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            relevance_scores: List of relevance scores (0-1)
            k: Number of top results to consider
            
        Returns:
            NDCG@K score
        """
        if not relevance_scores or k <= 0:
            return 0.0
        
        # Take top-k scores
        scores = relevance_scores[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, score in enumerate(scores):
            dcg += score / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        ideal_scores = sorted(relevance_scores, reverse=True)[:k]
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            idcg += score / np.log2(i + 2)
        
        # Return NDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_chunk_coverage(self, query: str, results: List[Dict], 
                              episode_title: str = None) -> Dict:
        """
        Evaluate how well the results cover the query topic across time.
        
        Args:
            query: Search query
            results: Search results
            episode_title: Optional episode filter
            
        Returns:
            Coverage metrics
        """
        if not results:
            return {"temporal_coverage": 0, "episode_coverage": 0, "speaker_coverage": 0}
        
        # Filter by episode if specified
        if episode_title:
            results = [r for r in results if r["metadata"]["episode_title"] == episode_title]
        
        if not results:
            return {"temporal_coverage": 0, "episode_coverage": 0, "speaker_coverage": 0}
        
        # Temporal coverage: span of time covered by results
        start_times = [r["metadata"]["start_time"] for r in results]
        end_times = [r["metadata"]["end_time"] for r in results]
        temporal_span = max(end_times) - min(start_times)
        
        # Episode coverage: number of unique episodes
        episodes = set([r["metadata"]["episode_title"] for r in results])
        episode_coverage = len(episodes)
        
        # Speaker coverage: number of unique speakers
        all_speakers = set()
        for result in results:
            speakers = result["metadata"].get("speakers", [])
            if isinstance(speakers, list):
                all_speakers.update(speakers)
        speaker_coverage = len(all_speakers)
        
        return {
            "temporal_coverage": temporal_span,
            "episode_coverage": episode_coverage,
            "speaker_coverage": speaker_coverage,
            "result_count": len(results)
        }
    
    def get_performance_summary(self) -> Dict:
        """Get summary of all collected performance metrics."""
        summary = {
            "latency_stats": {
                "mean": np.mean(self.search_times) if self.search_times else 0,
                "median": np.median(self.search_times) if self.search_times else 0,
                "std": np.std(self.search_times) if self.search_times else 0,
                "min": min(self.search_times) if self.search_times else 0,
                "max": max(self.search_times) if self.search_times else 0,
                "count": len(self.search_times)
            },
            "relevance_stats": {
                "mean": np.mean(self.relevance_scores) if self.relevance_scores else 0,
                "median": np.median(self.relevance_scores) if self.relevance_scores else 0,
                "std": np.std(self.relevance_scores) if self.relevance_scores else 0,
                "min": min(self.relevance_scores) if self.relevance_scores else 0,
                "max": max(self.relevance_scores) if self.relevance_scores else 0,
                "count": len(self.relevance_scores)
            },
            "precision_at_k_stats": {
                "mean": np.mean(self.precision_at_k_scores) if self.precision_at_k_scores else 0,
                "count": len(self.precision_at_k_scores)
            },
            "recall_stats": {
                "mean": np.mean(self.recall_scores) if self.recall_scores else 0,
                "count": len(self.recall_scores)
            }
        }
        
        return summary


class BenchmarkSuite:
    """Comprehensive benchmark suite for the podcast RAG system."""
    
    def __init__(self, retrieval_system, text_processor):
        """
        Initialize benchmark suite.
        
        Args:
            retrieval_system: The retrieval system to benchmark
            text_processor: Text processor with embedding model
        """
        self.retrieval_system = retrieval_system
        self.text_processor = text_processor
        self.metrics = EvaluationMetrics()
    
    def create_test_queries(self) -> List[Dict]:
        """Create a set of test queries for evaluation."""
        test_queries = [
            {
                "query": "artificial intelligence and machine learning",
                "category": "technology",
                "expected_topics": ["AI", "ML", "algorithms", "neural networks"]
            },
            {
                "query": "climate change and environmental policy",
                "category": "environment",
                "expected_topics": ["climate", "environment", "policy", "sustainability"]
            },
            {
                "query": "entrepreneurship and startup advice",
                "category": "business",
                "expected_topics": ["startup", "business", "entrepreneur", "funding"]
            },
            {
                "query": "health and wellness tips",
                "category": "health",
                "expected_topics": ["health", "wellness", "fitness", "nutrition"]
            },
            {
                "query": "financial planning and investment strategies",
                "category": "finance",
                "expected_topics": ["finance", "investment", "planning", "money"]
            }
        ]
        
        return test_queries
    
    def run_latency_benchmark(self, test_queries: List[Dict], 
                            n_iterations: int = 5) -> Dict:
        """
        Run latency benchmark across multiple queries and iterations.
        
        Args:
            test_queries: List of test queries
            n_iterations: Number of iterations per query
            
        Returns:
            Latency benchmark results
        """
        print("Running latency benchmark...")
        
        latency_results = []
        
        for query_data in test_queries:
            query = query_data["query"]
            query_latencies = []
            
            for i in range(n_iterations):
                results, latency = self.metrics.measure_search_latency(
                    self.retrieval_system.vector_store.search_by_text,
                    query,
                    self.text_processor.embedding_model,
                    10  # n_results
                )
                query_latencies.append(latency)
            
            latency_results.append({
                "query": query,
                "category": query_data["category"],
                "mean_latency": np.mean(query_latencies),
                "std_latency": np.std(query_latencies),
                "min_latency": min(query_latencies),
                "max_latency": max(query_latencies)
            })
        
        return latency_results
    
    def run_relevance_benchmark(self, test_queries: List[Dict]) -> Dict:
        """
        Run relevance benchmark to evaluate search quality.
        
        Args:
            test_queries: List of test queries
            
        Returns:
            Relevance benchmark results
        """
        print("Running relevance benchmark...")
        
        relevance_results = []
        
        for query_data in test_queries:
            query = query_data["query"]
            expected_topics = query_data["expected_topics"]
            
            # Perform search
            results = self.retrieval_system.vector_store.search_by_text(
                query, self.text_processor.embedding_model, 10
            )
            
            # Calculate relevance scores
            relevance_scores = self.metrics.calculate_relevance_score(
                query, results, self.text_processor.embedding_model
            )
            
            # Evaluate topic coverage
            topic_coverage = self._evaluate_topic_coverage(results, expected_topics)
            
            # Calculate coverage metrics
            coverage_metrics = self.metrics.evaluate_chunk_coverage(query, results)
            
            relevance_results.append({
                "query": query,
                "category": query_data["category"],
                "mean_relevance": np.mean(relevance_scores) if relevance_scores else 0,
                "topic_coverage": topic_coverage,
                "coverage_metrics": coverage_metrics,
                "result_count": len(results)
            })
        
        return relevance_results
    
    def _evaluate_topic_coverage(self, results: List[Dict], 
                                expected_topics: List[str]) -> float:
        """
        Evaluate how well results cover expected topics.
        
        Args:
            results: Search results
            expected_topics: List of expected topic keywords
            
        Returns:
            Topic coverage score (0-1)
        """
        if not results or not expected_topics:
            return 0.0
        
        # Combine all result texts
        all_text = " ".join([result["text"].lower() for result in results])
        
        # Count how many expected topics are mentioned
        topics_found = 0
        for topic in expected_topics:
            if topic.lower() in all_text:
                topics_found += 1
        
        return topics_found / len(expected_topics)
    
    def run_scalability_test(self, query: str, result_counts: List[int]) -> Dict:
        """
        Test how latency scales with number of results requested.
        
        Args:
            query: Test query
            result_counts: List of result counts to test
            
        Returns:
            Scalability test results
        """
        print("Running scalability test...")
        
        scalability_results = []
        
        for n_results in result_counts:
            results, latency = self.metrics.measure_search_latency(
                self.retrieval_system.vector_store.search_by_text,
                query,
                self.text_processor.embedding_model,
                n_results
            )
            
            scalability_results.append({
                "n_results": n_results,
                "latency": latency,
                "actual_results": len(results)
            })
        
        return scalability_results
    
    def run_comprehensive_benchmark(self) -> Dict:
        """
        Run comprehensive benchmark suite.
        
        Returns:
            Complete benchmark results
        """
        print("Starting comprehensive benchmark...")
        
        # Create test queries
        test_queries = self.create_test_queries()
        
        # Run benchmarks
        latency_results = self.run_latency_benchmark(test_queries)
        relevance_results = self.run_relevance_benchmark(test_queries)
        scalability_results = self.run_scalability_test(
            "technology and innovation", [5, 10, 20, 50, 100]
        )
        
        # Get performance summary
        performance_summary = self.metrics.get_performance_summary()
        
        # Compile results
        benchmark_results = {
            "timestamp": time.time(),
            "latency_benchmark": latency_results,
            "relevance_benchmark": relevance_results,
            "scalability_benchmark": scalability_results,
            "performance_summary": performance_summary,
            "test_queries": test_queries
        }
        
        return benchmark_results
    
    def save_benchmark_results(self, results: Dict, filepath: str):
        """Save benchmark results to a JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Benchmark results saved to {filepath}")
        except Exception as e:
            print(f"Error saving benchmark results: {e}")
    
    def generate_benchmark_report(self, results: Dict) -> str:
        """
        Generate a human-readable benchmark report.
        
        Args:
            results: Benchmark results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=== PODCAST RAG SYSTEM BENCHMARK REPORT ===\n")
        
        # Performance Summary
        perf = results["performance_summary"]
        report.append("PERFORMANCE SUMMARY:")
        report.append(f"  Average Search Latency: {perf['latency_stats']['mean']:.3f}s")
        report.append(f"  Average Relevance Score: {perf['relevance_stats']['mean']:.3f}")
        report.append(f"  Total Searches Performed: {perf['latency_stats']['count']}")
        report.append("")
        
        # Latency Benchmark
        report.append("LATENCY BENCHMARK:")
        for result in results["latency_benchmark"]:
            report.append(f"  {result['category'].upper()}: {result['mean_latency']:.3f}s (±{result['std_latency']:.3f})")
        report.append("")
        
        # Relevance Benchmark
        report.append("RELEVANCE BENCHMARK:")
        for result in results["relevance_benchmark"]:
            report.append(f"  {result['category'].upper()}:")
            report.append(f"    Relevance Score: {result['mean_relevance']:.3f}")
            report.append(f"    Topic Coverage: {result['topic_coverage']:.3f}")
            report.append(f"    Results Found: {result['result_count']}")
        report.append("")
        
        # Scalability Test
        report.append("SCALABILITY TEST:")
        for result in results["scalability_benchmark"]:
            report.append(f"  {result['n_results']} results: {result['latency']:.3f}s")
        report.append("")
        
        return "\n".join(report)


def create_sample_evaluation_data() -> List[Dict]:
    """Create sample data for evaluation purposes."""
    sample_data = [
        {
            "query": "machine learning algorithms",
            "relevant_chunks": [
                "Today we're discussing various machine learning algorithms and their applications in real-world scenarios.",
                "Neural networks and deep learning have revolutionized the field of artificial intelligence.",
                "Understanding different ML algorithms is crucial for data scientists."
            ],
            "irrelevant_chunks": [
                "The weather has been quite nice this week.",
                "I had a great lunch at a new restaurant yesterday.",
                "Sports news and recent game results."
            ]
        }
    ]
    
    return sample_data