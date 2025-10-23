#!/usr/bin/env python3
"""
Analysis script for evaluation results
Analyzes metrics by conversation length (short, medium, long)
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict

class ResultsAnalyzer:
    def __init__(self, results_dir: str = "evaluation_results"):
        self.results_dir = results_dir
        self.results = []
        self.load_results()
    
    def load_results(self):
        """Load all result files from the results directory."""
        for filename in os.listdir(self.results_dir):
            if filename.startswith('bench_') and filename.endswith('_result.json'):
                filepath = os.path.join(self.results_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        self.results.append(result)
                        print(f"Loaded {filename}: {result.get('test_id', 'unknown')}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        print(f"\nTotal results loaded: {len(self.results)}")
    
    def group_by_length(self) -> Dict[str, List[Dict]]:
        """Group results by conversation length."""
        groups = defaultdict(list)
        for result in self.results:
            length = result.get('length', 'unknown')
            groups[length].append(result)
        return dict(groups)
    
    def calculate_metrics_by_group(self, group: List[Dict]) -> Dict[str, float]:
        """Calculate metrics for a group of results."""
        if not group:
            return {}
        
        metrics = {
            'count': len(group),
            'passed_rate': sum(1 for r in group if r.get('passed', False)) / len(group),
            'nutrition_valid_rate': sum(1 for r in group if r.get('nutrition_valid', False)) / len(group),
            'avg_user_req_score': np.mean([r.get('user_req_score', 0) for r in group]),
            'avg_context_score': np.mean([r.get('context_score', 0) for r in group]),
            'avg_advanced_score': np.mean([r.get('advanced_score', 0) for r in group]),
            'avg_total_turns': np.mean([r.get('total_turns', 0) for r in group]),
        }
        
        # Advanced evaluation metrics
        advanced_metrics = [
            'no_repeats_valid', 'protein_rotation_valid', 'fasting_window_valid',
            'netcarb_caps_valid', 'sodium_valid', 'ingredient_diversity_valid',
            'enhanced_nutrition_valid', 'tool_usage_valid'
        ]
        
        for metric in advanced_metrics:
            metrics[f'{metric}_rate'] = sum(1 for r in group if r.get(metric, False)) / len(group)
        
        return metrics
    
    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends across conversation lengths."""
        groups = self.group_by_length()
        analysis = {}
        
        print("\n" + "="*60)
        print("CONVERSATION LENGTH ANALYSIS")
        print("="*60)
        
        for length in ['short', 'medium', 'long']:
            if length in groups:
                group = groups[length]
                metrics = self.calculate_metrics_by_group(group)
                analysis[length] = metrics
                
                print(f"\n{length.upper()} CONVERSATIONS ({len(group)} tests):")
                print(f"  Count: {metrics['count']}")
                print(f"  Passed Rate: {metrics['passed_rate']:.2%}")
                print(f"  Nutrition Valid Rate: {metrics['nutrition_valid_rate']:.2%}")
                print(f"  Avg User Req Score: {metrics['avg_user_req_score']:.3f}")
                print(f"  Avg Context Score: {metrics['avg_context_score']:.3f}")
                print(f"  Avg Advanced Score: {metrics['avg_advanced_score']:.3f}")
                print(f"  Avg Total Turns: {metrics['avg_total_turns']:.1f}")
                
                print(f"\n  Advanced Evaluation Rates:")
                for metric in ['no_repeats_valid', 'protein_rotation_valid', 'fasting_window_valid',
                              'netcarb_caps_valid', 'sodium_valid', 'ingredient_diversity_valid',
                              'enhanced_nutrition_valid', 'tool_usage_valid']:
                    rate = metrics.get(f'{metric}_rate', 0)
                    print(f"    {metric.replace('_valid', '')}: {rate:.2%}")
        
        return analysis
    
    def analyze_user_req_vs_context_trends(self):
        """Analyze user_req_score and context_score trends by length."""
        groups = self.group_by_length()
        
        print("\n" + "="*60)
        print("USER REQUIREMENTS vs CONTEXT SCORES TREND ANALYSIS")
        print("="*60)
        
        trend_data = []
        for length in ['short', 'medium', 'long']:
            if length in groups:
                group = groups[length]
                user_req_scores = [r.get('user_req_score', 0) for r in group]
                context_scores = [r.get('context_score', 0) for r in group]
                
                trend_data.append({
                    'length': length,
                    'avg_user_req': np.mean(user_req_scores),
                    'avg_context': np.mean(context_scores),
                    'std_user_req': np.std(user_req_scores),
                    'std_context': np.std(context_scores),
                    'count': len(group)
                })
                
                print(f"\n{length.upper()}:")
                print(f"  User Req Score: {np.mean(user_req_scores):.3f} ± {np.std(user_req_scores):.3f}")
                print(f"  Context Score: {np.mean(context_scores):.3f} ± {np.std(context_scores):.3f}")
                print(f"  Sample size: {len(group)}")
        
        return trend_data
    
    def analyze_advanced_metrics_trends(self):
        """Analyze advanced metrics trends by conversation length."""
        groups = self.group_by_length()
        
        print("\n" + "="*60)
        print("ADVANCED METRICS TREND ANALYSIS")
        print("="*60)
        
        advanced_metrics = [
            'no_repeats_valid', 'protein_rotation_valid', 'fasting_window_valid',
            'netcarb_caps_valid', 'sodium_valid', 'ingredient_diversity_valid',
            'enhanced_nutrition_valid', 'tool_usage_valid'
        ]
        
        for metric in advanced_metrics:
            print(f"\n{metric.replace('_valid', '').replace('_', ' ').title()}:")
            for length in ['short', 'medium', 'long']:
                if length in groups:
                    group = groups[length]
                    rate = sum(1 for r in group if r.get(metric, False)) / len(group)
                    print(f"  {length}: {rate:.2%} ({sum(1 for r in group if r.get(metric, False))}/{len(group)})")
    
    def find_context_weaknesses(self):
        """Identify specific context weaknesses."""
        print("\n" + "="*60)
        print("CONTEXT WEAKNESSES IDENTIFICATION")
        print("="*60)
        
        # Find tests with low context scores
        low_context_tests = [r for r in self.results if r.get('context_score', 0) < 0.5]
        print(f"\nTests with low context scores (< 0.5): {len(low_context_tests)}")
        
        for test in low_context_tests[:5]:  # Show first 5
            print(f"\n  {test['test_id']} ({test.get('length', 'unknown')}):")
            print(f"    Context Score: {test.get('context_score', 0):.3f}")
            print(f"    User Req Score: {test.get('user_req_score', 0):.3f}")
            print(f"    Question: {test.get('question', 'N/A')[:100]}...")
            if test.get('context_dependencies'):
                print(f"    Context Dependencies: {test['context_dependencies']}")
        
        # Find tests with inter-session dependencies
        inter_session_tests = [r for r in self.results 
                               if r.get('memory_dependencies', {}).get('inter_session')]
        print(f"\nTests with inter-session dependencies: {len(inter_session_tests)}")
        
        for test in inter_session_tests[:3]:  # Show first 3
            print(f"\n  {test['test_id']} ({test.get('length', 'unknown')}):")
            print(f"    Context Score: {test.get('context_score', 0):.3f}")
            print(f"    Inter-session deps: {test.get('memory_dependencies', {}).get('inter_session', [])}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*80)
        
        # Overall statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.get('passed', False))
        nutrition_valid = sum(1 for r in self.results if r.get('nutrition_valid', False))
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed Tests: {passed_tests} ({passed_tests/total_tests:.1%})")
        print(f"  Nutrition Valid: {nutrition_valid} ({nutrition_valid/total_tests:.1%})")
        
        # Average scores
        avg_user_req = np.mean([r.get('user_req_score', 0) for r in self.results])
        avg_context = np.mean([r.get('context_score', 0) for r in self.results])
        avg_advanced = np.mean([r.get('advanced_score', 0) for r in self.results])
        
        print(f"\nAVERAGE SCORES:")
        print(f"  User Requirements: {avg_user_req:.3f}")
        print(f"  Context Handling: {avg_context:.3f}")
        print(f"  Advanced Evaluation: {avg_advanced:.3f}")
        
        # Length-based analysis
        self.analyze_trends()
        self.analyze_user_req_vs_context_trends()
        self.analyze_advanced_metrics_trends()
        self.find_context_weaknesses()
    
    def save_analysis_to_file(self, output_file: str = "analysis_report.txt"):
        """Save analysis to a text file."""
        import sys
        from io import StringIO
        
        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            self.generate_summary_report()
        finally:
            sys.stdout = old_stdout
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(captured_output.getvalue())
        
        print(f"\nAnalysis report saved to: {output_file}")

def main():
    """Main analysis function."""
    print("Starting evaluation results analysis...")
    
    analyzer = ResultsAnalyzer()
    
    if not analyzer.results:
        print("No results found in evaluation_results directory!")
        return
    
    # Generate comprehensive analysis
    analyzer.generate_summary_report()
    
    # Save analysis to file
    analyzer.save_analysis_to_file()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
