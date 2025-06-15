"""
Neural Odyssey - Week 27: Natural Language Processing - Transformers in Action
Phase 3: Advanced Topics and Modern AI (Week 3)

From Architecture to Application: Transformers Revolutionize Language

Welcome to the pinnacle of NLP! This week, you'll witness transformers in their natural 
habitat - solving real-world language problems that seemed impossible just a few years ago. 
From BERT's bidirectional understanding to GPT's creative generation, from T5's unified 
framework to modern instruction-following models.

You'll not just learn about these systems - you'll build them, train them, evaluate them, 
and deploy them. Experience the complete journey from research breakthrough to production 
system, understanding every nuance that makes modern NLP work.

Journey Overview:
1. Pre-training paradigms: MLM, autoregressive, and text-to-text frameworks
2. Language understanding: BERT-style classification, NER, and question answering
3. Language generation: GPT-style autoregressive modeling and controlled generation
4. Advanced techniques: In-context learning, PEFT methods, and instruction tuning
5. Safety and alignment: RLHF, bias detection, and responsible AI deployment
6. Production systems: Optimization, serving, and monitoring at scale
7. Research frontiers: Multimodal models, tool use, and emerging capabilities
8. Real-world applications: Building systems that actually work

By week's end, you'll have mastered the complete NLP stack that powers ChatGPT, 
Google Translate, and every other modern language AI system.

Author: Neural Explorer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Union, Any
import math
import json
import re
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import time
import warnings
warnings.filterwarnings('ignore')

# For NLP-specific functionality
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        AutoModelForTokenClassification, AutoModelForQuestionAnswering,
        AutoModelForCausalLM, AutoModelForSeq2SeqLM,
        BertTokenizer, BertForSequenceClassification,
        GPT2Tokenizer, GPT2LMHeadModel,
        T5Tokenizer, T5ForConditionalGeneration,
        Trainer, TrainingArguments,
        pipeline
    )
    from datasets import Dataset, DatasetDict
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ Transformers library not available. Some features will be simulated.")
    TRANSFORMERS_AVAILABLE = False

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("🎯 Neural Odyssey - Week 27: Natural Language Processing")
print("=" * 60)
print("🗣️ Transformers in Action - From Research to Production")
print("🚀 Building the NLP systems that power modern AI!")
print("=" * 60)


# ==========================================
# SECTION 1: NLP FOUNDATIONS AND TOKENIZATION
# ==========================================

class TokenizationDemo:
    """
    Comprehensive demonstration of tokenization strategies for NLP
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
    
    def demonstrate_tokenization_strategies(self):
        """
        Compare different tokenization approaches
        """
        print("\n📝 Tokenization Strategies for Modern NLP")
        print("=" * 50)
        
        # Sample texts for demonstration
        sample_texts = [
            "Hello, world! How are you doing today?",
            "The transformer architecture revolutionized NLP in 2017.",
            "COVID-19 and AI/ML are trending topics in 2023.",
            "She's working on state-of-the-art models.",
            "Subword tokenization handles out-of-vocabulary words effectively."
        ]
        
        print("🔤 Tokenization Strategy Comparison:")
        print()
        
        for i, text in enumerate(sample_texts):
            print(f"Text {i+1}: '{text}'")
            
            # 1. Word-level tokenization (baseline)
            word_tokens = text.lower().split()
            print(f"   Word-level: {word_tokens}")
            print(f"   Count: {len(word_tokens)} tokens")
            
            # 2. Character-level tokenization
            char_tokens = list(text.lower().replace(' ', '_'))
            print(f"   Character-level: {char_tokens[:20]}{'...' if len(char_tokens) > 20 else ''}")
            print(f"   Count: {len(char_tokens)} tokens")
            
            # 3. Simple subword tokenization (simulate BPE)
            subword_tokens = self.simple_subword_tokenization(text)
            print(f"   Subword (simulated): {subword_tokens}")
            print(f"   Count: {len(subword_tokens)} tokens")
            
            if TRANSFORMERS_AVAILABLE:
                # 4. BERT tokenization
                try:
                    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                    bert_tokens = bert_tokenizer.tokenize(text.lower())
                    print(f"   BERT tokenization: {bert_tokens}")
                    print(f"   Count: {len(bert_tokens)} tokens")
                except:
                    print("   BERT tokenization: Not available")
                
                # 5. GPT-2 tokenization
                try:
                    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                    gpt2_tokens = gpt2_tokenizer.tokenize(text)
                    print(f"   GPT-2 tokenization: {gpt2_tokens}")
                    print(f"   Count: {len(gpt2_tokens)} tokens")
                except:
                    print("   GPT-2 tokenization: Not available")
            
            print()
        
        # Analyze tokenization properties
        self.analyze_tokenization_properties()
        
        return sample_texts
    
    def simple_subword_tokenization(self, text):
        """
        Simulate simple subword tokenization (BPE-like)
        """
        # Simple heuristic subword splitting
        words = text.lower().split()
        subwords = []
        
        for word in words:
            # Remove punctuation for simplicity
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) <= 4:
                subwords.append(clean_word)
            else:
                # Split longer words into subwords
                subwords.extend([clean_word[:3] + '##', '##' + clean_word[3:]])
        
        return subwords
    
    def analyze_tokenization_properties(self):
        """
        Analyze properties of different tokenization strategies
        """
        print("🔍 Tokenization Strategy Analysis:")
        
        properties = {
            'Word-level': {
                'vocabulary_size': 'Large (50K-100K+)',
                'oov_handling': 'Poor (unknown words)',
                'subword_info': 'None',
                'memory': 'High',
                'use_cases': 'Simple tasks, limited vocabulary'
            },
            'Character-level': {
                'vocabulary_size': 'Small (~100)',
                'oov_handling': 'Perfect (any text)',
                'subword_info': 'Maximum',
                'memory': 'Low',
                'use_cases': 'Morphologically rich languages, noise handling'
            },
            'Subword (BPE/WordPiece)': {
                'vocabulary_size': 'Medium (30K)',
                'oov_handling': 'Good (decomposition)',
                'subword_info': 'Balanced',
                'memory': 'Medium',
                'use_cases': 'Modern NLP, multilingual models'
            },
            'SentencePiece': {
                'vocabulary_size': 'Medium (32K)',
                'oov_handling': 'Excellent',
                'subword_info': 'Optimal',
                'memory': 'Medium',
                'use_cases': 'Multilingual, no preprocessing'
            }
        }
        
        for strategy, props in properties.items():
            print(f"\n   📊 {strategy}:")
            for prop_name, prop_value in props.items():
                print(f"      {prop_name}: {prop_value}")
        
        # Visualize comparison
        self.visualize_tokenization_comparison()
    
    def visualize_tokenization_comparison(self):
        """
        Create visualizations comparing tokenization strategies
        """
        plt.figure(figsize=(15, 10))
        
        # Sample analysis data
        strategies = ['Word-level', 'Character-level', 'BPE', 'WordPiece', 'SentencePiece']
        vocab_sizes = [75000, 128, 30000, 30000, 32000]
        oov_handling = [2, 10, 8, 8, 9]  # Score out of 10
        efficiency = [6, 4, 9, 9, 9]  # Score out of 10
        
        # Plot 1: Vocabulary size comparison
        plt.subplot(2, 3, 1)
        bars = plt.bar(strategies, vocab_sizes, alpha=0.7, 
                      color=['red', 'blue', 'green', 'orange', 'purple'])
        plt.title('Vocabulary Size Comparison', fontweight='bold')
        plt.ylabel('Vocabulary Size')
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, size in zip(bars, vocab_sizes):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{size:,}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: OOV handling capability
        plt.subplot(2, 3, 2)
        plt.bar(strategies, oov_handling, alpha=0.7, color='lightcoral')
        plt.title('Out-of-Vocabulary Handling', fontweight='bold')
        plt.ylabel('Capability Score (1-10)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 10)
        
        # Plot 3: Efficiency comparison
        plt.subplot(2, 3, 3)
        plt.bar(strategies, efficiency, alpha=0.7, color='lightgreen')
        plt.title('Computational Efficiency', fontweight='bold')
        plt.ylabel('Efficiency Score (1-10)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 10)
        
        # Plot 4: Token length distribution (simulated)
        plt.subplot(2, 3, 4)
        text_sample = "The quick brown fox jumps over the lazy dog. Natural language processing is amazing!"
        
        word_lengths = [len(word) for word in text_sample.split()]
        char_lengths = [1] * len(text_sample.replace(' ', ''))
        subword_lengths = [len(token) for token in self.simple_subword_tokenization(text_sample)]
        
        plt.hist([word_lengths, subword_lengths], bins=10, alpha=0.7, 
                label=['Word-level', 'Subword'], color=['blue', 'green'])
        plt.title('Token Length Distribution', fontweight='bold')
        plt.xlabel('Token Length (characters)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Compression ratio
        plt.subplot(2, 3, 5)
        original_chars = len(text_sample)
        word_tokens = len(text_sample.split())
        char_tokens = len(text_sample.replace(' ', ''))
        subword_tokens = len(self.simple_subword_tokenization(text_sample))
        
        compression_ratios = [
            original_chars / word_tokens,
            original_chars / char_tokens,
            original_chars / subword_tokens
        ]
        
        plt.bar(['Word', 'Char', 'Subword'], compression_ratios, 
               alpha=0.7, color=['red', 'blue', 'green'])
        plt.title('Compression Ratio\n(Chars per Token)', fontweight='bold')
        plt.ylabel('Characters per Token')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Use case matrix
        plt.subplot(2, 3, 6)
        use_cases = ['Multilingual', 'Low Resource', 'Fast Inference', 'Rich Morphology']
        strategy_scores = np.array([
            [3, 2, 8, 2],  # Word-level
            [8, 9, 3, 9],  # Character-level
            [9, 7, 7, 7],  # BPE
            [9, 7, 7, 7],  # WordPiece
            [10, 8, 7, 8]  # SentencePiece
        ])
        
        sns.heatmap(strategy_scores, 
                   xticklabels=use_cases,
                   yticklabels=strategies,
                   annot=True, cmap='RdYlGn', 
                   cbar_kws={'label': 'Suitability Score'})
        plt.title('Use Case Suitability Matrix', fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.show()
        
        print("📈 Key Takeaways from Tokenization Analysis:")
        print("   • Subword tokenization (BPE/WordPiece) offers best balance")
        print("   • Character-level handles any text but loses semantic info")
        print("   • Word-level simple but struggles with OOV and morphology")
        print("   • SentencePiece best for multilingual and preprocessing-free pipelines")
        print("   • Choice depends on language, domain, and computational constraints")


# ==========================================
# SECTION 2: BERT-STYLE LANGUAGE UNDERSTANDING
# ==========================================

class BERTLanguageUnderstanding:
    """
    Comprehensive demonstration of BERT-style language understanding tasks
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        
    def demonstrate_bert_architecture(self):
        """
        Demonstrate BERT architecture and masked language modeling
        """
        print("\n🤖 BERT: Bidirectional Encoder Representations from Transformers")
        print("=" * 60)
        
        print("🧠 BERT Innovation: Bidirectional Training")
        print("   Traditional: Left-to-right or right-to-left")
        print("   BERT: Bidirectional context from both directions")
        print("   Method: Masked Language Modeling (MLM)")
        print()
        
        # Demonstrate masked language modeling concept
        self.demonstrate_masked_language_modeling()
        
        # Demonstrate BERT fine-tuning for different tasks
        self.demonstrate_bert_fine_tuning()
    
    def demonstrate_masked_language_modeling(self):
        """
        Demonstrate the core MLM training objective
        """
        print("🎭 Masked Language Modeling (MLM) Demonstration:")
        
        # Example sentences for MLM
        sentences = [
            "The capital of France is [MASK].",
            "Machine learning is a subset of [MASK] intelligence.",
            "The [MASK] revolution in NLP started with transformers.",
            "BERT uses [MASK] attention to understand context.",
            "Fine-tuning adapts [MASK] models to specific tasks."
        ]
        
        # Expected answers (for demonstration)
        expected_answers = ["Paris", "artificial", "transformer", "self", "pre-trained"]
        
        print("   Example MLM predictions:")
        for sentence, answer in zip(sentences, expected_answers):
            print(f"   Input:  {sentence}")
            print(f"   Target: {answer}")
            
            if TRANSFORMERS_AVAILABLE:
                try:
                    # Use a pre-trained BERT model for actual prediction
                    mlm_pipeline = pipeline("fill-mask", model="bert-base-uncased")
                    predictions = mlm_pipeline(sentence.replace("[MASK]", "<mask>"))
                    top_pred = predictions[0]['token_str']
                    confidence = predictions[0]['score']
                    print(f"   BERT:   {top_pred} (confidence: {confidence:.3f})")
                except:
                    print(f"   BERT:   [Simulated prediction: {answer}]")
            else:
                print(f"   BERT:   [Simulated prediction: {answer}]")
            print()
        
        # Explain MLM training process
        print("🔍 MLM Training Process:")
        print("   1. Randomly mask 15% of tokens in input")
        print("   2. 80% replaced with [MASK], 10% random token, 10% unchanged")
        print("   3. Predict original token for masked positions")
        print("   4. Loss only computed on masked tokens")
        print("   5. Bidirectional context enables rich representations")
        
        # Visualize attention patterns
        self.visualize_bert_attention()
    
    def visualize_bert_attention(self):
        """
        Visualize BERT attention patterns
        """
        print("\n👁️ BERT Attention Pattern Analysis:")
        
        # Simulate attention patterns (in practice, extract from actual BERT)
        sentence = "The cat sat on the mat"
        tokens = sentence.split()
        seq_len = len(tokens)
        
        # Create realistic attention patterns
        np.random.seed(42)
        
        # Different heads show different patterns
        head_patterns = {
            'Head 1 (Syntactic)': self.create_syntactic_attention(seq_len),
            'Head 2 (Semantic)': self.create_semantic_attention(seq_len),
            'Head 3 (Positional)': self.create_positional_attention(seq_len),
            'Head 4 (Global)': self.create_global_attention(seq_len)
        }
        
        plt.figure(figsize=(20, 12))
        
        for i, (head_name, pattern) in enumerate(head_patterns.items()):
            plt.subplot(2, 2, i + 1)
            sns.heatmap(pattern, 
                       xticklabels=tokens,
                       yticklabels=tokens,
                       annot=True, fmt='.2f',
                       cmap='Blues', cbar=True)
            plt.title(f'{head_name}', fontweight='bold')
            plt.xlabel('Key')
            plt.ylabel('Query')
        
        plt.tight_layout()
        plt.show()
        
        print("   Attention Pattern Interpretation:")
        print("   • Syntactic heads focus on grammatical relationships")
        print("   • Semantic heads attend to related meanings") 
        print("   • Positional heads emphasize nearby tokens")
        print("   • Global heads distribute attention broadly")
    
    def create_syntactic_attention(self, seq_len):
        """Create syntactic attention pattern"""
        pattern = np.eye(seq_len) * 0.3  # Self-attention baseline
        # Add syntactic relationships
        for i in range(seq_len - 1):
            pattern[i, i + 1] = 0.4  # Next word
            pattern[i + 1, i] = 0.3  # Previous word
        # Normalize
        return pattern / pattern.sum(axis=1, keepdims=True)
    
    def create_semantic_attention(self, seq_len):
        """Create semantic attention pattern"""
        pattern = np.random.rand(seq_len, seq_len) * 0.1
        # Stronger connections for semantically related words
        pattern[0, 1] = 0.6  # "The cat"
        pattern[2, 5] = 0.5  # "sat" -> "mat"
        pattern[1, 5] = 0.4  # "cat" -> "mat"
        # Normalize
        return pattern / pattern.sum(axis=1, keepdims=True)
    
    def create_positional_attention(self, seq_len):
        """Create positional attention pattern"""
        pattern = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                # Gaussian decay based on distance
                distance = abs(i - j)
                pattern[i, j] = np.exp(-distance**2 / 2)
        # Normalize
        return pattern / pattern.sum(axis=1, keepdims=True)
    
    def create_global_attention(self, seq_len):
        """Create global attention pattern"""
        pattern = np.ones((seq_len, seq_len)) / seq_len
        # Slightly higher self-attention
        np.fill_diagonal(pattern, 0.3)
        # Normalize
        return pattern / pattern.sum(axis=1, keepdims=True)
    
    def demonstrate_bert_fine_tuning(self):
        """
        Demonstrate BERT fine-tuning for different NLP tasks
        """
        print("\n🎯 BERT Fine-tuning for Different Tasks:")
        
        tasks = {
            'Sentiment Analysis': {
                'description': 'Classify text sentiment (positive/negative/neutral)',
                'architecture': 'BERT + classification head',
                'example_input': 'I love this movie!',
                'example_output': 'Positive (0.89 confidence)',
                'training_data': 'Labeled sentiment datasets (IMDB, Stanford Sentiment)'
            },
            'Named Entity Recognition': {
                'description': 'Identify and classify named entities',
                'architecture': 'BERT + token classification head',
                'example_input': 'John works at Google in California',
                'example_output': 'John[PER] works at Google[ORG] in California[LOC]',
                'training_data': 'CoNLL-2003, OntoNotes 5.0'
            },
            'Question Answering': {
                'description': 'Extract answer spans from context',
                'architecture': 'BERT + span prediction heads',
                'example_input': 'Context: ... Question: When was BERT released?',
                'example_output': 'Span: "2018" (start:45, end:49)',
                'training_data': 'SQuAD 1.1/2.0, Natural Questions'
            },
            'Natural Language Inference': {
                'description': 'Determine entailment between sentence pairs',
                'architecture': 'BERT + classification head for pairs',
                'example_input': 'Premise: A man is walking. Hypothesis: A person is moving.',
                'example_output': 'Entailment (0.92 confidence)',
                'training_data': 'SNLI, MultiNLI, XNLI'
            }
        }
        
        for task_name, task_info in tasks.items():
            print(f"\n   📋 {task_name}:")
            print(f"      Description: {task_info['description']}")
            print(f"      Architecture: {task_info['architecture']}")
            print(f"      Example Input: {task_info['example_input']}")
            print(f"      Example Output: {task_info['example_output']}")
            print(f"      Training Data: {task_info['training_data']}")
        
        # Demonstrate fine-tuning process
        self.demonstrate_fine_tuning_process()
    
    def demonstrate_fine_tuning_process(self):
        """
        Show the fine-tuning process step by step
        """
        print("\n🔧 BERT Fine-tuning Process:")
        
        steps = [
            "1. Load pre-trained BERT model and tokenizer",
            "2. Add task-specific head (classification/token classification/span prediction)",
            "3. Prepare task-specific dataset with proper formatting",
            "4. Set learning rate (typically 2e-5 to 5e-5)",
            "5. Fine-tune for 2-4 epochs to avoid catastrophic forgetting",
            "6. Use linear learning rate decay and warmup",
            "7. Evaluate on held-out test set",
            "8. Apply early stopping based on validation performance"
        ]
        
        for step in steps:
            print(f"   {step}")
        
        print("\n   💡 Fine-tuning Best Practices:")
        best_practices = [
            "Use lower learning rates than pre-training",
            "Apply gradual unfreezing (optional)",
            "Monitor for catastrophic forgetting",
            "Use task-appropriate evaluation metrics",
            "Consider layer-wise learning rate decay",
            "Implement proper regularization (dropout, weight decay)"
        ]
        
        for practice in best_practices:
            print(f"      • {practice}")


# ==========================================
# SECTION 3: GPT-STYLE LANGUAGE GENERATION
# ==========================================

class GPTLanguageGeneration:
    """
    Comprehensive demonstration of GPT-style autoregressive language generation
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def demonstrate_gpt_architecture(self):
        """
        Demonstrate GPT architecture and autoregressive generation
        """
        print("\n🚀 GPT: Generative Pre-trained Transformer")
        print("=" * 50)
        
        print("🔄 GPT Innovation: Autoregressive Generation")
        print("   Approach: Predict next token given previous tokens")
        print("   Training: Maximize likelihood of next token")
        print("   Generation: Sequential token-by-token prediction")
        print("   Strength: Natural generation, in-context learning")
        print()
        
        # Demonstrate autoregressive generation
        self.demonstrate_autoregressive_generation()
        
        # Show different generation strategies
        self.demonstrate_generation_strategies()
        
        # Explore in-context learning
        self.demonstrate_in_context_learning()
    
    def demonstrate_autoregressive_generation(self):
        """
        Show step-by-step autoregressive generation process
        """
        print("🔮 Autoregressive Generation Process:")
        
        # Example generation sequence
        prompt = "The future of artificial intelligence"
        
        # Simulate step-by-step generation
        generation_steps = [
            ("The future of artificial intelligence", "is"),
            ("The future of artificial intelligence is", "likely"),
            ("The future of artificial intelligence is likely", "to"),
            ("The future of artificial intelligence is likely to", "be"),
            ("The future of artificial intelligence is likely to be", "transformative"),
            ("The future of artificial intelligence is likely to be transformative", "for"),
            ("The future of artificial intelligence is likely to be transformative for", "humanity"),
            ("The future of artificial intelligence is likely to be transformative for humanity", ".")
        ]
        
        print(f"   Starting prompt: '{prompt}'")
        print()
        
        for i, (context, next_token) in enumerate(generation_steps):
            print(f"   Step {i+1}:")
            print(f"      Context: '{context}'")
            print(f"      Next token: '{next_token}'")
            
            # Simulate probability distribution
            candidates = self.simulate_next_token_probabilities(context, next_token)
            print(f"      Top candidates: {candidates}")
            print()
        
        # Explain causal masking
        self.explain_causal_masking()
    
    def simulate_next_token_probabilities(self, context, actual_token):
        """
        Simulate probability distribution for next token prediction
        """
        # Create realistic probability distribution
        vocab_sample = ["the", "and", "to", "is", "likely", "transformative", 
                       "humanity", ".", ",", "will", "can", "may"]
        
        # Give higher probability to the actual token
        probabilities = np.random.rand(len(vocab_sample)) * 0.1
        
        if actual_token in vocab_sample:
            idx = vocab_sample.index(actual_token)
            probabilities[idx] = 0.8
        else:
            # Add the actual token
            vocab_sample.append(actual_token)
            probabilities = np.append(probabilities, 0.8)
        
        # Normalize
        probabilities = probabilities / probabilities.sum()
        
        # Sort by probability
        sorted_indices = np.argsort(probabilities)[::-1]
        top_candidates = [(vocab_sample[i], probabilities[i]) 
                         for i in sorted_indices[:3]]
        
        return top_candidates
    
    def explain_causal_masking(self):
        """
        Explain causal masking in GPT training
        """
        print("🎭 Causal Masking in GPT Training:")
        
        # Example sequence
        sequence = ["The", "cat", "sat", "on", "mat"]
        seq_len = len(sequence)
        
        print(f"   Example sequence: {sequence}")
        print()
        
        # Show attention mask
        mask = np.tril(np.ones((seq_len, seq_len)))
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(mask, 
                   xticklabels=sequence,
                   yticklabels=sequence,
                   annot=True, fmt='.0f',
                   cmap='RdBu', cbar=True)
        plt.title('Causal Attention Mask\n(1=allowed, 0=masked)', fontweight='bold')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        # Show training targets
        plt.subplot(1, 2, 2)
        training_data = []
        for i in range(seq_len):
            context = sequence[:i+1]
            if i < seq_len - 1:
                target = sequence[i+1]
            else:
                target = "<EOS>"
            training_data.append((context, target))
        
        # Visualize training examples
        contexts = [" ".join(ctx) for ctx, _ in training_data]
        targets = [tgt for _, tgt in training_data]
        
        y_pos = np.arange(len(contexts))
        
        plt.barh(y_pos, [len(ctx.split()) for ctx in contexts], alpha=0.7)
        
        for i, (ctx, tgt) in enumerate(zip(contexts, targets)):
            plt.text(0.1, i, f"'{ctx}' → '{tgt}'", 
            va='center', fontsize=10, ha='left')
        
        plt.title('Training Examples from Sequence', fontweight='bold')
        plt.xlabel('Context Length')
        plt.ylabel('Training Example')
        plt.yticks(y_pos, [f"Ex {i+1}" for i in range(len(contexts))])
        
        plt.tight_layout()
        plt.show()
        
        print("   Key Points:")
        print("   • Each position predicts the next token")
        print("   • Cannot see future tokens (causal constraint)")
        print("   • Enables parallel training on sequences")
        print("   • Sequential generation during inference")
    
    def demonstrate_generation_strategies(self):
        """
        Compare different text generation strategies
        """
        print("\n🎲 Text Generation Strategies:")
        
        # Simulate probability distribution for next token
        vocab = ["amazing", "interesting", "good", "great", "wonderful", 
                "nice", "excellent", "fantastic", "brilliant", "the"]
        logits = np.array([2.1, 1.8, 1.5, 2.0, 1.2, 1.0, 1.7, 1.3, 1.4, 0.5])
        probabilities = np.exp(logits) / np.exp(logits).sum()
        
        prompt = "This movie is"
        
        print(f"   Prompt: '{prompt}'")
        print(f"   Next token probabilities:")
        
        for token, prob in zip(vocab, probabilities):
            print(f"      {token}: {prob:.3f}")
        print()
        
        # 1. Greedy Decoding
        greedy_choice = vocab[np.argmax(probabilities)]
        print(f"   🎯 Greedy Decoding: Always pick highest probability")
        print(f"      Selected: '{greedy_choice}' (prob: {probabilities.max():.3f})")
        print(f"      Pros: Deterministic, fast")
        print(f"      Cons: Repetitive, no diversity")
        print()
        
        # 2. Random Sampling
        np.random.seed(42)
        random_choice = np.random.choice(vocab, p=probabilities)
        print(f"   🎲 Random Sampling: Sample from full distribution")
        print(f"      Selected: '{random_choice}'")
        print(f"      Pros: Diverse output")
        print(f"      Cons: Can be incoherent")
        print()
        
        # 3. Temperature Sampling
        temperatures = [0.5, 1.0, 2.0]
        print(f"   🌡️ Temperature Sampling: Control randomness")
        
        for temp in temperatures:
            temp_logits = logits / temp
            temp_probs = np.exp(temp_logits) / np.exp(temp_logits).sum()
            temp_choice = vocab[np.argmax(temp_probs)]
            
            print(f"      Temperature {temp}: '{temp_choice}'")
            print(f"         Distribution shape: {self.describe_distribution(temp_probs)}")
        print()
        
        # 4. Top-k Sampling
        k_values = [3, 5, 8]
        print(f"   🔝 Top-k Sampling: Sample from top k tokens")
        
        for k in k_values:
            top_k_indices = np.argsort(probabilities)[-k:]
            top_k_probs = probabilities[top_k_indices]
            top_k_probs = top_k_probs / top_k_probs.sum()  # Renormalize
            
            k_choice = vocab[top_k_indices[np.argmax(top_k_probs)]]
            top_k_tokens = [vocab[i] for i in top_k_indices]
            
            print(f"      Top-{k}: {top_k_tokens}")
            print(f"      Selected: '{k_choice}'")
        print()
        
        # 5. Top-p (Nucleus) Sampling
        p_values = [0.8, 0.9, 0.95]
        print(f"   🌰 Top-p (Nucleus) Sampling: Dynamic vocabulary size")
        
        for p in p_values:
            sorted_indices = np.argsort(probabilities)[::-1]
            cumsum_probs = np.cumsum(probabilities[sorted_indices])
            nucleus_size = np.sum(cumsum_probs <= p) + 1  # +1 to include the boundary
            
            nucleus_indices = sorted_indices[:nucleus_size]
            nucleus_tokens = [vocab[i] for i in nucleus_indices]
            
            print(f"      Top-p {p}: {nucleus_tokens} ({nucleus_size} tokens)")
        
        # Visualize sampling strategies
        self.visualize_sampling_strategies(vocab, probabilities, logits)
    
    def describe_distribution(self, probs):
        """Describe the shape of a probability distribution"""
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_entropy
        
        if normalized_entropy > 0.8:
            return "uniform"
        elif normalized_entropy > 0.5:
            return "spread"
        else:
            return "peaked"
    
    def visualize_sampling_strategies(self, vocab, probabilities, logits):
        """
        Visualize different sampling strategies
        """
        plt.figure(figsize=(18, 12))
        
        # Plot 1: Original probability distribution
        plt.subplot(2, 3, 1)
        bars = plt.bar(range(len(vocab)), probabilities, alpha=0.7, color='skyblue')
        plt.title('Original Probability Distribution', fontweight='bold')
        plt.xlabel('Tokens')
        plt.ylabel('Probability')
        plt.xticks(range(len(vocab)), vocab, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Highlight top tokens
        top_3_indices = np.argsort(probabilities)[-3:]
        for idx in top_3_indices:
            bars[idx].set_color('orange')
        
        # Plot 2: Temperature effects
        plt.subplot(2, 3, 2)
        temperatures = [0.1, 0.5, 1.0, 2.0]
        
        for temp in temperatures:
            temp_logits = logits / temp
            temp_probs = np.exp(temp_logits) / np.exp(temp_logits).sum()
            plt.plot(range(len(vocab)), temp_probs, 'o-', 
                    label=f'T={temp}', linewidth=2, markersize=6)
        
        plt.title('Temperature Effects on Distribution', fontweight='bold')
        plt.xlabel('Tokens')
        plt.ylabel('Probability')
        plt.xticks(range(len(vocab)), vocab, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Top-k sampling
        plt.subplot(2, 3, 3)
        k_values = [3, 5, 8]
        
        for k in k_values:
            k_probs = np.zeros_like(probabilities)
            top_k_indices = np.argsort(probabilities)[-k:]
            k_probs[top_k_indices] = probabilities[top_k_indices]
            k_probs = k_probs / k_probs.sum()  # Renormalize
            
            plt.bar(range(len(vocab)), k_probs, alpha=0.6, 
                   label=f'Top-{k}', width=0.8/(len(k_values)))
        
        plt.title('Top-k Sampling Comparison', fontweight='bold')
        plt.xlabel('Tokens')
        plt.ylabel('Probability')
        plt.xticks(range(len(vocab)), vocab, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Top-p (nucleus) sampling
        plt.subplot(2, 3, 4)
        sorted_indices = np.argsort(probabilities)[::-1]
        cumsum_probs = np.cumsum(probabilities[sorted_indices])
        
        plt.plot(range(len(vocab)), cumsum_probs, 'ro-', linewidth=2, markersize=8)
        
        # Mark different p values
        p_values = [0.8, 0.9, 0.95]
        colors = ['red', 'green', 'blue']
        
        for p, color in zip(p_values, colors):
            nucleus_size = np.sum(cumsum_probs <= p) + 1
            plt.axhline(y=p, color=color, linestyle='--', alpha=0.7, 
                       label=f'p={p} (size={nucleus_size})')
            plt.axvline(x=nucleus_size-1, color=color, linestyle='--', alpha=0.7)
        
        plt.title('Top-p (Nucleus) Sampling', fontweight='bold')
        plt.xlabel('Token Rank')
        plt.ylabel('Cumulative Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Generation quality comparison
        plt.subplot(2, 3, 5)
        
        strategies = ['Greedy', 'Random', 'Temp=0.7', 'Top-k=5', 'Top-p=0.9']
        coherence_scores = [9, 3, 7, 8, 8]
        diversity_scores = [2, 9, 6, 7, 7]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        plt.bar(x - width/2, coherence_scores, width, label='Coherence', alpha=0.8, color='lightblue')
        plt.bar(x + width/2, diversity_scores, width, label='Diversity', alpha=0.8, color='lightcoral')
        
        plt.title('Generation Strategy Comparison', fontweight='bold')
        plt.xlabel('Strategy')
        plt.ylabel('Score (1-10)')
        plt.xticks(x, strategies, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Beam search visualization
        plt.subplot(2, 3, 6)
        
        # Simulate beam search with beam_size=3
        beam_steps = ['Step 1', 'Step 2', 'Step 3', 'Step 4']
        beam_scores = [
            [0.9, 0.7, 0.6],  # Step 1 top 3 beams
            [0.8, 0.7, 0.5],  # Step 2 top 3 beams  
            [0.7, 0.6, 0.5],  # Step 3 top 3 beams
            [0.6, 0.5, 0.4]   # Step 4 top 3 beams
        ]
        
        for i, scores in enumerate(beam_scores):
            plt.plot(range(len(scores)), scores, 'o-', 
                    linewidth=2, markersize=8, label=beam_steps[i])
        
        plt.title('Beam Search Score Evolution', fontweight='bold')
        plt.xlabel('Beam Rank')
        plt.ylabel('Sequence Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("📊 Sampling Strategy Guidelines:")
        print("   • Greedy: Fast, deterministic, good for factual tasks")
        print("   • Temperature: Control creativity (0.7-1.0 often good)")
        print("   • Top-k: Prevents low-quality tokens, fixed vocabulary")
        print("   • Top-p: Dynamic vocabulary, adapts to confidence")
        print("   • Beam Search: Better for tasks requiring consistency")
    
    def demonstrate_in_context_learning(self):
        """
        Demonstrate in-context learning capabilities of large language models
        """
        print("\n🧠 In-Context Learning: Learning Without Parameter Updates")
        print("=" * 60)
        
        print("💡 Key Insight: Large models can learn from examples in the prompt")
        print("   No fine-tuning needed - just provide examples in context")
        print("   Emergent capability that appears with sufficient scale")
        print()
        
        # Demonstrate different types of in-context learning
        self.show_few_shot_examples()
        self.show_chain_of_thought()
        self.show_instruction_following()
    
    def show_few_shot_examples(self):
        """
        Show few-shot learning examples
        """
        print("🎯 Few-Shot Learning Examples:")
        
        examples = {
            "Sentiment Analysis": {
                "examples": [
                    "Text: 'I love this product!' → Sentiment: Positive",
                    "Text: 'This is terrible.' → Sentiment: Negative",
                    "Text: 'It's okay, I guess.' → Sentiment: Neutral"
                ],
                "query": "Text: 'Amazing quality and fast shipping!' → Sentiment:",
                "expected": "Positive"
            },
            "Language Translation": {
                "examples": [
                    "English: Hello → French: Bonjour",
                    "English: Thank you → French: Merci",
                    "English: Good morning → French: Bonjour"
                ],
                "query": "English: How are you? → French:",
                "expected": "Comment allez-vous?"
            },
            "Math Word Problems": {
                "examples": [
                    "Q: If I have 5 apples and buy 3 more, how many do I have? A: 8",
                    "Q: A train travels 60 mph for 2 hours. How far does it go? A: 120 miles"
                ],
                "query": "Q: If a book costs $12 and I buy 4 books, what's the total cost? A:",
                "expected": "$48"
            }
        }
        
        for task, data in examples.items():
            print(f"\n   📚 {task}:")
            print("      Examples in prompt:")
            for example in data["examples"]:
                print(f"         {example}")
            print(f"      Query: {data['query']}")
            print(f"      Expected: {data['expected']}")
        
        print("\n   🎯 Few-Shot Learning Principles:")
        principles = [
            "Use diverse, representative examples",
            "Maintain consistent format across examples",
            "Order examples from simple to complex",
            "Include edge cases and counterexamples",
            "Use clear separators between examples"
        ]
        
        for principle in principles:
            print(f"      • {principle}")
    
    def show_chain_of_thought(self):
        """
        Demonstrate chain-of-thought reasoning
        """
        print("\n🔗 Chain-of-Thought Reasoning:")
        
        print("   💭 Standard Reasoning:")
        print("      Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.")
        print("         Each can has 3 tennis balls. How many tennis balls does he have now?")
        print("      A: 11")
        print()
        
        print("   🧠 Chain-of-Thought Reasoning:")
        print("      Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.")
        print("         Each can has 3 tennis balls. How many tennis balls does he have now?")
        print("      A: Roger starts with 5 balls. He buys 2 cans of balls.")
        print("         Each can has 3 balls, so 2 cans have 2 × 3 = 6 balls.")
        print("         In total, Roger has 5 + 6 = 11 tennis balls.")
        print()
        
        print("   🎯 Chain-of-Thought Benefits:")
        benefits = [
            "Improves reasoning on complex problems",
            "Makes reasoning process interpretable",
            "Helps model avoid silly mistakes",
            "Enables step-by-step verification",
            "Works with zero-shot prompting too"
        ]
        
        for benefit in benefits:
            print(f"      • {benefit}")
        
        print("\n   🔢 Zero-Shot Chain-of-Thought:")
        print("      Prompt: 'Let's think step by step.'")
        print("      Effect: Encourages reasoning without examples")
        print("      Usage: Add to end of any reasoning question")
    
    def show_instruction_following(self):
        """
        Demonstrate instruction following capabilities
        """
        print("\n📋 Instruction Following:")
        
        print("   📝 Natural Language Instructions:")
        instruction_examples = [
            {
                "instruction": "Summarize the following text in exactly 2 sentences:",
                "input": "[Long article about climate change...]",
                "output": "Climate change is causing global temperatures to rise due to greenhouse gas emissions. Immediate action is needed to reduce emissions and mitigate environmental impacts."
            },
            {
                "instruction": "Translate the following text to Spanish, keeping a formal tone:",
                "input": "Thank you for your attention to this matter.",
                "output": "Gracias por su atención a este asunto."
            },
            {
                "instruction": "Extract all dates mentioned in the following text:",
                "input": "The meeting was scheduled for March 15, 2023, but was moved to April 2nd.",
                "output": "March 15, 2023; April 2nd"
            }
        ]
        
        for i, example in enumerate(instruction_examples, 1):
            print(f"\n      Example {i}:")
            print(f"         Instruction: {example['instruction']}")
            print(f"         Input: {example['input']}")
            print(f"         Output: {example['output']}")
        
        print("\n   🎯 Instruction Following Best Practices:")
        practices = [
            "Be specific and clear in instructions",
            "Provide output format specifications",
            "Include constraints and requirements",
            "Use consistent instruction phrasing",
            "Test instructions with diverse inputs"
        ]
        
        for practice in practices:
            print(f"      • {practice}")


# ==========================================
# SECTION 4: T5 TEXT-TO-TEXT FRAMEWORK
# ==========================================

class T5TextToTextFramework:
    """
    Demonstration of T5's unified text-to-text approach
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def demonstrate_t5_framework(self):
        """
        Demonstrate T5's text-to-text unified framework
        """
        print("\n🔄 T5: Text-to-Text Transfer Transformer")
        print("=" * 50)
        
        print("🎯 T5 Innovation: Everything is Text-to-Text")
        print("   Key Insight: Convert all NLP tasks to text generation")
        print("   Input: Natural language with task prefix")
        print("   Output: Natural language text")
        print("   Benefit: Unified framework for diverse tasks")
        print()
        
        # Show T5 task formulations
        self.demonstrate_task_formulations()
        
        # Explain span corruption pre-training
        self.demonstrate_span_corruption()
        
        # Show multitask training
        self.demonstrate_multitask_training()
    
    def demonstrate_task_formulations(self):
        """
        Show how different NLP tasks are formulated as text-to-text
        """
        print("📝 T5 Task Formulations:")
        
        tasks = {
            "Translation": {
                "input": "translate English to German: That is good.",
                "output": "Das ist gut.",
                "explanation": "Translation with explicit source and target language"
            },
            "Summarization": {
                "input": "summarize: [Long article text...]",
                "output": "Brief summary of the article's main points.",
                "explanation": "Generate concise summary of input text"
            },
            "Question Answering": {
                "input": "question: What is the capital of France? context: France is a country in Europe. Its capital city is Paris.",
                "output": "Paris",
                "explanation": "Extract answer from given context"
            },
            "Sentiment Analysis": {
                "input": "sentiment: I love this movie!",
                "output": "positive",
                "explanation": "Classify sentiment as text generation"
            },
            "Text Classification": {
                "input": "cola sentence: The book was read by the student.",
                "output": "acceptable",
                "explanation": "Grammatical acceptability as text output"
            },
            "Paraphrasing": {
                "input": "paraphrase: The cat sat on the mat.",
                "output": "A cat was sitting on a mat.",
                "explanation": "Generate alternative phrasing"
            },
            "Reading Comprehension": {
                "input": "question: Who wrote Romeo and Juliet? context: Romeo and Juliet is a tragedy written by William Shakespeare.",
                "output": "William Shakespeare",
                "explanation": "Answer based on provided context"
            }
        }
        
        for task_name, task_info in tasks.items():
            print(f"\n   📋 {task_name}:")
            print(f"      Input:  {task_info['input']}")
            print(f"      Output: {task_info['output']}")
            print(f"      Note:   {task_info['explanation']}")
        
        print("\n   💡 T5 Framework Benefits:")
        benefits = [
            "Unified architecture for all tasks",
            "Transfer learning across diverse tasks",
            "Simple training and inference pipeline",
            "Easy to add new tasks",
            "Leverages sequence-to-sequence strengths"
        ]
        
        for benefit in benefits:
            print(f"      • {benefit}")
    
    def demonstrate_span_corruption(self):
        """
        Demonstrate T5's span corruption pre-training objective
        """
        print("\n🎭 T5 Span Corruption Pre-training:")
        
        print("   🎯 Objective: Predict corrupted spans in text")
        
        # Example span corruption
        original_text = "Thank you for inviting me to your party last week."
        
        # Show corruption process
        corruption_steps = [
            {
                "step": "Original text",
                "text": original_text,
                "explanation": "Start with clean text"
            },
            {
                "step": "Identify spans",
                "text": "Thank you [SPAN1] inviting me [SPAN2] your party [SPAN3] week.",
                "explanation": "Mark random spans for corruption (15% of tokens)"
            },
            {
                "step": "Corrupted input",
                "text": "Thank you <extra_id_0> inviting me <extra_id_1> your party <extra_id_2> week.",
                "explanation": "Replace spans with sentinel tokens"
            },
            {
                "step": "Target output",
                "text": "<extra_id_0> for <extra_id_1> to <extra_id_2> last <extra_id_3>",
                "explanation": "Generate original spans with sentinels"
            }
        ]
        
        for step_info in corruption_steps:
            print(f"\n      {step_info['step']}:")
            print(f"         Text: {step_info['text']}")
            print(f"         Note: {step_info['explanation']}")
        
        # Visualize span corruption
        self.visualize_span_corruption()
        
        print("\n   🎯 Span Corruption Benefits:")
        benefits = [
            "Bidirectional context like BERT",
            "Generation capability like GPT",
            "Efficient: only predict corrupted spans",
            "Flexible span lengths",
            "Good for both understanding and generation"
        ]
        
        for benefit in benefits:
            print(f"      • {benefit}")
    
    def visualize_span_corruption(self):
        """
        Visualize the span corruption process
        """
        plt.figure(figsize=(15, 8))
        
        # Original sequence
        tokens = ["Thank", "you", "for", "inviting", "me", "to", "your", "party", "last", "week", "."]
        token_positions = range(len(tokens))
        
        # Corruption mask (1 = corrupted, 0 = kept)
        corruption_mask = [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        
        plt.subplot(2, 2, 1)
        colors = ['lightblue' if mask == 0 else 'lightcoral' for mask in corruption_mask]
        bars = plt.bar(token_positions, [1] * len(tokens), color=colors, alpha=0.7)
        
        # Add token labels
        for i, (token, bar) in enumerate(zip(tokens, bars)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                    token, ha='center', va='center', fontsize=9, rotation=45)
        
        plt.title('Original Text with Corruption Mask', fontweight='bold')
        plt.xlabel('Token Position')
        plt.ylabel('Token')
        plt.yticks([])
        
        # Corrupted input
        plt.subplot(2, 2, 2)
        corrupted_tokens = ["Thank", "you", "<extra_id_0>", "me", "<extra_id_1>", "your", "party", "<extra_id_2>", "week", "."]
        
        # Filter out corrupted spans, keep sentinels
        input_tokens = []
        for i, (token, mask) in enumerate(zip(tokens, corruption_mask)):
            if mask == 0:
                input_tokens.append(token)
            elif i == 0 or corruption_mask[i-1] == 0:  # Start of new span
                span_id = sum(corruption_mask[:i+1]) - 1
                input_tokens.append(f"<extra_id_{span_id}>")
        
        input_colors = ['lightblue' if not token.startswith('<extra_id_') else 'yellow' 
                       for token in input_tokens]
        
        bars = plt.bar(range(len(input_tokens)), [1] * len(input_tokens), 
                      color=input_colors, alpha=0.7)
        
        for i, (token, bar) in enumerate(zip(input_tokens, bars)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                    token, ha='center', va='center', fontsize=9, rotation=45)
        
        plt.title('Corrupted Input Sequence', fontweight='bold')
        plt.xlabel('Token Position')
        plt.ylabel('Token')
        plt.yticks([])
        
        # Target output
        plt.subplot(2, 2, 3)
        target_tokens = ["<extra_id_0>", "for", "inviting", "<extra_id_1>", "to", "<extra_id_2>", "last", "<extra_id_3>"]
        
        target_colors = ['yellow' if token.startswith('<extra_id_') else 'lightgreen' 
                        for token in target_tokens]
        
        bars = plt.bar(range(len(target_tokens)), [1] * len(target_tokens), 
                      color=target_colors, alpha=0.7)
        
        for i, (token, bar) in enumerate(zip(target_tokens, bars)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                    token, ha='center', va='center', fontsize=9, rotation=45)
        
        plt.title('Target Output Sequence', fontweight='bold')
        plt.xlabel('Token Position')
        plt.ylabel('Token')
        plt.yticks([])
        
        # Span corruption statistics
        plt.subplot(2, 2, 4)
        
        span_lengths = [2, 1, 1]  # Length of each corrupted span
        span_ids = ['Span 0', 'Span 1', 'Span 2']
        
        plt.bar(span_ids, span_lengths, alpha=0.7, color='lightcoral')
        plt.title('Corrupted Span Lengths', fontweight='bold')
        plt.xlabel('Span ID')
        plt.ylabel('Length (tokens)')
        plt.grid(True, alpha=0.3)
        
        # Add corruption statistics
        total_tokens = len(tokens)
        corrupted_tokens = sum(corruption_mask)
        corruption_rate = corrupted_tokens / total_tokens
        
        plt.text(0.5, max(span_lengths) * 0.8, 
                f'Corruption Rate: {corruption_rate:.1%}\n'
                f'Original: {total_tokens} tokens\n'
                f'Corrupted: {corrupted_tokens} tokens',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_multitask_training(self):
        """
        Show T5's multitask training approach
        """
        print("\n🎯 T5 Multitask Training:")
        
        print("   📚 Training Data Composition:")
        datasets = {
            "C4 (Colossal Clean Crawled Corpus)": {
                "size": "750GB",
                "purpose": "Unsupervised span corruption",
                "percentage": "50%"
            },
            "GLUE Tasks": {
                "size": "Various",
                "purpose": "Language understanding",
                "percentage": "15%"
            },
            "SuperGLUE Tasks": {
                "size": "Various", 
                "purpose": "Advanced understanding",
                "percentage": "10%"
            },
            "CNN/DailyMail": {
                "size": "300K articles",
                "purpose": "Summarization",
                "percentage": "10%"
            },
            "WMT Translation": {
                "size": "36M sentence pairs",
                "purpose": "Machine translation",
                "percentage": "10%"
            },
            "SQuAD": {
                "size": "100K questions",
                "purpose": "Question answering",
                "percentage": "5%"
            }
        }
        
        for dataset, info in datasets.items():
            print(f"\n      📊 {dataset}:")
            print(f"         Size: {info['size']}")
            print(f"         Purpose: {info['purpose']}")
            print(f"         Training %: {info['percentage']}")
        
        # Visualize training data composition
        plt.figure(figsize=(12, 8))
        
        plt.subplot(1, 2, 1)
        dataset_names = list(datasets.keys())
        percentages = [float(info['percentage'].rstrip('%')) for info in datasets.values()]
        colors