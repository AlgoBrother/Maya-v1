# MAYA

## Problem Statement

Modern language models are designed to run on computationally expensive hardware. MAYA takes the opposite approach.

The goal is to design and train a small language model that can run locally on everyday machines, with a strict hardware constraint of **8GB VRAM**.

---

## Vision

Build an efficient, small-scale language model that:

* Trains on limited consumer hardware
* Runs smoothly on the same hardware it was trained on
* Does not rely on external API wrappers
* Prioritizes architectural control and system-level optimization

---

## Technical Constraints

* Target Hardware: 8GB VRAM (NVIDIA 4070 Laptop GPU)
* Parameter Budget: ≤ 100 million parameters
* Dataset: Cosmopedia
* Target Perplexity: < 20
* Estimated Training Time: ~3 months (single GPU setup)

---

## Language Stack

### Rust

* High performance
* Efficient memory management
* Prevent memory spilling during training
* Used for tokenizer and planned full training pipeline rewrite

### Python

* Initial Transformer implementation
* Rapid prototyping
* PyTorch experimentation

---

## Tokenizer

### MayaTok v1

* Custom Byte Pair Encoding (BPE) tokenizer
* Written in Rust
* Currently slower than production-grade tokenizers
* Already used to tokenize dataset, so retained for consistency

Planned improvements:

* Performance optimization
* Benchmarking against industry tokenizers
* Efficient serialization pipeline

---

## Model Architecture

* Transformer-based architecture
* Initially implemented in Python
* Full rewrite in Rust in progress
* Focus on efficient memory usage and stable training

Components:

* Multi-head self-attention
* Masking
* Custom DataLoader
* Training loop (incomplete)

---

## Dataset Pipeline

* Tokenized dataset currently stored as `.pt` files
* Planned migration to `.bin` format
* Will use `memmap` for memory-efficient streaming during training

Objective:

* Avoid loading full dataset into RAM
* Enable long training cycles without memory overflow

---

## Current Status

### Completed

* Custom BPE tokenizer (Rust implementation)
* Transformer architecture (Python implementation)
* Masking implementation (Python)
* DataLoader prototype (Python)
* V1 Training Completed

### In Progress

* Rust-based DataLoader
* Training loop completion (V2)


### Pending

* V2(340M) training run
* Perplexity benchmarking
* Math Solving integration

---

## Core Objective

Train a small language model under 100M parameters on limited hardware while maintaining:

* Stable training
* Efficient memory usage (This will e tested on my 8GB VRAM NVIDIA CARD)
* Practical inference performance
* Competitive perplexity (< 20)

---

## Long-Term Direction

* Optimized tokenizer pipeline
* Efficient small-scale foundation model
* Independent local AI system

---

## Why MAYA?

Most research scales upward with more compute. MAYA scales downward.

The focus is architectural efficiency over brute-force hardware scaling.

---
