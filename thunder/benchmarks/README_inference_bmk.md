# Thunder Inference Benchmark

A comprehensive inference benchmark following the SemiAnalysis methodology for evaluating Large Language Model (LLM) performance across different hardware configurations and compilation modes.

## Overview

This benchmark implements the methodology from the SemiAnalysis article: ["AMD vs NVIDIA Inference Benchmark: Who Wins? - Performance & Cost Per Million Tokens"](https://semianalysis.com/2025/05/23/amd-vs-nvidia-inference-benchmark-who-wins-performance-cost-per-million-tokens)

### Key Features

- **Standardized Scenarios**: Three predefined benchmark scenarios targeting different workload characteristics
- **Multiple Compilation Modes**: Support for Thunder, PyTorch eager mode, and torch.compile
- **Detailed Metrics**: Comprehensive performance metrics including TTFT, TBOT, throughput, and cost analysis
- **Flexible Configuration**: Customizable parameters for experimentation
- **Hardware Optimization Focus**: Scenarios designed to highlight different hardware strengths

## Standard Benchmark Scenarios

### 1. Summarization (Prefill-Heavy) 4k -> 1k

- **Configuration**: 4,000 input → 1,000 output tokens
- **Workload Balance**: 80% prefill, 20% decode computational cost
- **Hardware Focus**: Compute optimization provides maximum impact
- **Use Case**: Document summarization, content analysis
- **Target**: Primary demonstration scenario for B300 hardware

### 2. Chat (Balanced) 1k -> 1k

- **Configuration**: 1,000 input → 1,000 output tokens
- **Workload Balance**: 50% prefill, 50% decode computational cost
- **Hardware Focus**: Mixed optimization requirements
- **Use Case**: Conversational AI, general-purpose inference
- **Target**: Baseline comparison scenario

### 3. Reasoning (Decode-Heavy) 1k -> 4k

- **Configuration**: 1,000 input → 4,000 output tokens
- **Workload Balance**: 20% prefill, 80% decode computational cost
- **Hardware Focus**: Memory bandwidth optimization dominates
- **Use Case**: Complex reasoning tasks, code generation
- **Target**: Memory bandwidth optimization showcase

## Supported Models

- **Llama 3.1 8B** (default) - ~16GB VRAM required
- **Llama 3.1 70B** - ~140GB VRAM required
- **Llama 3.1 405B** - ~810GB VRAM required
- **DeepSeekV3 670B** - ~1340GB VRAM required
- **Llama 4 Scout 17B** - ~LOTS VRAM required
- **Llama 4 Maverick** - ~LOTS VRAM required

Can lower memory requirements by specifying lower number of layers! `--num-layers 4` a minimum of 4 is recommended with Llama4.

## Compilation Modes

### Thunder Mode (Default)

Thunder compilation with various executor configurations:

- **Default**: Standard Thunder jit executors
- **dynamo**: ThunderFX
- **transformerengine**: Thunder jit/fx + TransformerEngine executor
- **transformerengine_v2**: Thunder jit/fx + TransformerEngine V2 executor

### Eager Mode

Pure PyTorch eager execution (no compilation)

### Inductor Mode

PyTorch torch.compile

## Command Line Reference

### Basic Usage

```bash
python benchmark_inference.py [OPTIONS]
```

### Core Options

#### Model Configuration

```bash
--model-name {llama3.1-8b,llama3.1-70b,llama3.1-405b,deepseekv3-670b,llama4-scout,llama4-maverick}
    Model to benchmark (default: llama3.1-8b)

--precision {fp16,bf16}
    Model precision (default: bf16)

--num-layers INTEGER
    Override number of model layers (for experimentation)
```

#### Scenario Configuration

```bash
--scenario {summarization,chat,reasoning}
    Use standardized benchmark scenario
    - summarization: 4,000 input → 1,000 output (prefill-heavy)
    - chat: 1,000 input → 1,000 output (balanced)
    - reasoning: 1,000 input → 4,000 output (decode-heavy)

--list-scenarios
    List detailed scenario descriptions and exit
```

#### Custom Workload Configuration

```bash
--batch-size INTEGER
    Batch size for inference (default: 1)

--input-length INTEGER
    Input sequence length (default: 2048, ignored if --scenario used)

--output-length INTEGER
    Output sequence length (default: 128, ignored if --scenario used)
```

#### Execution Configuration

```bash
--mode {thunder,eager,inductor}
    Compilation mode (default: thunder)

--thunder-executors STRING
    Thunder executor configuration (optional, only with --mode thunder)
    Examples: inductor, inductor_cat, transformerengine, transformerengine_v2, dynamo
```

#### Benchmark Configuration

```bash
--num-iterations INTEGER
    Number of benchmark iterations (default: 100)

--warmup-iterations INTEGER
    Number of warmup iterations (default: 10)
```

#### Output Configuration

```bash
--save-results
    Save detailed results to JSON file

--output-dir PATH
    Directory to save results (default: ./results)
```

## Usage Examples

### Standard Scenarios

#### Run Chat Scenario with Default Thunder

```bash
python benchmark_inference.py --scenario chat --model-name llama3.1-8b
```

#### Run Summarization Scenario with Thunder + Inductor

```bash
python benchmark_inference.py --scenario summarization --mode thunder --thunder-executors inductor
```

#### Run Reasoning Scenario with Eager Mode

```bash
python benchmark_inference.py --scenario reasoning --mode eager --save-results
```

### Custom Configurations

#### Custom Input/Output Lengths

```bash
python benchmark_inference.py --input-length 2048 --output-length 512 --mode thunder
```

#### Large Model with Multiple Iterations

```bash
python benchmark_inference.py --model-name llama3.1-70b --scenario chat --num-iterations 50
```

#### Experimentation with Different Executors

```bash
# Thunder with TransformerEngine
python benchmark_inference.py --scenario chat --mode thunder --thunder-executors transformerengine_v2

# Pure Inductor mode
python benchmark_inference.py --scenario reasoning --mode inductor

# Eager mode for baseline
python benchmark_inference.py --scenario summarization --mode eager
```

### Performance Comparison

#### Compare Different Modes for Same Scenario

```bash
# Thunder (default executors)
python benchmark_inference.py --scenario chat --mode thunder --save-results

# Thunder + Inductor
python benchmark_inference.py --scenario chat --mode thunder --thunder-executors inductor --save-results

# Pure Inductor
python benchmark_inference.py --scenario chat --mode inductor --save-results

# Eager baseline
python benchmark_inference.py --scenario chat --mode eager --save-results
```

## Output Metrics

### Throughput Metrics

- **Overall Throughput**: Total tokens processed per second
- **Prefill Throughput**: Input processing tokens per second
- **Decode Throughput**: Output generation tokens per second
- **Latency**: Milliseconds per generated token

### Latency Breakdown

- **Time to First Token (TTFT)**: Time from input to first generated token
- **Time Between Output Tokens (TBOT)**: Average time between consecutive output tokens
- **Prefill Time**: Time to process input prompt
- **Decode Time**: Time for output generation phase

### Resource Usage

- **Current Memory**: GPU memory usage during inference
- **Peak Memory**: Maximum GPU memory usage
- **Cost per Million Tokens**: Estimated cost based on GPU pricing

### Variance Analysis

- **Throughput Standard Deviation**: Consistency of performance
- **TTFT Standard Deviation**: First token latency variance
- **TBOT Standard Deviation**: Output token timing consistency

## Best Practices

### Performance Optimization

1. **For Compute-Bound Workloads**: Use summarization scenario, focus on Thunder + specialized executors
1. **For Memory-Bound Workloads**: Use reasoning scenario, optimize for memory bandwidth
1. **For Balanced Workloads**: Use chat scenario for general optimization
1. **For Baseline Comparison**: Always include eager mode results

## Troubleshooting

### Getting Help

```bash
# List all available options
python benchmark_inference.py --help

# List detailed scenario descriptions
python benchmark_inference.py --list-scenarios
```

## Example Workflow

```bash
# 1. List available scenarios
python benchmark_inference.py --list-scenarios

# 2. Run standard chat benchmark
python benchmark_inference.py --scenario chat --save-results

# 3. Compare with different executor
python benchmark_inference.py --scenario chat --thunder-executors inductor --save-results

# 4. Test eager mode baseline
python benchmark_inference.py --scenario chat --mode eager --save-results

# 5. Run all three scenarios for comprehensive evaluation
python benchmark_inference.py --scenario summarization --save-results
python benchmark_inference.py --scenario chat --save-results
python benchmark_inference.py --scenario reasoning --save-results
```

This benchmark provides a comprehensive evaluation framework for LLM inference performance, enabling systematic comparison across different models, hardware configurations, and optimization strategies.
