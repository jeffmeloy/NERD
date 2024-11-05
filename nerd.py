"""
NER (Normalized Effective Rank) quantifies dimensional utilization across layers using entropy analysis of singular value distributions. NER calculation involves Singular Value Decomposition (SVD) of the weight matrix A. Singular values form a probability distribution through normalization. The entropy H of this distribution indicates how evenly the singular values are spread, yielding the Effective Rank (ERank) as 2^H. Normalizing by the maximum possible entropy H_max produces a value between 0 and 1, measuring how efficiently the layer uses its dimensions.

**Why Entropy as a Measure**
Entropy is used here because it captures the spread or uniformity of the singular value distribution. A higher entropy value indicates that the singular values are more evenly distributed, meaning the layer uses all available dimensions more effectively. In contrast, lower entropy suggests that only a few dimensions dominate, leading to inefficient dimensional utilization. Thus, entropy provides an intuitive and mathematically grounded way to quantify how well a layer’s representational capacity is being utilized.

Run the script with:
    python mastermerge.py --config mastermerge_config.yaml (optional)

The script loads the configuration, processes each model by downloading, loading weights, normalizing layers, and calculating NER for each layer. It then selects the optimal layer based on NER and creates a composite model with the highest NER in each layer, aiming to maximize information flow and improve model performance.

**Unified Theoretical Analysis of NER in Neural Networks**

Given a neural network layer W ∈ ℝᵐˣⁿ with SVD W = UΣV*:

1. **Capacity and Information Flow**

The fundamental relationships:
• SVD components: Σ = diag(σ₁,...,σᵣ), σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0
• Probability distribution: pᵢ = σᵢ/∑σⱼ
• Entropy: H(W) = -∑pᵢlog(pᵢ)
• NER: NER(W) = H(W)/log(r)

Capacity bounds:
C(W) ≤ min{I(x;y), log(r)}
C_eff(W) = exp(H(W))  // effective dimensions
η(W) = exp(H(W))/r    // utilization efficiency

Subject to constraints:
tr(WW^T) ≤ P  // power constraint
E[||x||²] ≤ σ²  // input constraint

Optimal capacity:
C_opt = (r/2)log(1 + P/rσ²)
C_actual ≈ NER(W) * C_opt

2. **Information Bottleneck Analysis**

For network path x → t → y:
• Data Processing Inequality: I(x;t) ≥ I(t;y)
• IB Lagrangian: L(W) = I(t;y) - βI(x;t)
• Optimal solution requires: ∂L/∂W = 0

**Intuitive Explanation**:
Higher NER implies that a layer can transmit more information without significant loss. Since entropy measures how well the information is spread across dimensions, a high NER indicates efficient information transfer, minimizing bottlenecks. In practical terms, this means that layers with higher NER are better at preserving and passing along the complexity of the input data, which is crucial for deep learning models that rely on rich feature representations.

Information flow properties:
I(x;t) ≈ log(exp(H(W))) = H(W)
I(t;y) ≤ H(W)  // information ceiling

NER implications:
Higher NER → Better information preservation
Lower NER → Information bottleneck

3. **Optimization Landscape Properties**

Gradient and conditioning analysis:
• Condition number: κ(W) = σ₁/σₘᵢₙ
• Gradient bound: ||∇L||₂ ≤ κ(W)||∇L*||₂
• NER relationship: κ(W) ∝ 1/NER(W)

**Practical Example**:
Consider a layer with poor conditioning (high condition number). It results in inefficient gradient flow, where small updates in some directions are overwhelmed by large updates in others. Higher NER leads to better conditioning, meaning gradients propagate more uniformly, leading to smoother optimization and faster convergence. This is especially important for deep models where gradient flow can be a significant challenge.

Hessian properties:
• λₘₐₓ(H)/λₘᵢₙ(H) ≈ κ(W)²
• Higher NER → Better conditioning

Optimization implications:
• Smoother loss landscape
• Better gradient flow
• Faster convergence

4. **Statistical Learning Theory**

Generalization bounds:
Rademacher complexity:
R_n(F_W) ≤ √(∑σᵢ²/n)

Generalization bound:
|L_test - L_train| ≤ 2R_n(F_W) + √(log(1/δ)/2n)

With probability 1-δ:
E[L_test] ≤ L_train + O(√(1/NER(W) * log(r)/n))

**Practical Takeaways**:
- **Layer Selection for Fine-Tuning**: Use NER to identify which layers are most efficient in terms of information utilization. These layers can be prioritized during fine-tuning or targeted for modifications to improve model performance.
- **Model Design**: When designing or merging models, consider maximizing NER across critical layers to ensure robust information flow and efficient training dynamics.

**Unified Performance Bound**:

Performance P(W) bounded by:
P(W) ≤ O(1/NER(W)) * [
    1/√n +                     // sample complexity
    κ(W) +                     // optimization difficulty
    exp(-I(t;y)/β) +          // information loss
    √(tr(WW^T)/C_opt)         // capacity utilization
]

**License**
Use, modify, and distribute as you see fit. Good luck with that shit.
Copyright 2024, nobody. No rights reserved.
"""

import torch
import json
import argparse
import shutil
from tqdm import tqdm
import os
import yaml
import numpy as np
from typing import Optional
from datetime import datetime
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
from transformers import AutoConfig


def download_model(model_name: str, models_dir: str) -> Optional[str]:
    """Download model from Hugging Face Hub."""
    local_path = os.path.join(models_dir, model_name.replace("/", "_"))
    if not os.path.exists(local_path):
        print(f"Downloading {model_name} to {local_path}")
        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                revision="main",
            )
            print(f"Successfully downloaded {model_name}")
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")
            return None
    else:
        print(f"Model {model_name} already exists at {local_path}")

    return local_path


def load_model(model_path: str, device: str = "cuda") -> Optional[AutoModelForCausalLM]:
    """Load model from local path."""
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def calculate_normalized_effective_rank(A: torch.Tensor) -> float:
    """Calculate the Normalized Effective Rank (NER) of a matrix."""
    try:
        # get the singular values
        if A.dtype != torch.float32:
            A = A.float()
        if A.dim() == 1:
            A = A.unsqueeze(0)
        if 1 in A.shape:
            S = A.abs().view(-1).cpu().numpy()
        else:
            S = torch.linalg.svdvals(A).cpu().numpy()

        # compute the normalized effective rank
        S = S[S > 1e-12]
        if S.size == 0:
            return 0.0
        S_sum = S.sum()
        S = S / S_sum
        log_S = np.log2(S)
        H = -np.dot(S, log_S)
        H_max = np.log2(S.size)
        return float(H / H_max) if H_max > 0 else 0.0

    except Exception as e:
        print(f"Error calculating NER: {e}")
        return 0.0


def normalize_tensor(A: torch.Tensor) -> torch.Tensor:
    """Normalize input tensor."""
    A_min, A_max = A.min(), A.max()
    return (A - A_min) / max(A_max - A_min, 1e-10)


def save_metrics_to_json(model_name: str, layer_metrics: dict, output_dir: str) -> None:
    model_name_slug = model_name.replace("/", "-").replace("_", "-")
    filename = os.path.join(output_dir, f"metrics_results_{model_name_slug}.json")
    with open(filename, "w") as f:
        json.dump(layer_metrics, f, indent=4)
    print(f"Metrics saved to {filename}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def metric_file_exists(model_name: str, metric_dir: str) -> bool:
    """Check if metric file already exists for the given model."""
    model_name_slug = model_name.replace("/", "-").replace("_", "-")
    filename = os.path.join(metric_dir, f"metrics_results_{model_name_slug}.json")
    return os.path.exists(filename)


def load_all_metrics(config: dict) -> dict:
    """Load all metrics from the metric directory."""
    all_metrics = {}
    for model_name in [config["base_model"]] + config["fine_tuned_models"]:
        model_name_slug = model_name.replace("/", "-").replace("_", "-")
        filename = os.path.join(
            config["metric_dir"], f"metrics_results_{model_name_slug}.json"
        )
        with open(filename, "r") as f:
            all_metrics[model_name] = json.load(f)
    return all_metrics


def identify_common_layers(all_metrics: dict) -> list:
    """Identify common layers across all models."""
    layer_sets = [set(model_metrics.keys()) for model_metrics in all_metrics.values()]
    common_layers = set.intersection(*layer_sets)
    return list(common_layers)


def identify_layers(all_metrics: dict) -> list:
    """Identify the superset of layers across all models, maintaining their relative order."""
    superset_layers = []
    added_layers = set()
    for model_metrics in all_metrics.values():
        for layer in model_metrics.keys():
            if layer not in added_layers:
                superset_layers.append(layer)
                added_layers.add(layer)
    return superset_layers


def select_best_layers(common_layers: list, all_metrics: dict) -> dict:
    """Select best layers"""
    layer_selection = {}
    for layer in common_layers:
        best_model = max(
            all_metrics.keys(), key=lambda model: all_metrics[model][layer]["ner"]
        )
        layer_selection[layer] = best_model

    print("Selected layers:")
    print(json.dumps(layer_selection, indent=4))
    return layer_selection


def save_composite_model(
    composite_model: AutoModelForCausalLM, layer_selection: dict, config: dict
) -> str:
    """Save composite model to the output directory and return the path."""
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"composite_model_{date_str}"
    output_dir = os.path.join(config["output_dir"], output_name)
    os.makedirs(output_dir, exist_ok=True)
    composite_model.save_pretrained(output_dir)
    generate_merge_report(layer_selection, output_dir, config)

    # Copy tokenizer files from the base model to the output directory
    base_model_path = os.path.join(
        config["models_dir"], config["base_model"].replace("/", "_")
    )
    tokenizer_files = ["tokenizer_config.json", "tokenizer.json", "vocab.json"]

    for file in tokenizer_files:
        src_path = os.path.join(base_model_path, file)
        dst_path = os.path.join(output_dir, file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: {file} not found in the base model directory.")

    print(f"Composite model and tokenizer files saved to: {output_dir}")
    return output_dir


def generate_merge_report(layer_selection: dict, output_dir, config: dict) -> None:
    """Generate merge report and save to the output directory."""
    report = {
        "base_model": config["base_model"],
        "fine_tuned_models": config["fine_tuned_models"],
        "layer_selection": layer_selection,
    }
    report_file = os.path.join(output_dir, "merge_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Merge report saved to {report_file}")
    print(json.dumps(report, indent=4))


def create_composite_model(
    base_model_name: str, layer_selection: dict, config: dict
) -> AutoModelForCausalLM:
    """Create composite model by merging selected layers."""
    models_dir = config["models_dir"]
    base_model_path = os.path.join(models_dir, base_model_name.replace("/", "_"))
    base_model = load_model(base_model_path)

    for layer_name, source_model_name in layer_selection.items():
        print(f"Processing: {source_model_name} - {layer_name}")
        source_model_path = os.path.join(
            models_dir, source_model_name.replace("/", "_")
        )
        source_model = load_model(source_model_path, device="cpu")

        layer_parts = layer_name.split(".")
        source_layer = source_model
        for part in layer_parts:
            source_layer = getattr(source_layer, part)
        source_layer = source_layer.to("cuda")

        target_layer = base_model
        for part in layer_parts[:-1]:
            target_layer = getattr(target_layer, part)
        setattr(target_layer, layer_parts[-1], source_layer)

        del source_model, source_layer, part, target_layer, layer_parts
        torch.cuda.empty_cache()

    return base_model


def get_num_layers(model_path: str) -> int:
    """Dynamically determine the number of layers in the model."""
    config = AutoConfig.from_pretrained(model_path)
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    elif hasattr(config, "n_layer"):
        return config.n_layer
    else:
        raise ValueError("Could not determine the number of layers in the model.")


def get_model_metrics(config: dict) -> None:
    """Get metrics for all models in the configuration."""
    models_dir = config["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)
    models = [config["base_model"]] + config["fine_tuned_models"]
    metrics = ["ner"]

    for model_name in models:
        if metric_file_exists(model_name, config["metric_dir"]):
            print(f"Metric file for {model_name} already exists. Skipping...")
            continue

        local_model_path = download_model(model_name, models_dir)
        if not local_model_path:
            print(f"Skipping failed model: {model_name}")
            continue

        layer_metrics = process_model(model_name, local_model_path, metrics, config)
        save_metrics_to_json(model_name, layer_metrics, config["metric_dir"])


def process_model(
    model_name: str, local_model_path: str, metrics: list, config: dict
) -> dict:
    """Process a single model to calculate and save metrics."""
    print(f"Processing model: {model_name}")
    with autocast(enabled=True):
        model = load_model(local_model_path)
        if not model:
            print(f"Failed to load model: {model_name}")
            return

        all_layers, layer_names = collect_and_normalize_weights(model)
        del model
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        layer_metrics = calculate_metrics_for_layers(layer_names, all_layers, metrics)
        del all_layers
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    save_metrics_to_json(model_name, layer_metrics, config["metric_dir"])
    plot_normalized_metrics(layer_metrics, model_name, config["metric_dir"])

    return layer_metrics


def collect_and_normalize_weights(
    model: AutoModelForCausalLM,
) -> tuple[list[torch.Tensor], list[str]]:
    """Collect and normalize all layers from the model (only normalize once)."""
    all_layers = [
        module.weight.data
        for name, module in model.named_modules()
        if hasattr(module, "weight")
    ]

    for i, layer in enumerate(all_layers):  # Normalize weights
        if layer.ndim < 2:
            layer = layer.unsqueeze(0)  # Make it at least 2D
        layer = normalize_tensor(layer.to(torch.float32))
        all_layers[i] = layer.to(torch.bfloat16)  # Back to bfloat16 and original device

    layer_names = [
        name for name, module in model.named_modules() if hasattr(module, "weight")
    ]
    return all_layers, layer_names


def calculate_metrics_for_layers(
    layer_names: list[str], normalized_layers: list[torch.Tensor], metrics: list[str]
) -> dict:
    """Calculate metrics for each layer."""
    layer_metrics = {}
    with torch.no_grad():
        for idx, (name, normalized_layer) in enumerate(
            tqdm(zip(layer_names, normalized_layers), desc="Processing:")
        ):
            print(f" Layer: {name}")
            layer_metrics[name] = {}

            print(f"Layer {name} shape: {normalized_layer.shape}")
            for metric in metrics:
                print(f"Calculating {metric} for layer {name}")
                try:
                    result = calculate_normalized_effective_rank(normalized_layer)
                except Exception as e:
                    print(f"Error calculating {metric} for layer {name}: {e}")
                    result = 0.0
                layer_metrics[name][metric] = result
                print(f"{metric} for layer {name}: {result}")

            torch.cuda.empty_cache()
    return layer_metrics


def normalize_metrics(metrics: dict) -> dict:
    """Normalize each metric to be between 0 and 1."""
    normalized = {metric: [] for metric in next(iter(metrics.values())).keys()}

    for metric in normalized.keys():
        values = [layer_metrics[metric] for layer_metrics in metrics.values()]
        min_val, max_val = min(values), max(values)
        normalized[metric] = [
            0 if max_val == min_val else (v - min_val) / (max_val - min_val)
            for v in values
        ]
    return normalized


def plot_normalized_metrics(metrics: dict, model_name: str, output_dir: str):
    """Plot normalized metrics for each layer and save as a PNG file."""
    normalized = normalize_metrics(metrics)
    layers = list(metrics.keys())

    plt.figure(figsize=(10, 10))  # This will give us a 768x768 pixel image at 96 DPI
    for metric, values in normalized.items():
        plt.plot(values, label=metric)

    plt.xlabel("Layers")
    plt.ylabel("Normalized Metric Value")
    plt.title(f"Normalized Metrics Across Layers - {model_name}")
    plt.legend()

    # Set x-axis ticks
    num_layers = len(layers)
    if num_layers > 20:
        step = num_layers // 10
        plt.xticks(range(0, num_layers, step), layers[::step], rotation=45, ha="right")
    else:
        plt.xticks(range(num_layers), layers, rotation=45, ha="right")

    # Save the plot as a PNG file
    plt.tight_layout()
    model_name_slug = model_name.replace("/", "-").replace("_", "-")
    filename = os.path.join(output_dir, f"metrics_plot_{model_name_slug}.png")
    plt.savefig(filename, dpi=96, bbox_inches="tight")
    plt.close()

    print(f"Metrics plot saved to {filename}")


def merge_models(config: dict) -> None:
    """Merge models based on the given configuration."""
    all_metrics = load_all_metrics(config)
    layers = identify_layers(all_metrics)
    layer_selection = select_best_layers(layers, all_metrics)
    layer_selection = dict(sorted(layer_selection.items()))
    composite_model = create_composite_model(
        config["base_model"], layer_selection, config
    )
    composite_model_path = save_composite_model(
        composite_model, layer_selection, config
    )
    return composite_model_path


def stabalize_ner_merge(config: dict, composite_model_path: str) -> str:
    """Merge using peak-value TIES approach, with refined logic to boost lower weights selectively."""
    if not config.get("stabalize_ner", True):
        return composite_model_path  # Return the original composite model path

    base_model_name = config["base_model"]
    models_dir = config["models_dir"]
    nerd_path = composite_model_path
    layer_stats = {}

    print(f"Base model: {base_model_name}")
    base_path = os.path.join(models_dir, base_model_name.replace("/", "_"))
    base_model = load_model(base_path, device="cpu")
    if base_model is None:
        print(f"Failed to load base model from {base_path}")
        return composite_model_path  # Return the original composite model path

    print(f"NERD model: {nerd_path}")
    nerd_model = load_model(nerd_path, device="cpu")
    if nerd_model is None:
        print(f"Failed to load NERD model from {nerd_path}")
        return composite_model_path  # Return the original composite model path

    print("Merging NERD model with base model to stabalize and boost ner...")
    with torch.no_grad():
        for name, nerd_module in nerd_model.named_modules():
            if hasattr(nerd_module, "weight") and nerd_module.weight is not None:

                base_module = base_model
                try:
                    for part in name.split("."):
                        base_module = getattr(base_module, part)
                except AttributeError:
                    print(f"Layer {name} not found in base model. Retaining NERD layer.")
                    continue  # Skip merging

                if not hasattr(base_module, "weight") or base_module.weight is None:
                    print(f"Layer {name} in base model has no weight. Skipping.")
                    continue  # Skip merging
                
                w1 = nerd_module.weight
                w2 = base_module.weight
                if w1.shape != w2.shape:
                    print(f"Shape mismatch at {name}: {w1.shape} vs {w2.shape}. Skipping.")
                    continue  # Skip merging

                # Replace weights in w1 with w2 if w2 is higher in magnitude and lower in value than median of w1 
                lower_med_mask = w2.abs() < w1.abs().median().item()
                mask = (w2.abs() >= w1.abs()) & lower_med_mask
                w1.data[mask] = w2.data[mask]

                # Calculate and store the percentage of weights replaced
                replaced = mask.sum().item() / mask.numel()
                layer_stats[name] = f"{replaced:.1%}"
                print(f"Layer: {name}, Base model weights used: {replaced:.2%}")

                del w2, mask, lower_med_mask
                torch.cuda.empty_cache()

    # Save merged model
    output_path = nerd_path + "_final_stabalized"
    os.makedirs(output_path, exist_ok=True)
    nerd_model.save_pretrained(output_path)

    # Save layer stats
    stats_file = os.path.join(output_path, "layer_stats.json")
    with open(stats_file, "w") as f:
        json.dump(layer_stats, f, indent=4)
    print(f"Layer stats saved to {stats_file}")

    # Copy tokenizer files
    tokenizer_files = ["tokenizer_config.json", "tokenizer.json", "vocab.json"]
    for file in tokenizer_files:
        src_path = os.path.join(base_path, file)
        dst_path = os.path.join(output_path, file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: {file} not found in the base model directory.")

    del nerd_model, base_model
    torch.cuda.empty_cache()

    return output_path


@torch.inference_mode()
def main(config_path: str) -> None:
    """Main function to run the model merging process."""
    config = load_config(config_path)

    # Stage 1: NERD merge
    get_model_metrics(config)
    print("Metric calculation completed.")
    composite_model_path = merge_models(config)
    print(f"Saved NERD composite model to: {composite_model_path}")

    # Stage 2: Modified TIES with base
    final_model_path = stabalize_ner_merge(config, composite_model_path)
    print(f"Saved final model to: {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="mastermerge: Advanced model merging tool"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="nerd_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--scorecard",
        action="store_true",
        help="Generate a scorecard showing the higher NER model for each layer based on all of the ./metrics/ files",
    )
    args = parser.parse_args()

    if args.scorecard:
        config = load_config(args.config)
        all_metrics = load_all_metrics(config)
        layers = identify_layers(all_metrics)
        layer_selection = select_best_layers(layers, all_metrics)
        layer_selection = dict(sorted(layer_selection.items()))
        print("Scorecard:")
        print(json.dumps(layer_selection, indent=4))
    else:
        main(args.config)
