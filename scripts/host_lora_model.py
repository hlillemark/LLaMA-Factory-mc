import argparse
from vllm.entrypoints.openai.api_server import main as vllm_main
import os
from huggingface_hub import snapshot_download

def download_lora_from_hf(repo_id, local_dir):
    """Download LoRA weights from Hugging Face."""
    print(f"Downloading LoRA adapter from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"LoRA adapter downloaded to {local_dir}")
    return local_dir

def main():
    parser = argparse.ArgumentParser(description="Host a model with LoRA from Hugging Face using vLLM")
    
    parser.add_argument("--base-model", type=str, required=True, 
                        help="Base model name or path (e.g., 'meta-llama/Llama-2-7b-hf')")
    parser.add_argument("--lora-repo-id", type=str, required=True,
                        help="Hugging Face repo ID for your LoRA adapter (e.g., 'username/your-lora-adapter')")
    parser.add_argument("--lora-local-dir", type=str, default="./lora-adapter",
                        help="Local directory to save the LoRA adapter")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the server on")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                        help="Fraction of GPU memory to use")
    
    args = parser.parse_args()
    
    # Download LoRA adapter from Hugging Face
    lora_dir = download_lora_from_hf(args.lora_repo_id, args.lora_local_dir)
    
    # Prepare vLLM server arguments
    vllm_args = [
        "--model", args.base_model,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--enable-lora",
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization)
    ]
    
    # Register the LoRA adapter with vLLM
    lora_modules_path = os.path.join(lora_dir, "adapter_model.safetensors")
    if not os.path.exists(lora_modules_path):
        lora_modules_path = os.path.join(lora_dir, "adapter_model.bin")
    
    os.environ["VLLM_LORA_MODULES"] = f"default:{lora_modules_path}"
    
    print(f"Starting vLLM server with base model: {args.base_model}")
    print(f"LoRA adapter loaded from: {lora_modules_path}")
    print(f"Server will run on port: {args.port}")
    
    # Start the vLLM server
    vllm_main(vllm_args)

if __name__ == "__main__":
    main()