from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import json
from tqdm import tqdm
from datasets import Dataset, load_dataset
import pandas as pd
import os
from typing import List, Dict, Any
import math
import argparse

def setup_distributed(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def format_prompt_to_template(prompt_data):
    """Format prompt data to match the instruction/input/output template"""
    
    if isinstance(prompt_data, dict):
        # If it already has the correct format
        if 'instruction' in prompt_data and 'input' in prompt_data:
            return prompt_data
        
        # Convert other formats to the template
        elif 'prompt' in prompt_data and 'input' in prompt_data:
            return {
                "instruction": prompt_data['prompt'],
                "input": prompt_data['input'],
                "output": prompt_data.get('output', '')
            }
        
        # Handle case where there's just instruction
        elif 'instruction' in prompt_data:
            return {
                "instruction": prompt_data['instruction'],
                "input": prompt_data.get('input', ''),
                "output": prompt_data.get('output', '')
            }
        
        # Fallback for other dictionary formats
        else:
            # Try to guess the instruction from common keys
            instruction = (prompt_data.get('question') or 
                          prompt_data.get('task') or 
                          prompt_data.get('prompt') or 
                          'Complete the following task:')
            
            input_text = (prompt_data.get('input') or 
                         prompt_data.get('context') or 
                         prompt_data.get('text') or '')
            
            return {
                "instruction": instruction,
                "input": input_text,
                "output": prompt_data.get('output', '')
            }
    
    # If it's a string, treat it as instruction
    elif isinstance(prompt_data, str):
        return {
            "instruction": prompt_data,
            "input": "",
            "output": ""
        }
    
    else:
        return {
            "instruction": str(prompt_data),
            "input": "",
            "output": ""
        }

def create_formatted_prompt_string(formatted_data, include_output=False):
    """Convert the formatted data into a string for the model"""
    
    prompt_parts = []
    
    # Add instruction
    if formatted_data.get('instruction'):
        prompt_parts.append(f"Instruction: {formatted_data['instruction']}")
    
    # Add input if it exists and is not empty
    if formatted_data.get('input') and formatted_data['input'].strip():
        prompt_parts.append(f"Input: {formatted_data['input']}")
    
    # Add output section
    if include_output and formatted_data.get('output'):
        prompt_parts.append(f"Output: {formatted_data['output']}")
    else:
        prompt_parts.append("Output:")
    
    return '\n\n'.join(prompt_parts)

def process_conversation_input(conversation_data):
    """Special handler for conversation-style inputs like in your example"""
    
    if isinstance(conversation_data, list):
        # Extract the conversation context
        conversation_text = ""
        for turn in conversation_data:
            role = turn.get('role', '')
            content = turn.get('content', '')
            if role and content:
                conversation_text += f"{role}: {content}\n"
        
        return conversation_text.strip()
    
    return str(conversation_data)

def prepare_dataset(data_file_path):
    """Load and prepare dataset from JSON file"""
    
    # Load JSON data
    with open(data_file_path) as f:
        json_data = json.load(f)
    
    # Convert to Dataset
    if isinstance(json_data, list):
        dataset = Dataset.from_list(json_data)
    else:
        # If it's a dict, assume it's already in dataset format
        dataset = Dataset.from_dict(json_data)
    
    return dataset

def format_prompt_for_generation(example):
    """Function to apply to each example in the dataset"""
    
    # Handle the specific format from your example
    if 'instruction' in example and 'input' in example:
        instruction = example['instruction']
        
        # Process input (might be conversation format)
        input_data = example['input']
        if isinstance(input_data, str):
            try:
                # Try to parse as JSON if it's a string representation
                parsed_input = json.loads(input_data)
                formatted_input = process_conversation_input(parsed_input)
            except:
                # If not JSON, use as is
                formatted_input = input_data
        else:
            formatted_input = process_conversation_input(input_data)
        
        # Create the full prompt
        full_prompt = f"{instruction}\n\nConversation History:\n{formatted_input}\n\nResponse:"
        
    else:
        # Use the standard template formatting
        formatted_template = format_prompt_to_template(example)
        full_prompt = create_formatted_prompt_string(formatted_template)
    
    # Add the formatted prompt to the example
    example['formatted_prompt'] = full_prompt
    return example

class MultiGPUTextGenerator:
    """Multi-GPU text generator using model parallelism"""
    
    def __init__(self, model_name: str, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.padding_side = "left"
        
        # Load model on specific GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=f"cuda:{rank}",
            trust_remote_code=True
        )
        
        # Wrap model with DDP
        self.model = DDP(self.model, device_ids=[rank])
        self.model.eval()
    
    def generate_batch(self, prompts: List[str], max_length: int = 8000, 
                      temperature: float = 0.7, do_sample: bool = True) -> List[str]:
        """Generate responses for a batch of prompts"""
        
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=max_length // 2  # Leave room for generation
        ).to(f"cuda:{self.rank}")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.module.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode outputs
        responses = []
        for i, output in enumerate(outputs):
            # Remove input tokens to get only the generated part
            input_length = inputs['input_ids'][i].shape[0]
            generated_tokens = output[input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response.strip())
        
        return responses

def split_dataset_for_rank(dataset: Dataset, rank: int, world_size: int) -> Dataset:
    """Split dataset across multiple GPUs"""
    
    total_size = len(dataset)
    per_gpu_size = math.ceil(total_size / world_size)
    start_idx = rank * per_gpu_size
    end_idx = min(start_idx + per_gpu_size, total_size)
    
    # Create subset for this rank
    indices = list(range(start_idx, end_idx))
    return dataset.select(indices)

def generate_responses_worker(rank: int, world_size: int, data_file_path: str, 
                            model_name: str, batch_size: int, save_path: str,
                            generation_params: Dict[str, Any]):
    """Worker function for each GPU process"""
    
    try:
        # Setup distributed environment
        setup_distributed(rank, world_size)
        
        if rank == 0:
            print(f"Setting up multi-GPU generation with {world_size} GPUs")
        
        # Load and prepare dataset
        if rank == 0:
            print("Loading and preparing dataset...")
        
        dataset = prepare_dataset(data_file_path)
        
        # Format prompts
        dataset = dataset.map(
            format_prompt_for_generation, 
            desc=f"Formatting prompts (GPU {rank})" if rank == 0 else None
        )
        
        # Split dataset for this rank
        rank_dataset = split_dataset_for_rank(dataset, rank, world_size)
        
        if rank == 0:
            print(f"Total dataset size: {len(dataset)}")
            print(f"Processing {len(rank_dataset)} examples on GPU {rank}")
        
        # Initialize generator for this GPU
        generator = MultiGPUTextGenerator(model_name, rank, world_size)
        
        # Process data in batches
        results = []
        num_batches = math.ceil(len(rank_dataset) / batch_size)
        
        for i in tqdm(range(0, len(rank_dataset), batch_size), 
                     desc=f"Generating on GPU {rank}", 
                     disable=rank != 0):
            
            batch_data = rank_dataset[i:i+batch_size]
            
            # Handle single example vs batch
            if not isinstance(batch_data['formatted_prompt'], list):
                batch_data = {k: [v] for k, v in batch_data.items()}
            
            prompts = batch_data['formatted_prompt']
            
            # Generate responses
            responses = generator.generate_batch(
                prompts, 
                max_length=generation_params.get('max_length', 8000),
                temperature=generation_params.get('temperature', 0.7),
                do_sample=generation_params.get('do_sample', True)
            )
            
            # Create result entries
            for j, response in enumerate(responses):
                if j < len(batch_data['instruction']):  # Safety check
                    result = {
                        "instruction": batch_data['instruction'][j],
                        "input": batch_data['input'][j],
                        "dispreferred": response,
                        "preferred": batch_data.get('output', [''])[j] if isinstance(batch_data.get('output', ['']), list) else batch_data.get('output', ''),
                        "formatted_prompt": batch_data['formatted_prompt'][j]
                    }
                    results.append(result)
        
        # Save results for this rank
        rank_save_path = f"{save_path}_rank_{rank}.json"
        with open(rank_save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if rank == 0:
            print(f"GPU {rank} completed processing {len(results)} examples")
        
        # Wait for all processes to complete
        dist.barrier()
        
        # Combine results on rank 0
        if rank == 0:
            print("Combining results from all GPUs...")
            all_results = []
            
            for r in range(world_size):
                rank_file = f"{save_path}_rank_{r}.json"
                if os.path.exists(rank_file):
                    with open(rank_file, 'r') as f:
                        rank_results = json.load(f)
                        all_results.extend(rank_results)
                    # Clean up individual rank files
                    os.remove(rank_file)
            
            # Save combined results
            with open(save_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Calculate statistics
            same_response_count = sum(1 for result in all_results 
                                    if result['dispreferred'] == result['preferred'])
            total_count = len(all_results)
            
            print(f"Generation completed!")
            print(f"Total examples processed: {total_count}")
            print(f"Same responses: {same_response_count}/{total_count} ({same_response_count/total_count*100:.2f}%)")
            print(f"Results saved to {save_path}")
        
    except Exception as e:
        print(f"Error in GPU {rank}: {str(e)}")
        raise e
    
    finally:
        cleanup_distributed()

def generate_with_multi_gpu(data_file_path: str, model_name: str, save_path: str,
                           batch_size: int = 4, world_size: int = None,
                           generation_params: Dict[str, Any] = None):
    """Main function to run multi-GPU generation"""
    
    if world_size is None:
        world_size = torch.cuda.device_count()
    
    if world_size == 0:
        raise ValueError("No CUDA devices available")
    
    if generation_params is None:
        generation_params = {
            'max_length': 8000,
            'temperature': 0.7,
            'do_sample': True
        }
    
    print(f"Starting multi-GPU generation with {world_size} GPUs")
    print(f"Model: {model_name}")
    print(f"Data file: {data_file_path}")
    print(f"Batch size per GPU: {batch_size}")
    print(f"Generation parameters: {generation_params}")
    
    # Spawn processes for each GPU
    mp.spawn(
        generate_responses_worker,
        args=(world_size, data_file_path, model_name, batch_size, save_path, generation_params),
        nprocs=world_size,
        join=True
    )

def generate_single_gpu_fallback(data_file_path: str, model_name: str, save_path: str,
                                batch_size: int = 8, generation_params: Dict[str, Any] = None):
    """Fallback to single GPU if multi-GPU setup fails"""
    
    print("Falling back to single GPU generation...")
    
    if generation_params is None:
        generation_params = {
            'max_length': 8000,
            'temperature': 0.7,
            'do_sample': True
        }
    
    # Load dataset
    dataset = prepare_dataset(data_file_path)
    dataset = dataset.map(format_prompt_for_generation, desc="Formatting prompts")
    
    # Initialize pipeline
    generator = pipeline(
        'text-generation', 
        model=model_name, 
        tokenizer=model_name,
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Generate responses
    results = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating responses"):
        batch = dataset[i:i+batch_size]
        
        if not isinstance(batch['formatted_prompt'], list):
            batch = {k: [v] for k, v in batch.items()}
        
        prompts = batch['formatted_prompt']
        
        generated_outputs = generator(
            prompts,
            max_length=generation_params['max_length'],
            temperature=generation_params['temperature'],
            do_sample=generation_params['do_sample'],
            truncation=True,
            pad_token_id=generator.tokenizer.eos_token_id,
            batch_size=len(prompts)
        )
        
        for j, output in enumerate(generated_outputs):
            if isinstance(output, list):
                output = output[0]
            
            full_output = output['generated_text']
            prompt_length = len(prompts[j])
            response_only = full_output[prompt_length:].strip()
            
            result = {
                "instruction": batch['instruction'][j],
                "input": batch['input'][j],
                "dispreferred": response_only,
                "preferred": batch.get('output', [''])[j] if isinstance(batch.get('output', ['']), list) else batch.get('output', ''),
                "formatted_prompt": batch['formatted_prompt'][j]
            }
            results.append(result)
    
    # Save results
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Single GPU generation completed. Results saved to {save_path}")
    return results

# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_file_path', type=str, default="data/custom/cooking_sft_success_new_mem.json",
                        help='Path to the data file')
    parser.add_argument('--save_path', type=str, default="data/custom/dpo_pairs_cooking.json",
                        help='Path to save the generated pairs')
    parser.add_argument('--model_name', type=str, default="izzcw/large_cooking_sft_fail",
                        help='Name of the model to use for generation')

    args = parser.parse_args()
    data_file_path = args.data_file_path
    save_path = args.save_path
    model_name = args.model_name
    
    generation_params = {
        'max_length': 8000,
        'temperature': 0.7,
        'do_sample': True
    }
    
    try:
        # Try multi-GPU generation
        generate_with_multi_gpu(
            data_file_path=data_file_path,
            model_name=model_name,
            save_path=save_path,
            batch_size=2,  # Smaller batch size per GPU for memory efficiency
            world_size=None,  # Auto-detect number of GPUs
            generation_params=generation_params
        )
        
    except Exception as e:
        print(f"Multi-GPU generation failed: {str(e)}")
        print("Falling back to single GPU...")
        
        # Fallback to single GPU
        results = generate_single_gpu_fallback(
            data_file_path=data_file_path,
            model_name=model_name,
            save_path=save_path,
            batch_size=4,
            generation_params=generation_params
        )
        
        # Print sample results
        print("\nSample results:")
        for i in range(min(3, len(results))):
            result = results[i]
            print(f"\nExample {i+1}:")
            print(f"Instruction: {result['instruction'][:100]}...")
            print(f"Input: {str(result['input'])[:100]}...")
            print(f"Dispreferred: {result['dispreferred'][:100]}...")
            print(f"Preferred: {result['preferred'][:100]}...")
            print("-" * 80)