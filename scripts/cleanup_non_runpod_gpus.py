#!/usr/bin/env python3
"""
Script to delete raw data JSON files that contain GPUs not available on RunPod
"""
import json
import os
from pathlib import Path
import sys

def get_runpod_gpu_list():
    """Return list of GPUs available on RunPod based on pricing data"""
    # From runpod_gpu_pricing.json, extract all GPU models that are available
    runpod_gpus = {
        # B200 series
        'B200',

        # H200 series
        'H200',

        # H100 series
        'H100',

        # A100 series
        'A100',

        # 48GB tier (L40/L40S/6000 Ada, A6000/A40)
        'L40', 'L40S', '6000 Ada',
        'A6000', 'A40',

        # 32GB tier (5090)
        '5090',

        # 24GB tier (4090, L4/A5000/3090)
        '4090',
        'L4', 'A5000', '3090',

        # 16GB tier (A4000/A4500/RTX 4000/RTX 2000)
        'A4000', 'A4500', 'RTX 4000', 'RTX 2000'
    }

    # Convert to full NVIDIA naming patterns that appear in data
    full_names = set()
    for gpu in runpod_gpus:
        # Add common NVIDIA naming patterns
        full_names.add(f"NVIDIA {gpu}")
        full_names.add(f"NVIDIA RTX {gpu}")
        full_names.add(f"NVIDIA GeForce RTX {gpu}")
        full_names.add(f"NVIDIA RTX A{gpu[1:]}" if gpu.startswith('A') and gpu[1:].isdigit() else f"NVIDIA RTX {gpu}")

        # Handle special cases
        if gpu == '3090':
            full_names.add("NVIDIA GeForce RTX 3090")
        elif gpu == 'A5000':
            full_names.add("NVIDIA RTX A5000")
        elif gpu == 'A4000':
            full_names.add("NVIDIA RTX A4000")
        elif gpu == 'A4500':
            full_names.add("NVIDIA RTX A4500")

    return full_names

def is_runpod_gpu(gpu_name, runpod_gpus):
    """Check if a GPU name matches any RunPod available GPU"""
    if not gpu_name:
        return False

    # Direct match
    if gpu_name in runpod_gpus:
        return True

    # Partial matching for variations
    gpu_name_clean = gpu_name.strip().upper()
    for runpod_gpu in runpod_gpus:
        if runpod_gpu.upper() in gpu_name_clean or gpu_name_clean in runpod_gpu.upper():
            return True

    return False

def analyze_file(file_path, runpod_gpus):
    """Analyze a JSON file and return GPU info"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different JSON structures
        if data is None:
            return None

        gpu_info = data.get('gpuInfo', {}) if isinstance(data, dict) else {}
        active_gpu = gpu_info.get('active_gpu_device', '') if gpu_info else ''

        # If no GPU info, skip file (might be request file or different format)
        if not active_gpu:
            return None

        return {
            'file': file_path,
            'gpu': active_gpu,
            'is_runpod': is_runpod_gpu(active_gpu, runpod_gpus),
            'size': os.path.getsize(file_path)
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    """Main function"""
    raw_data_dir = Path("../raw_data")

    if not raw_data_dir.exists():
        print(f"Raw data directory not found: {raw_data_dir}")
        return

    # Get RunPod GPU list
    runpod_gpus = get_runpod_gpu_list()
    print(f"RunPod GPUs ({len(runpod_gpus)}): {sorted(list(runpod_gpus)[:5])}...")  # Show first 5

    # Find all JSON response files
    json_files = list(raw_data_dir.glob("*-response.json"))
    print(f"\nFound {len(json_files)} response JSON files")

    # Analyze files
    results = []
    for file_path in json_files:
        result = analyze_file(file_path, runpod_gpus)
        if result:
            results.append(result)

    # Separate RunPod vs non-RunPod files
    runpod_files = [r for r in results if r['is_runpod']]
    non_runpod_files = [r for r in results if not r['is_runpod']]

    print(f"\n=== ANALYSIS RESULTS ===")
    print(f"RunPod GPU files: {len(runpod_files)}")
    print(f"Non-RunPod GPU files: {len(non_runpod_files)}")

    # Show GPU breakdown
    gpu_counts = {}
    for r in results:
        gpu = r['gpu']
        gpu_counts[gpu] = gpu_counts.get(gpu, 0) + 1

    print(f"\n=== GPU BREAKDOWN ===")
    for gpu, count in sorted(gpu_counts.items()):
        status = "[RunPod]" if is_runpod_gpu(gpu, runpod_gpus) else "[Not RunPod]"
        print(f"{gpu}: {count} files {status}")

    # Calculate sizes
    non_runpod_size = sum(r['size'] for r in non_runpod_files)
    total_size = sum(r['size'] for r in results)

    print(f"\n=== SIZE ANALYSIS ===")
    print(f"Non-RunPod files size: {non_runpod_size / 1024 / 1024:.1f} MB")
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"Space to be freed: {non_runpod_size / 1024 / 1024:.1f} MB ({non_runpod_size/total_size*100:.1f}%)")

    if non_runpod_files:
        print(f"\n=== FILES TO DELETE ===")
        print("The following files contain GPUs not available on RunPod:")
        for r in non_runpod_files[:10]:  # Show first 10
            print(f"  {r['file'].name} - {r['gpu']}")
        if len(non_runpod_files) > 10:
            print(f"  ... and {len(non_runpod_files) - 10} more files")

        # Auto-delete mode for non-interactive environments
        if len(sys.argv) > 1 and sys.argv[1] == '--delete':
            print(f"\n[AUTO-DELETE] Deleting all {len(non_runpod_files)} non-RunPod GPU files...")
            should_delete = True
        else:
            try:
                response = input(f"\nDelete all {len(non_runpod_files)} non-RunPod GPU files? (y/N): ")
                should_delete = response.lower() == 'y'
            except EOFError:
                print(f"\n[AUTO-DELETE] Non-interactive mode detected. Deleting all {len(non_runpod_files)} non-RunPod GPU files...")
                should_delete = True

        if should_delete:
            deleted_count = 0
            for r in non_runpod_files:
                try:
                    os.remove(r['file'])
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {r['file']}: {e}")

            print(f"\nDeleted {deleted_count} files ({non_runpod_size / 1024 / 1024:.1f} MB freed)")
        else:
            print("No files deleted.")
    else:
        print("\n[OK] All files contain RunPod-compatible GPUs!")

if __name__ == "__main__":
    main()