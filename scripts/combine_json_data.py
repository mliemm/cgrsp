#!/usr/bin/env python3
"""
Script to combine all response JSON files into a single JSON file and generate CSV
"""

import orjson
import os
import polars as pl
from pathlib import Path
import re
from datetime import datetime

def extract_model_id_from_path(render_paths):
    """Extract model ID from the render paths"""
    try:
        model_path = render_paths.get('Model', {}).get('model', '')
        # Extract number from path like './models/41.obj'
        match = re.search(r'/(\d+)\.obj', model_path)
        if match:
            return int(match.group(1))
    except:
        pass
    return None

def combine_response_files(data_dir="../raw_data"):
    """Combine all response JSON files into a single structured dataset"""
    
    all_data = []
    raw_data = []
    
    # Get all response files
    data_path = Path(data_dir)
    response_files = list(data_path.glob("*-response.json"))
    
    print(f"Processing {len(response_files)} response files...")
    
    for file_path in response_files:
        try:
            with open(file_path, 'rb') as f:
                data = orjson.loads(f.read())
            
            # Add filename for traceability
            data['source_file'] = file_path.name
            raw_data.append(data)
            
            # Skip failed jobs for processed data
            if not data.get('isSuccess', False):
                continue
                
            # Extract and structure relevant information
            material_id = data.get('materialU3mId', None)
            gpu_used = data.get('gpuInfo', {}).get('active_gpu_device', 'Unknown')
            
            # Image dimensions
            image_info = data.get('imageInfo', {})
            width = image_info.get('width', 0)
            height = image_info.get('height', 0)
            dpi = image_info.get('dpi', 0)
            
            # Times
            total_time = data.get('executionTime', 0)
            render_time = data.get('renderTime', 0)
            import_time = data.get('importTime', 0)
            
            # Model ID from render paths
            model_id = extract_model_id_from_path(data.get('render_paths', {}))
            if model_id is None:
                # Try from job_input
                job_input = data.get('job_input', {})
                if 'model' in job_input:
                    try:
                        model_id = int(job_input['model'])
                    except:
                        model_id = None
            
            # GPU info details
            gpu_info = data.get('gpuInfo', {})
            compute_device_type = gpu_info.get('compute_device_type', 'Unknown')
            
            # Environment
            env = data.get('env', 'Unknown')
            
            # Cost calculations using RunPod Flex pricing (per second)
            gpu_cost_per_second = {
                'NVIDIA RTX A5000': 0.00019,  # 24GB - L4, A5000, 3090 category
                'NVIDIA RTX A4000': 0.00016,  # 16GB - A4000, A4500, RTX 4000, RTX 2000 category
                'NVIDIA GeForce RTX 3090': 0.00019,  # 24GB - L4, A5000, 3090 category
                'NVIDIA RTX A4500': 0.00016,  # 16GB - A4000, A4500, RTX 4000, RTX 2000 category
            }
            
            cost_per_second = gpu_cost_per_second.get(gpu_used, 0.00019)  # Default to A5000 rate
            cost_per_hour = cost_per_second * 3600  # Convert to per hour for reference
            total_cost = total_time * cost_per_second
            render_cost = render_time * cost_per_second
            
            # File timestamp for temporal analysis
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', file_path.name)
            timestamp = timestamp_match.group(1).replace('-', ':') if timestamp_match else None
            
            # Create structured record
            processed_record = {
                'source_file': file_path.name,
                'timestamp': timestamp,
                'material_id': material_id,
                'gpu_used': gpu_used,
                'compute_device_type': compute_device_type,
                'environment': env,
                'model_id': model_id,
                'image_width': width,
                'image_height': height,
                'image_dimensions': f"{width}x{height}",
                'image_dpi': dpi,
                'total_pixels': width * height,
                'total_time_s': round(total_time, 2),
                'render_time_s': round(render_time, 2),
                'import_time_s': round(import_time, 2),
                'render_ratio': round(render_time / total_time, 3) if total_time > 0 else 0,
                'cost_per_second': cost_per_second,
                'cost_per_hour': cost_per_hour,
                'total_cost': round(total_cost, 4),
                'render_cost': round(render_cost, 4),
                'cost_per_megapixel': round(total_cost / (width * height / 1000000), 4) if width * height > 0 else 0,
                'is_success': data.get('isSuccess', False)
            }
            
            all_data.append(processed_record)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return all_data, raw_data

def save_combined_data(processed_data, raw_data):
    """Save the combined data as JSONL and CSV files"""

    print(f"\nSaving data for {len(processed_data)} successful jobs...")

    # Ensure output directory exists
    output_dir = Path("../outputs")
    output_dir.mkdir(exist_ok=True)

    # 1. Save processed data as JSONL (one JSON object per line)
    combined_jsonl_path = output_dir / 'combined_rendering_data.jsonl'
    with open(combined_jsonl_path, 'wb') as f:
        for record in processed_data:
            f.write(orjson.dumps(record) + b'\n')
    print(f"- Saved processed data to: {combined_jsonl_path}")

    # 2. Save raw data as JSONL (for reference)
    raw_jsonl_path = output_dir / 'raw_rendering_responses.jsonl'
    with open(raw_jsonl_path, 'wb') as f:
        for record in raw_data:
            f.write(orjson.dumps(record) + b'\n')
    print(f"- Saved raw data to: {raw_jsonl_path}")

    # 3. Convert to Polars DataFrame and save as CSV
    df = pl.DataFrame(processed_data)

    # Sort by timestamp and material_id
    df = df.sort(['timestamp', 'material_id'])

    # Save as CSV
    csv_path = output_dir / 'rendering_performance_data.csv'
    df.write_csv(csv_path)
    print(f"- Saved CSV to: {csv_path}")

    return df

def generate_summary_stats(df):
    """Generate summary statistics"""
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    print(f"Total successful jobs: {len(df)}")
    print(f"Unique materials: {df['material_id'].n_unique()}")
    print(f"GPU types: {df['gpu_used'].n_unique()}")
    print(f"Model IDs: {sorted(df['model_id'].unique().to_list())}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    print("\nGPU Distribution:")
    gpu_counts = df.group_by('gpu_used').agg(pl.len()).sort('len', descending=True)
    for row in gpu_counts.iter_rows():
        print(f"{row[0]}: {row[1]}")

    print("\nModel ID Distribution:")
    model_counts = df.group_by('model_id').agg(pl.len()).sort('model_id')
    for row in model_counts.iter_rows():
        print(f"{row[0]}: {row[1]}")
    
    print(f"\nPerformance Range:")
    print(f"Total time: {df['total_time_s'].min():.1f}s - {df['total_time_s'].max():.1f}s")
    print(f"Render time: {df['render_time_s'].min():.1f}s - {df['render_time_s'].max():.1f}s")
    print(f"Total cost: ${df['total_cost'].min():.3f} - ${df['total_cost'].max():.3f}")
    
    print(f"\nResolution Range:")
    print(f"Top dimensions:")
    dim_counts = df.group_by('image_dimensions').agg(pl.len()).sort('len', descending=True).head(5)
    for row in dim_counts.iter_rows():
        print(f"  {row[0]}: {row[1]}")

def main():
    """Main function"""
    
    print("Combining response JSON files...")
    print("="*50)
    
    # Process all files
    processed_data, raw_data = combine_response_files()
    
    if not processed_data:
        print("No valid data found!")
        return
    
    # Save combined data
    df = save_combined_data(processed_data, raw_data)
    
    # Generate summary
    generate_summary_stats(df)
    
    print(f"\n- Processing complete!")
    print(f"Generated files:")
    print(f"  - combined_rendering_data.jsonl ({len(processed_data)} records)")
    print(f"  - raw_rendering_responses.jsonl ({len(raw_data)} records)")
    print(f"  - rendering_performance_data.csv ({len(df)} rows)")

if __name__ == "__main__":
    main()