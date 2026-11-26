#!/usr/bin/env python3
"""
Script to process rendering job data from JSON response files
and generate a summary similar to Rendering_Job_Summary.md
"""

import json
import os
import pandas as pd
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

def process_response_files(data_dir="data"):
    """Process all response JSON files and extract rendering performance data"""
    
    rendering_data = []
    
    # Get all response files
    data_path = Path(data_dir)
    response_files = list(data_path.glob("*-response.json"))
    
    print(f"Found {len(response_files)} response files to process...")
    
    for file_path in response_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Skip failed jobs
            if not data.get('isSuccess', False):
                continue
                
            # Extract relevant information
            material_id = data.get('materialU3mId', 'Unknown')
            gpu_used = data.get('gpuInfo', {}).get('active_gpu_device', 'Unknown')
            
            # Image dimensions
            image_info = data.get('imageInfo', {})
            width = image_info.get('width', 0)
            height = image_info.get('height', 0)
            
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
                        model_id = 'Unknown'
            
            # Add to data list
            rendering_data.append({
                'Material ID': material_id,
                'GPU Used': gpu_used,
                'Image Dimensions': f"{width}x{height}",
                'Model ID': model_id,
                'Total Time (s)': round(total_time, 2),
                'Render Time (s)': round(render_time, 2),
                'Import Time (s)': round(import_time, 2)
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return rendering_data

def generate_summary_table(rendering_data):
    """Generate a markdown table similar to Rendering_Job_Summary.md"""
    
    if not rendering_data:
        print("No rendering data found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(rendering_data)
    
    # Sort by Material ID
    df = df.sort_values('Material ID')
    
    # Generate markdown content
    markdown_content = """### Rendering Job Performance Summary

This table summarizes the performance metrics for a series of rendering jobs. Each job is identified by its `materialU3mId`. The data includes the GPU used, the output image dimensions, and a breakdown of the processing time into total execution, asset import, and rendering.

| Material ID | GPU Used | Image Dimensions | Model ID | Total Time (s) | Render Time (s) | Import Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    
    # Add data rows
    for _, row in df.iterrows():
        markdown_content += f"| {row['Material ID']} | {row['GPU Used']} | {row['Image Dimensions']} | {row['Model ID']} | {row['Total Time (s)']} | {row['Render Time (s)']} | {row['Import Time (s)']} |\n"
    
    # Write to file
    output_file = "Processed_Rendering_Job_Summary.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Generated summary table with {len(df)} jobs in {output_file}")
    
    return df

def analyze_performance_by_gpu(df):
    """Analyze performance differences between GPUs"""
    
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS BY GPU")
    print("="*60)
    
    # Group by GPU and calculate averages
    gpu_stats = df.groupby('GPU Used').agg({
        'Total Time (s)': ['mean', 'std', 'count'],
        'Render Time (s)': ['mean', 'std'],
        'Import Time (s)': ['mean', 'std']
    }).round(2)
    
    print(gpu_stats)
    
    # Analyze by Model ID and GPU
    print("\n" + "-"*60)
    print("PERFORMANCE BY MODEL ID AND GPU")
    print("-"*60)
    
    model_gpu_stats = df.groupby(['Model ID', 'GPU Used'])['Render Time (s)'].agg(['mean', 'count']).round(2)
    print(model_gpu_stats)
    
    # Calculate cost comparison for A5000 vs A4000
    print("\n" + "-"*60)
    print("COST ANALYSIS (RunPod Flex: A5000: $0.00019/s, A4000: $0.00016/s)")
    print("-"*60)
    
    a5000_data = df[df['GPU Used'].str.contains('A5000', na=False)]
    a4000_data = df[df['GPU Used'].str.contains('A4000', na=False)]
    
    if not a5000_data.empty and not a4000_data.empty:
        for model_id in df['Model ID'].unique():
            if model_id == 'Unknown':
                continue
                
            a5000_model = a5000_data[a5000_data['Model ID'] == model_id]['Render Time (s)']
            a4000_model = a4000_data[a4000_data['Model ID'] == model_id]['Render Time (s)']
            
            if not a5000_model.empty and not a4000_model.empty:
                a5000_avg = a5000_model.mean()
                a4000_avg = a4000_model.mean()
                
                a5000_cost = a5000_avg * 0.00019
                a4000_cost = a4000_avg * 0.00016
                
                time_diff = a4000_avg - a5000_avg
                cost_diff = a5000_cost - a4000_cost
                
                print(f"Model {model_id}:")
                print(f"  A5000: {a5000_avg:.1f}s (${a5000_cost:.3f})")
                print(f"  A4000: {a4000_avg:.1f}s (${a4000_cost:.3f})")
                print(f"  Time difference: {time_diff:.1f}s ({time_diff/a5000_avg*100:.1f}% slower on A4000)")
                print(f"  Cost difference: ${cost_diff:.3f} ({cost_diff/a4000_cost*100:.1f}% more for A5000)")
                print()

def main():
    """Main function to process data and generate summaries"""
    
    print("Processing rendering job data...")
    print("="*50)
    
    # Process all response files
    rendering_data = process_response_files()
    
    if not rendering_data:
        print("No valid rendering data found!")
        return
    
    # Generate summary table
    df = generate_summary_table(rendering_data)
    
    # Perform analysis
    analyze_performance_by_gpu(df)
    
    print(f"\nProcessing complete! Generated summary from {len(rendering_data)} rendering jobs.")

if __name__ == "__main__":
    main()