#!/usr/bin/env python3
"""
Memory Analysis Tool for Wan2.1 Enhanced

This script analyzes memory profiler logs and generates detailed reports
about memory consumption for different model components.
"""

import json
import argparse
import os
from collections import defaultdict

def analyze_log_file(log_file):
    """Analyzes a single log file and returns memory and time data."""
    with open(log_file, 'r') as f:
        content = f.read().strip()
        
    # Try to parse as JSON array first, then fall back to JSONL format
    try:
        if content.startswith('['):
            # JSON array format
            events = json.loads(content)
        else:
            # JSONL format - each line is a separate JSON object
            events = []
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line: {line[:50]}... Error: {e}")
                        continue
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file {log_file}: {e}")
        return {}
    
    config_name = os.path.splitext(os.path.basename(log_file))[0]
    return {config_name: parse_events(events)}

def parse_events(events):
    """Parses events from a single log file to extract memory and time metrics."""
    memory_data = defaultdict(lambda: {'base': 0, 'peak': 0, 'activation': 0, 'runtime': 0, 'kv_cache': 0})
    time_data = {}
    durations = {}
    
    for event in events:
        event_name = event.get('event', '')
        model_name = event.get('model_name')
        
        if 'timestamp' in event:
            time_data[event_name] = event['timestamp']

        if 'duration' in event:
            durations[event_name] = event['duration']

        # Handle events without model_name but with memory info
        if not model_name:
            # Try to infer model from event name
            if 'dit' in event_name.lower() or 'step_' in event_name:
                model_name = 'DiT'
            elif 't5' in event_name.lower():
                model_name = 'T5'
            elif 'vae' in event_name.lower():
                model_name = 'VAE'
            else:
                continue
        
        # Remap t2v model_name to VAE for VAE-related events
        if model_name == 't2v' and ('vae' in event_name.lower() or 'decode' in event_name.lower()):
            model_name = 'VAE'
        # Skip t2v entries that are not VAE-related
        elif model_name == 't2v':
            continue

        # Parse memory events more comprehensively
        if 'load' in event_name or 'loaded' in event_name or 'init' in event_name:
            # Use incremental_memory for base memory, fallback to 0 if not available
            incremental_mb = event.get('incremental_memory', 0)
            if incremental_mb > 0:
                memory_data[model_name]['base'] = incremental_mb / (1024 * 1024)  # Convert bytes to MB
            else:
                memory_data[model_name]['base'] = event.get('incremental_memory_mb', 0)
        
        elif 'forward' in event_name or 'encode' in event_name or 'decode' in event_name or 'step_' in event_name or 'after_decode' in event_name:
            # Use incremental_memory for activation memory during forward/encode/decode/steps
            incremental_mb = event.get('incremental_memory', 0)
            if incremental_mb != 0:  # Changed from > 0 to != 0 to handle negative values like after_decode
                activation_memory = abs(incremental_mb) / (1024 * 1024)  # Convert bytes to MB, use absolute value
                # For step events, attribute to DiT since they occur during DiT forward pass
                # For after_decode events, attribute to VAE since they occur after VAE decode
                if 'step_' in event_name:
                    target_model = 'DiT'
                elif 'after_decode' in event_name:
                    target_model = 'VAE'
                else:
                    target_model = model_name
                
                # For step events, accumulate activation memory instead of just taking max
                # For after_decode events, treat as VAE activation memory
                if 'step_' in event_name:
                    memory_data[target_model]['activation'] += activation_memory
                elif 'after_decode' in event_name:
                    memory_data[target_model]['activation'] = max(
                        memory_data[target_model]['activation'], 
                        activation_memory
                    )
                else:
                    memory_data[target_model]['peak'] = max(
                        memory_data[target_model]['peak'], 
                        activation_memory
                    )
            else:
                # Fallback to incremental_memory_mb if available
                activation_memory = event.get('incremental_memory_mb', 0)
                if activation_memory > 0:
                    if 'step_' in event_name:
                        target_model = 'DiT'
                    elif 'after_decode' in event_name:
                        target_model = 'VAE'
                    else:
                        target_model = model_name
                    
                    if 'step_' in event_name:
                        memory_data[target_model]['activation'] += activation_memory
                    elif 'after_decode' in event_name:
                        memory_data[target_model]['activation'] = max(
                            memory_data[target_model]['activation'], 
                            activation_memory
                        )
                    else:
                        memory_data[target_model]['peak'] = max(
                            memory_data[target_model]['peak'], 
                            activation_memory
                        )

        elif 'kv_cache' in event_name:
            # KV Cache memory handling
            incremental_mb = event.get('incremental_memory', 0)
            if incremental_mb > 0:
                kv_memory = incremental_mb / (1024 * 1024)  # Convert bytes to MB
            else:
                kv_memory = event.get('incremental_memory_mb', 0)
            
            # Attribute KV cache to the appropriate model
            target_model = event.get('model_name', 'DiT')
            if kv_memory > 0:
                memory_data[target_model]['kv_cache'] = max(
                    memory_data[target_model]['kv_cache'], 
                    kv_memory
                )
                # For T5 model, also treat kv_cache as activation memory since T5 encode events don't have incremental_memory
                if target_model == 'T5':
                    memory_data[target_model]['activation'] += kv_memory

    # Calculate pure runtime overhead (excluding model components)
    # Get total peak memory from events
    total_peak_memory = 0
    for event in events:
        peak_memory = event.get('peak_memory', 0)
        if peak_memory > 0:
            total_peak_memory = max(total_peak_memory, peak_memory / (1024 * 1024))  # Convert bytes to MB
    
    # Calculate total model memory (base + activation + kv_cache)
    total_model_memory = 0
    for component in ['DiT', 'T5', 'VAE']:
        if component in memory_data:
            mem = memory_data[component]
            base = mem.get('base', 0)
            activation = mem.get('activation', 0)
            peak = mem.get('peak', 0)
            kv = mem.get('kv_cache', 0)
            
            # Use activation if available, otherwise fall back to peak
            component_activation = activation if activation > 0 else peak
            total_model_memory += base + component_activation + kv
    
    # Pure runtime overhead = Total peak - Model components
    pure_runtime_overhead = total_peak_memory - total_model_memory
    
    # Store pure runtime memory in a special 'Runtime' component
    if pure_runtime_overhead > 0:
        memory_data['Runtime']['runtime'] = pure_runtime_overhead
    elif total_peak_memory > 0:  # Fallback to total peak if calculation results in negative
        memory_data['Runtime']['runtime'] = total_peak_memory
    
    # Calculate durations with more comprehensive time tracking
    if 't5_encode_start' in time_data and 't5_encode_end' in time_data:
        durations['T5 Encode'] = time_data['t5_encode_end'] - time_data['t5_encode_start']
    if 'dit_forward_start' in time_data and 'dit_forward_end' in time_data:
        durations['DiT Forward'] = time_data['dit_forward_end'] - time_data['dit_forward_start']
    if 'vae_decode_start' in time_data and 'vae_decode_end' in time_data:
        durations['VAE Decode'] = time_data['vae_decode_end'] - time_data['vae_decode_start']
    if 'generate_start' in time_data and 'generate_end' in time_data:
        durations['Total Generation'] = time_data['generate_end'] - time_data['generate_start']
    
    # Add model loading time - prefer 'Model Loaded' duration over calculated time
    if 'Model Loaded' in durations:
        # Use the duration from 'Model Loaded' event (this is the most accurate)
        durations['Model Loading'] = durations['Model Loaded']
        # Remove the 'Model Loaded' entry to avoid duplication
        del durations['Model Loaded']
    elif 'init' in time_data and 'generate_start' in time_data:
        # Fallback to calculated time if 'Model Loaded' duration is not available
        durations['Model Loading'] = time_data['generate_start'] - time_data['init']

    return {'memory': memory_data, 'time': durations}

def generate_report(analysis_results, output_file):
    """Generates a Markdown report from the analysis results, with improved format for model components and runtime."""
    with open(output_file, 'w') as f:
        f.write("# Memory and Performance Analysis Report\n\n")
        f.write("This report details the memory consumption and performance for different model components and configurations.\n\n")

        for config_name, results in sorted(analysis_results.items()):
            data = results['memory']
            durations = results['time']

            f.write(f"## Test Configuration: `{config_name}`\n\n")
            f.write("### Memory Analysis\n\n")
            
            # Model Components Section
            f.write("#### Model Components\n\n")
            f.write("| Component | Base Memory (MB) | Activation Memory (MB) | KV Cache (MB) |\n")
            f.write("|-----------|------------------|------------------------|---------------|\n")

            total_base = 0
            total_activation = 0
            total_kv = 0
            runtime_memory = 0

            for component in sorted(data.keys()):
                mem = data[component]
                base = mem.get('base', 0)
                peak = mem.get('peak', 0)
                activation = mem.get('activation', 0)  # Use dedicated activation field
                kv = mem.get('kv_cache', 0)
                runtime = mem.get('runtime', 0)
                
                # If no dedicated activation memory, fall back to peak
                if activation == 0 and peak > 0:
                    activation = peak

                # Skip Runtime component from model components table
                if component == 'Runtime':
                    runtime_memory = runtime
                    continue

                total_base += base
                total_activation += activation
                total_kv += kv

                base_str = f"{base:.2f}" if base > 0 else "-"
                activation_str = f"{activation:.2f}" if activation > 0 else "-"
                kv_str = f"{kv:.2f}" if kv > 0 else "-"

                f.write(f"| {component} | {base_str} | {activation_str} | {kv_str} |\n")

            f.write("|-----------|------------------|------------------------|---------------|\n")
            f.write(f"| **Total** | **{total_base:.2f}** | **{total_activation:.2f}** | **{total_kv:.2f}** |\n\n")
            
            # Runtime Memory Section
            if runtime_memory > 0:
                f.write("#### Runtime Memory\n\n")
                f.write("*Note: This represents pure PyTorch/CUDA runtime overhead, excluding model base memory, activation memory, and KV cache.*\n\n")
                f.write("| Component | Memory (MB) |\n")
                f.write("|-----------|-------------|\n")
                f.write(f"| PyTorch/CUDA Runtime Overhead | {runtime_memory:.2f} |\n\n")
            


            # Performance Analysis - always show this section
            f.write("### Performance Analysis\n\n")
            
            if durations:
                f.write("#### Timing Information\n\n")
                f.write("| Stage              | Duration (s) |\n")
                f.write("|--------------------|--------------|\n")
                
                # Define the desired order of stages
                stage_order = ["Model Loading", "T5 Encode", "DiT Forward", "VAE Decode", "Total Generation"]

                # Use a dictionary to store the data in the desired order
                ordered_stages = {stage: None for stage in stage_order}

                # Populate with actual data and collect others
                other_stages = {}
                for stage, duration in durations.items():
                    if stage in ordered_stages:
                        ordered_stages[stage] = duration
                    else:
                        other_stages[stage] = duration

                # Generate report lines in the specified order
                for stage in stage_order:
                    duration = ordered_stages.get(stage)
                    if duration is not None:
                        f.write(f"| {stage:<18} | {duration:.2f}       |\n")

                # Append any other stages that were not in the predefined order
                for stage, duration in sorted(other_stages.items()):
                    f.write(f"| {stage:<18} | {duration:.2f}       |\n")

                f.write("|--------------------|--------------|\n")
            else:
                f.write("#### Timing Information\n\n")
                f.write("No timing data available in the log file.\n\n")

    print(f"Report generated at {output_file}")

def find_latest_memory_events_file(base_dir="profiler_logs"):
    """Find the latest memory_events JSON file in the profiler_logs directory."""
    import glob
    
    # Look for memory_events files with timestamp pattern
    pattern = os.path.join(base_dir, "memory_events_*_*.json")
    files = glob.glob(pattern)
    
    if not files:
        # Fallback to the old baseline location
        baseline_file = os.path.join(base_dir, "baseline", "memory_events.json")
        if os.path.isfile(baseline_file):
            return baseline_file
        return None
    
    # Sort by modification time to get the latest file
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def main():
    parser = argparse.ArgumentParser(description='Analyze memory profiler logs and generate a report.')
    parser.add_argument('--log_file', type=str, help='Path to the memory events log file (JSONL format). If not provided, will auto-detect the latest file.')
    parser.add_argument('--output_file', type=str, default='enhanced_memory_analysis_report.md', help='Output file for the memory analysis report.')
    args = parser.parse_args()

    # Auto-detect log file if not provided
    if args.log_file:
        log_file = args.log_file
    else:
        log_file = find_latest_memory_events_file()
        if not log_file:
            print("Error: No memory events file found. Please specify --log_file or ensure memory_events files exist in profiler_logs directory.")
            return
        print(f"Auto-detected log file: {log_file}")

    if not os.path.isfile(log_file):
        print(f"Error: Log file '{log_file}' not found.")
        return

    analysis_results = analyze_log_file(log_file)
    if not analysis_results:
        print(f"No events could be parsed from '{log_file}'.")
        return
        
    generate_report(analysis_results, args.output_file)

if __name__ == '__main__':
    main()
