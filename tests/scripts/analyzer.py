import json
import os
import argparse
from collections import defaultdict

def analyze_log_file(log_file):
    """Analyzes a single memory events file in JSONL format."""
    analysis_results = {}
    config_name = os.path.splitext(os.path.basename(log_file))[0]
    with open(log_file, 'r') as f:
        # The new format is one JSON object per line (JSONL)
        events = [json.loads(line) for line in f if line.strip()]
    
    if not events:
        return {}

    analysis_results[config_name] = parse_events(events)
    return analysis_results

def parse_events(events):
    """Parses events from a single log file to extract memory and time metrics."""
    memory_data = defaultdict(lambda: {'base': 0, 'peak': 0, 'runtime': 0, 'kv_cache': 0})
    time_data = {}
    
    for event in events:
        event_name = event.get('event', '')
        model_name = event.get('model_name')
        
        if 'timestamp' in event:
            time_data[event_name] = event['timestamp']
            continue


        if not model_name:
            continue

        if 'load' in event_name:
            memory_data[model_name]['base'] = event.get('pytorch_tensor_peak_mb', 0)
        
        elif 'forward' in event_name:
            memory_data[model_name]['peak'] = max(
                memory_data[model_name]['peak'], 
                event.get('pytorch_tensor_peak_mb', 0)
            )
            memory_data[model_name]['runtime'] = max(
                memory_data[model_name]['runtime'],
                event.get('cuda_runtime_peak_mb', 0)
            )

        elif 'kv_cache' in event_name:
            memory_data[model_name]['kv_cache'] = event.get('incremental_memory_mb', 0)

    # Calculate durations
    durations = {}
    if 't5_encode_start' in time_data and 't5_encode_end' in time_data:
        durations['T5 Encode'] = time_data['t5_encode_end'] - time_data['t5_encode_start']
    if 'dit_forward_start' in time_data and 'dit_forward_end' in time_data:
        durations['DiT Forward'] = time_data['dit_forward_end'] - time_data['dit_forward_start']
    if 'vae_decode_start' in time_data and 'vae_decode_end' in time_data:
        durations['VAE Decode'] = time_data['vae_decode_end'] - time_data['vae_decode_start']
    if 'generate_start' in time_data and 'generate_end' in time_data:
        durations['Total Generation'] = time_data['generate_end'] - time_data['generate_start']

    return {'memory': memory_data, 'time': durations}

def generate_report(analysis_results, output_file):
    """Generates a Markdown report from the analysis results, separating activation and KV cache memory."""
    with open(output_file, 'w') as f:
        f.write("# Memory and Performance Analysis Report\n\n")
        f.write("This report details the memory consumption and performance for different model components and configurations.\n\n")

        for config_name, results in sorted(analysis_results.items()):
            data = results['memory']
            durations = results['time']

            f.write(f"## Test Configuration: `{config_name}`\n\n")
            f.write("### Memory Analysis\n\n")
            f.write("| Component      | Model Load (MB) | Activation (MB) | KV Cache (MB) | PyTorch/CUDA Runtime (MB) |\n")
            f.write("|----------------|-----------------|-----------------|---------------|---------------------------|\n")

            total_base = 0
            total_activation = 0
            total_kv = 0
            total_runtime = 0

            for component in sorted(data.keys()):
                mem = data[component]
                base = mem.get('base', 0)
                peak = mem.get('peak', 0)
                kv = mem.get('kv_cache', 0)
                runtime = mem.get('runtime', 0)
                
                activation = max(0, peak - base - kv)

                total_base += base
                total_activation += activation
                total_kv += kv
                total_runtime += runtime

                base_str = f"{base:.2f}" if base > 0 else "-"
                activation_str = f"{activation:.2f}" if peak > 0 else "-"
                kv_str = f"{kv:.2f}" if kv > 0 else "-"
                runtime_str = f"{runtime:.2f}" if runtime > 0 else "-"

                f.write(f"| {component:<14} | {base_str:<15} | {activation_str:<15} | {kv_str:<13} | {runtime_str:<25} |\n")

            f.write("|----------------|-----------------|-----------------|---------------|---------------------------|\n")
            f.write(f"| **Total**      | **{total_base:.2f}**    | **{total_activation:.2f}**      | **{total_kv:.2f}**    | **{total_runtime:.2f}**           |\n\n")

            if durations:
                f.write("### Performance Analysis\n\n")
                f.write("| Stage              | Duration (s) |\n")
                f.write("|--------------------|--------------|\n")
                for stage, duration in sorted(durations.items()):
                    f.write(f"| {stage:<18} | {duration:.2f}       |\n")
                f.write("|--------------------|--------------|\n\n")

    print(f"Report generated at {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze memory profiler logs and generate a report.')
    parser.add_argument('--log_file', type=str, required=True, help='Path to the memory events log file (JSONL format).')
    parser.add_argument('--output_file', type=str, default='enhanced_memory_analysis_report.md', help='Output file for the memory analysis report.')
    args = parser.parse_args()

    if not os.path.isfile(args.log_file):
        print(f"Error: Log file '{args.log_file}' not found.")
        return

    analysis_results = analyze_log_file(args.log_file)
    if not analysis_results:
        print(f"No events could be parsed from '{args.log_file}'.")
        return
        
    generate_report(analysis_results, args.output_file)

if __name__ == '__main__':
    main()