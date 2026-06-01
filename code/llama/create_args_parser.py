import argparse

MODEL_HF_PATHS = {
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}

def args_parser():
    parser = argparse.ArgumentParser(description='LLM4Mat-Bench Llama Inference')

    parser.add_argument('--model_name',
                        help='Model identifier. Options: ' + ', '.join(MODEL_HF_PATHS.keys()),
                        type=str,
                        default="llama2-7b")
    parser.add_argument('--dataset_name',
                        help='Dataset name, e.g. "mp", "snumat", "jarvis"',
                        type=str,
                        default="mp")
    parser.add_argument('--property_name',
                        help='Property to predict, e.g. "band_gap", "volume", "is_gap_direct"',
                        type=str,
                        default="band_gap")
    parser.add_argument('--input_type',
                        help='Input representation: "formula", "cif_structure", or "description"',
                        type=str,
                        default="formula")
    parser.add_argument('--prompt_type',
                        help='Prompting strategy: "zero_shot" or "few_shot"',
                        type=str,
                        default="zero_shot")
    parser.add_argument('--max_len',
                        help='Max token length for generation',
                        type=int,
                        default=800)
    parser.add_argument('--batch_size',
                        help='Inference batch size',
                        type=int,
                        default=8)
    parser.add_argument('--min_samples',
                        help='Minimum valid predictions required to report a metric (else "Invalid")',
                        type=int,
                        default=10)
    parser.add_argument('--data_path',
                        help='Path to dataset directory',
                        type=str,
                        default="data/")
    parser.add_argument('--results_path',
                        help='Path to save results',
                        type=str,
                        default="results/")

    args = parser.parse_args()
    return args
