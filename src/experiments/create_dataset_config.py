"""Utilities for generating per-dataset experiment configs."""

import argparse
import yaml
import os


def create_dataset_config(dataset_name: str, base_config_path: str, output_path: str = None):
    """Create a temporary config pointing to a single dataset."""
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    dataset_config = None
    for ds in config['experiment']['datasets']:
        if ds['name'] == dataset_name:
            dataset_config = ds
            break

    if dataset_config is None:
        raise ValueError(f"Dataset {dataset_name} not found in config")

    # Specialize experiment metadata for the requested dataset
    config['experiment']['name'] = f"semrocl_ablation_{dataset_name}"
    config['experiment']['dataset'] = dataset_name
    config['experiment']['dataset_path'] = dataset_config['path']
    config['experiment']['low_dir'] = dataset_config['low_dir']
    config['experiment']['high_dir'] = dataset_config['high_dir']
    config['experiment']['has_reference'] = dataset_config['has_reference']

    config['output']['base_dir'] = f"./outputs/ablation_multi_dataset/{dataset_name}"

    if output_path is None:
        output_path = f"../../configs/temp_{dataset_name}.yaml"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"Created config for {dataset_name}: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--base-config', required=True, help='Base config path')
    parser.add_argument('--output', default=None, help='Output config path')

    args = parser.parse_args()

    create_dataset_config(args.dataset, args.base_config, args.output)
