import sys
import os
import argparse
import yaml

# Add the project root to the path so we can import from src
# Assumes this script is run from scripts/ or root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import TemporalDatasetLoader
from src.trainer import Experiment
from src.utils import set_seed

def main():
    parser = argparse.ArgumentParser(description="Run socio-economic analysis experiment.")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Load Config
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Error: Config file not found at {args.config}")
        sys.exit(1)

    # Set Seed
    set_seed(42)

    # Initialize Data Loader
    print("⏳ Loading data...")
    data_loader = TemporalDatasetLoader(config)

    # Initialize and Run Experiment
    print("🚀 Starting experiment...")
    experiment = Experiment(config, data_loader)
    experiment.run_experiments()
    print("✨ Experiment completed.")

if __name__ == "__main__":
    main()
