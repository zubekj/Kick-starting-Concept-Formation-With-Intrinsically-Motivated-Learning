#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from params import Parameters
from SMMain import Main, build_episode_dataset
from storage import StorageManager


_ = Main


def render_episode(
    main_dump_path,
    episode_index,
    render_params=None,
    output_dir=None,
    seed_offset=0,
):
    """
    Render a single episode based on the unique episode index.

    Args:
        main_dump_path: Path to the main.dump.npy file
        episode_index: Index (0-17) specifying the episode type
        output_dir: Directory to save the rendered episode
        seed_offset: Additional offset to add to the seed for variation
    """
    if not Path(main_dump_path).is_file():
        raise FileNotFoundError(f"Could not find main dump file: {main_dump_path}")

    main = np.load(main_dump_path, allow_pickle=True)[0]
    orig_dir = (Path.cwd() / ".." / ".." / ".." / "..").resolve()
    exp_dir = (Path.cwd() / ".." / "..").resolve()
    main.sm = StorageManager(exp_dir.stem, orig_dir)

    # Validate index
    if episode_index < 0 or episode_index >= len(main.episode_dataset):
        raise ValueError(
            f"Episode index must be between 0 and "
            f"{len(main.episode_dataset) - 1}, got {episode_index}"
        )

    # Get episode configuration
    config = main.get_episode_config(episode_index)

    print(f"Rendering episode with index {episode_index}:")
    print(f"  - Object param index: {config['obj_param_index']}")
    print(f"  - Context: {config['context']}")
    print(f"  - Stretch: {config['stretch']}")
    print(f"  - Rotation: {config['rotation']}")

    # Setup output directory
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = str(output_dir / f"episode_{episode_index}")

    # Run and render the episode using Main's method
    main.run_single_episode(
        episode_index=episode_index,
        render="offline",
        output_path=output_path,
        seed_offset=seed_offset,
        zero_noise=True,
        render_params=render_params,
    )

    print(f"Episode rendered and saved to: {output_path}.gif")

    return main.episode_dataset


def print_episode_dataset(params):
    """Print the full episode dataset for reference."""
    dataset, _ = build_episode_dataset(params)
    print("\nEpisode Dataset (18 unique episode types):")
    print("=" * 60)
    print(dataset.to_string(index=False))
    print("=" * 60)
    return dataset


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Render a single evaluation episode by index (0-17)"
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=0,
        help="Episode index (0-17) specifying the episode type",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=200,
        help="Duration of the animation (secs)",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=float,
        default=1,
        help="The resolution of the animation (proportion to (3in, 3in))",
    )
    parser.add_argument(
        "-m",
        "--main_dump",
        help="Path to main.dump.npy file",
        default="main.dump.npy",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output directory for rendered episode",
        default="rendered_episodes",
    )
    parser.add_argument(
        "-s",
        "--seed_offset",
        type=int,
        help="Additional seed offset for variation",
        default=0,
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List all episode types and exit",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        action="store_true",
        help="Use GPU if available",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Setup device
    device = "cuda" if torch.cuda.is_available() and args.gpu else "cpu"
    torch.set_default_device(device)

    # If list flag is set, just print the dataset and exit
    if args.list:
        params = Parameters()
        if Path("params.json").is_file():
            params.load("params.json", mode="json")
        print_episode_dataset(params)
        sys.exit(0)

    # Validate index range
    if args.index < 0 or args.index > 17:
        print(f"Error: Episode index must be between 0 and 17, " f"got {args.index}")
        sys.exit(1)

    # Render the episode
    try:
        dataset = render_episode(
            main_dump_path=args.main_dump,
            episode_index=args.index,
            output_dir=args.output_dir,
            seed_offset=args.seed_offset,
            render_params={"resolution_prop": args.resolution, "duration": args.duration}
        )

        print("\nFull episode dataset for reference:")
        print(dataset.to_string(index=False))

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error rendering episode: {e}")
        raise
