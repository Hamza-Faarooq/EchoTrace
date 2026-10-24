"""Build the local audio embedding index (implementation planned)."""
import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True, help="Directory containing audio clips")
    parser.add_argument("--cache-dir", default="data/index", help="Output directory for vectors and manifest")
    parser.parse_args()
    raise SystemExit("Index building is not implemented yet; see README.md.")

if __name__ == "__main__":
    main()
