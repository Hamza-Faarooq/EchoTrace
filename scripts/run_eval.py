"""Run retrieval evaluation (implementation planned)."""
import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml", help="Evaluation config path")
    parser.parse_args()
    raise SystemExit("Evaluation is not implemented yet; see README.md.")

if __name__ == "__main__":
    main()
