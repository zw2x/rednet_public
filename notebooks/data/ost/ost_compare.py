#!/usr/bin/env python3

import os
import argparse
import json
import sys

# Try to import ost modules. If not available, print a warning but continue (for syntax check).
try:
    import ost
    from ost import io, mol
    from ost.mol.alg import scoring
except ImportError:
    print("Warning: OST modules not found. This script requires OpenStructure (ost) to run.", file=sys.stderr)
    # We don't exit here to allow for basic syntax checking if run in an environment without OST.
    # In a real run without OST, this would crash later.


def compare_structures(model_path, reference_path, output_path, options):
    """
    Compares a model structure against a reference structure using OpenStructure.
    """
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    print(f"Loading model: {model_path}")
    model = io.LoadEntity(model_path)
    
    print(f"Loading reference: {reference_path}")
    reference = io.LoadEntity(reference_path)

    # Initialize Scorer
    # The Scorer class handles structure cleanup, chain mapping, etc.
    # We pass the model and reference, and any options.
    
    print("Initializing Scorer...")
    scorer = scoring.Scorer(model, reference)
    
    results = {}
    
    # Compute requested scores
    if options.lddt:
        print("Computing LDDT...")
        # Accessing the lddt attribute triggers computation
        results["lddt"] = scorer.lddt
        
    if options.local_lddt:
        print("Computing Local LDDT...")
        results["local_lddt"] = scorer.local_lddt
        
    if options.qs_score:
        print("Computing QS Score...")
        results["qs_global"] = scorer.qs_global
        results["qs_best"] = scorer.qs_best
        
    if options.tm_score:
        print("Computing TM Score...")
        results["tm_score"] = scorer.tm_score
        
    if options.dockq:
        print("Computing DockQ...")
        results["dockq"] = scorer.dockq

    # Add general info
    results["model_file"] = model_path
    results["reference_file"] = reference_path
    results["ost_version"] = ost.__version__ if hasattr(ost, "__version__") else "unknown"

    # Save results to JSON
    print(f"Saving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Compare two structures using OpenStructure (OST).")
    
    parser.add_argument("model", help="Path to the model structure file (pdb, cif, etc.)")
    parser.add_argument("reference", help="Path to the reference structure file (pdb, cif, etc.)")
    parser.add_argument("-o", "--output", default="comparison_results.json", help="Path to output JSON file (default: comparison_results.json)")
    
    # Score options
    parser.add_argument("--lddt", action="store_true", help="Compute Global LDDT score")
    parser.add_argument("--local-lddt", action="store_true", help="Compute Local (per-residue) LDDT scores")
    parser.add_argument("--qs-score", action="store_true", help="Compute QS-score (Quaternary Structure)")
    parser.add_argument("--tm-score", action="store_true", help="Compute TM-score (using USalign)")
    parser.add_argument("--dockq", action="store_true", help="Compute DockQ score")
    
    args = parser.parse_args()
    
    try:
        compare_structures(args.model, args.reference, args.output, args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
