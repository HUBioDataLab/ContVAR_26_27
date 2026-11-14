import pandas as pd
import argparse
import sys

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Map UniProt sequences to GOA entries and export to JSON.")
    parser.add_argument("--uniprot", required=True, help="Path to the UniProt TSV file (contains Entry and Sequence)")
    parser.add_argument("--goa", required=True, help="Path to the GOA TSV file (no header)")
    parser.add_argument("--output", default="uniprot_dict.json", help="Output JSON file name (default: uniprot_dict.json)")

    args = parser.parse_args()

    try:
        print(f"[*] Loading UniProt data from: {args.uniprot}")
        # df1: Main dataset with Sequences
        df1 = pd.read_csv(args.uniprot, delimiter='\t')

        print(f"[*] Loading GOA data from: {args.goa}")
        # df2: GOA dataset (no header)
        df2 = pd.read_csv(args.goa, delimiter='\t', header=None)
        
        # Assigning column names based on your structure
        df2.columns = ["DB", "Entry"] + [f'val_{i}' for i in range(len(df2.columns) - 2)]

        # 2. Pre-processing: Keep only unique Entries from GOA to avoid redundant JSON records
        print("[*] Filtering unique entries...")
        unique_entries = df2[['Entry']].drop_duplicates()

        # 3. Merge: Match Sequences from df1 based on Entry
        print("[*] Merging datasets...")
        merged_df = pd.merge(
            unique_entries, 
            df1[['Entry', 'Sequence']], 
            on='Entry', 
            how='inner'
        )

        # 4. Export to JSON
        print(f"[*] Generating JSON for {len(merged_df)} entries...")
        json_result = merged_df.to_json(orient='records', indent=4, force_ascii=False)

        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(json_result)

        print(f"[+] Success! File saved as: {args.output}")

    except Exception as e:
        print(f"[!] An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()