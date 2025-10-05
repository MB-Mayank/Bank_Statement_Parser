import pandas as pd
from pathlib import Path
from custom_parsers.icici_parser import parse  # replace with actual file name, e.g., icici_parser

# ============================================================
# 🧩 STARTING PARSER VALIDATION PROCESS
# ============================================================
print("\n" + "=" * 70)
print("🔍 WE WILL NOW PARSE THE PDF OUTPUT AND VERIFY WITH EXPECTED CSV")
print("=" * 70 + "\n")

# Path to your PDF
pdf_path = Path("data/icici/icici_sample.pdf")

# Parse the PDF
print("➡️ Parsing the PDF file...")
df = parse(pdf_path)

# Show first few rows
print("\n✅ Parsed Data Preview:")
print(df.head())

# Save parsed data to CSV
csv_output_path = Path("output_icici.csv")
df.to_csv(csv_output_path, index=False)

print(f"\n💾 Parsed CSV saved successfully at: {csv_output_path}\n")

# ============================================================
# NEXT STEP: VALIDATION WITH EXPECTED CSV
# ============================================================
print("=" * 70)
print("📊 Next, we will compare the parsed CSV with the expected ground truth.")
print("=" * 70 + "\n")
