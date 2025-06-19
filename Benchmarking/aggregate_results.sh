#!/bin/bash
OUTPUT_DIR="${1:-zeopp_results}"
FINAL_CSV="${2:-zeopp_final_results.csv}"

echo "Aggregating results from $OUTPUT_DIR into $FINAL_CSV"

# Create header
echo "filename,Di_A,Df_A,Dif_A,density_gcc,unit_cell_volume_A3,SA_A2,SA_m2_cm3,SA_m2_g,AV_A3,AV_fraction,AV_cm3_g,num_channels,num_pockets" > "$FINAL_CSV"

# Concatenate all individual result files
cat "$OUTPUT_DIR"/*_result.csv >> "$FINAL_CSV"

echo "Final results written to: $FINAL_CSV"
echo "Total entries: $(wc -l < "$FINAL_CSV")"