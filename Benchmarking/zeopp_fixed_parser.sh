#!/bin/bash
#SBATCH --job-name=zeopp_fixed
#SBATCH --output=zeopp_%j.out
#SBATCH --error=zeopp_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mashrafi@nyu.edu

# Fixed Zeo++ batch analysis script with proper parsing

# Load required modules
# module purge
# module load zeopp-lsmo

# Parameters
INPUT_DIR="${1:-.}"
OUTPUT_CSV="${2:-zeopp_results_fixed.csv}"
PROBE_RADIUS="${3:-1.9}"
NETWORK_CMD="network"

# Create temporary directory
TEMP_DIR="zeopp_temp_$$"
mkdir -p "$TEMP_DIR"

# Function to safely extract numeric value
extract_number() {
    echo "$1" | grep -oE '[0-9]+\.?[0-9]*' | head -1
}

# Count CIF files
CIF_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -name "*.cif" -type f | wc -l)
echo "Found $CIF_COUNT CIF files to process"
echo "Using probe radius: $PROBE_RADIUS Å"

# Create CSV header
echo "filename,Di_A,Df_A,Dif_A,density_gcc,unit_cell_volume_A3,SA_m2_per_g,SA_m2_per_cm3,SA_A2_per_cell,AV_fraction,AV_A3_per_cell,channel_dim" > "$OUTPUT_CSV"

# Process each CIF file
COUNTER=0
for CIF_FILE in "$INPUT_DIR"/*.cif; do
    if [ -f "$CIF_FILE" ]; then
        COUNTER=$((COUNTER + 1))
        BASENAME=$(basename "$CIF_FILE" .cif)
        echo "Processing [$COUNTER/$CIF_COUNT]: $BASENAME.cif"
        
        # Initialize all variables
        Di="NA"; Df="NA"; Dif="NA"
        DENSITY="NA"; UNITCELL_VOL="NA"
        SA_M2_G="NA"; SA_M2_CM3="NA"; SA_A2_CELL="NA"
        AV_FRAC="NA"; AV_A3="NA"
        CHAN_DIM="NA"
        
        # Define output files
        RES_FILE="$TEMP_DIR/${BASENAME}.res"
        SA_OUT="$TEMP_DIR/${BASENAME}_sa.out"
        VOL_OUT="$TEMP_DIR/${BASENAME}_vol.out"
        CHAN_OUT="$TEMP_DIR/${BASENAME}_chan.out"
        
        # 1. Get pore diameters
        echo "  - Calculating pore diameters..."
        $NETWORK_CMD -ha -res "$RES_FILE" "$CIF_FILE" > "$TEMP_DIR/res.log" 2>&1
        
        if [ -f "$RES_FILE" ]; then
            # RES format: filename Di Df Dif
            read -r _ Di Df Dif <<< $(tail -1 "$RES_FILE" | tr -s ' ')
        fi
        
        # 2. Get density and unit cell volume from the log
        if [ -f "$TEMP_DIR/res.log" ]; then
            # Look for density in g/cm^3
            DENSITY_LINE=$(grep -i "density" "$TEMP_DIR/res.log" | grep -E "g/cm|gcc")
            if [ -n "$DENSITY_LINE" ]; then
                DENSITY=$(extract_number "$DENSITY_LINE")
            fi
            
            # Look for unit cell volume
            VOL_LINE=$(grep -i "unit.*cell.*volume\|cell.*volume" "$TEMP_DIR/res.log")
            if [ -n "$VOL_LINE" ]; then
                UNITCELL_VOL=$(extract_number "$VOL_LINE")
            fi
        fi
        
        # 3. Calculate surface area
        echo "  - Calculating surface area..."
        # Capture both stdout and file output
        $NETWORK_CMD -ha -sa $PROBE_RADIUS $PROBE_RADIUS 1000 "$TEMP_DIR/temp.sa" "$CIF_FILE" > "$SA_OUT" 2>&1
        
        # Check if output went to file
        if [ -f "$TEMP_DIR/temp.sa" ] && [ -s "$TEMP_DIR/temp.sa" ]; then
            cp "$TEMP_DIR/temp.sa" "$SA_OUT"
        fi
        
        # Parse surface area - try multiple formats
        if [ -f "$SA_OUT" ]; then
            # Format 1: @ ASA_A^2 ASA_m^2/cm^3 ASA_m^2/g
            SA_LINE=$(grep "^@" "$SA_OUT" | grep -i "ASA" | head -1)
            if [ -n "$SA_LINE" ]; then
                # Extract numbers after @ symbol
                SA_NUMS=($(echo "$SA_LINE" | sed 's/@//' | grep -oE '[0-9]+\.?[0-9]*'))
                if [ ${#SA_NUMS[@]} -ge 3 ]; then
                    SA_A2_CELL=${SA_NUMS[0]}
                    SA_M2_CM3=${SA_NUMS[1]}
                    SA_M2_G=${SA_NUMS[2]}
                fi
            fi
            
            # Format 2: Look for labeled values
            if [ "$SA_M2_G" = "NA" ]; then
                SA_M2_G_LINE=$(grep -E "ASA.*m\^2/g|m2/g" "$SA_OUT" | tail -1)
                SA_M2_G=$(extract_number "$SA_M2_G_LINE")
                
                SA_M2_CM3_LINE=$(grep -E "ASA.*m\^2/cm\^3|m2/cm3" "$SA_OUT" | tail -1)
                SA_M2_CM3=$(extract_number "$SA_M2_CM3_LINE")
                
                SA_A2_CELL_LINE=$(grep -E "ASA.*A\^2|ASA.*Angstrom" "$SA_OUT" | tail -1)
                SA_A2_CELL=$(extract_number "$SA_A2_CELL_LINE")
            fi
        fi
        
        # 4. Calculate accessible volume
        echo "  - Calculating accessible volume..."
        $NETWORK_CMD -ha -vol $PROBE_RADIUS $PROBE_RADIUS 10000 "$TEMP_DIR/temp.vol" "$CIF_FILE" > "$VOL_OUT" 2>&1
        
        if [ -f "$TEMP_DIR/temp.vol" ] && [ -s "$TEMP_DIR/temp.vol" ]; then
            cp "$TEMP_DIR/temp.vol" "$VOL_OUT"
        fi
        
        # Parse volume
        if [ -f "$VOL_OUT" ]; then
            # Look for AV (accessible volume) line
            AV_LINE=$(grep -E "^@.*AV|Accessible.*volume" "$VOL_OUT" | head -1)
            if [ -n "$AV_LINE" ]; then
                AV_NUMS=($(echo "$AV_LINE" | grep -oE '[0-9]+\.?[0-9]*'))
                if [ ${#AV_NUMS[@]} -ge 2 ]; then
                    AV_A3=${AV_NUMS[0]}
                    AV_FRAC=${AV_NUMS[1]}
                fi
            fi
        fi
        
        # 5. Analyze channels
        echo "  - Analyzing channel dimensionality..."
        $NETWORK_CMD -ha -chan $PROBE_RADIUS "$CIF_FILE" > "$CHAN_OUT" 2>&1
        
        if [ -f "$CHAN_OUT" ]; then
            # Look for dimensionality
            CHAN_DIM=$(grep -oE "[0-3][ -]?D" "$CHAN_OUT" | head -1 | grep -oE "[0-3]")
            if [ -z "$CHAN_DIM" ]; then
                # Alternative: look for "dimensionality" keyword
                DIM_LINE=$(grep -i "dimension" "$CHAN_OUT" | head -1)
                CHAN_DIM=$(echo "$DIM_LINE" | grep -oE "[0-3]" | head -1)
            fi
        fi
        
        # Write results to CSV
        echo "$BASENAME,$Di,$Df,$Dif,$DENSITY,$UNITCELL_VOL,$SA_M2_G,$SA_M2_CM3,$SA_A2_CELL,$AV_FRAC,$AV_A3,$CHAN_DIM" >> "$OUTPUT_CSV"
        
        # Clean up file-specific temps
        rm -f "$TEMP_DIR"/*.sa "$TEMP_DIR"/*.vol "$TEMP_DIR"/*.res "$TEMP_DIR"/*.out 2>/dev/null
        
        echo "  - Done"
    fi
done

# Final cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "Analysis complete! Results saved to: $OUTPUT_CSV"
echo "Processed $COUNTER CIF files"

# Show sample of results
echo ""
echo "Sample results (first 5 lines):"
head -5 "$OUTPUT_CSV" | column -t -s,