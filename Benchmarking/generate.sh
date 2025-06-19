#!/bin/bash
#SBATCH --job-name=zeopp_batch
#SBATCH --output=zeopp_%j.out
#SBATCH --error=zeopp_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mashrafi@nyu.edu

# Batch Zeo++ analysis script for MOF CIF files
# This script processes all CIF files in a directory and outputs results to a CSV file

# Usage: sbatch zeopp_batch_analysis.sh [input_directory] [output_csv] [probe_radius]
# Example: sbatch zeopp_batch_analysis.sh ./cif_files results.csv 1.82

# Load required modules (adjust based on your HPC setup)
module purge
module load zeopp-lsmo  # or module load conda and activate environment

# Default parameters
INPUT_DIR="${1:-/path/to/your/cif/files}"  # Specify your CIF directory here
OUTPUT_CSV="${2:-zeopp_results.csv}"
PROBE_RADIUS="${3:-1.82}"  # Default CO2 probe radius
NETWORK_CMD="network"  # Change this to full path if needed

# Number of Monte Carlo samples
SA_SAMPLES=10000
VOL_SAMPLES=100000
PSD_SAMPLES=100000

# Create temporary directory for intermediate files
TEMP_DIR="zeopp_temp_$$"
mkdir -p "$TEMP_DIR"

# Function to extract value from Zeo++ output files
extract_value() {
    local file=$1
    local pattern=$2
    local field=$3
    if [ -f "$file" ]; then
        grep "$pattern" "$file" | awk "{print \$$field}" | head -1
    else
        echo "NA"
    fi
}

# Function to extract channel dimensionality
extract_channel_dim() {
    local file=$1
    if [ -f "$file" ]; then
        # Look for dimensionality in the output
        dim=$(grep -E "dimensionality|channel system is [0-9]D" "$file" | grep -oE "[0-9]D" | head -1 | tr -d 'D')
        if [ -z "$dim" ]; then
            echo "NA"
        else
            echo "$dim"
        fi
    else
        echo "NA"
    fi
}

# Check if network command exists
if ! command -v $NETWORK_CMD &> /dev/null; then
    echo "Error: 'network' command not found. Please ensure zeopp-lsmo is installed and in PATH"
    echo "Or modify NETWORK_CMD variable in this script to point to the full path"
    exit 1
fi

# Count CIF files
CIF_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -name "*.cif" -type f | wc -l)
if [ "$CIF_COUNT" -eq 0 ]; then
    echo "No CIF files found in $INPUT_DIR"
    exit 1
fi

echo "Found $CIF_COUNT CIF files to process"
echo "Using probe radius: $PROBE_RADIUS Å"
echo "Results will be saved to: $OUTPUT_CSV"
echo ""

# Create CSV header
echo "filename,Di_A,Df_A,Dif_A,SA_m2_per_g,SA_m2_per_cm3,SA_A2_per_cell,AV_fraction,AV_A3_per_cell,POV_fraction,POV_A3_per_cell,channel_dim,psd_mean_A,psd_std_A" > "$OUTPUT_CSV"

# Process each CIF file
COUNTER=0
for CIF_FILE in "$INPUT_DIR"/*.cif; do
    if [ -f "$CIF_FILE" ]; then
        COUNTER=$((COUNTER + 1))
        BASENAME=$(basename "$CIF_FILE" .cif)
        echo "Processing [$COUNTER/$CIF_COUNT]: $BASENAME.cif"
        
        # Define output filenames
        RES_FILE="$TEMP_DIR/${BASENAME}.res"
        SA_FILE="$TEMP_DIR/${BASENAME}.sa"
        VOL_FILE="$TEMP_DIR/${BASENAME}.vol"
        VOLPO_FILE="$TEMP_DIR/${BASENAME}.volpo"
        PSD_FILE="$TEMP_DIR/${BASENAME}.psd"
        CHAN_FILE="$TEMP_DIR/${BASENAME}.chan"
        
        # Run Zeo++ analyses
        echo "  - Calculating pore diameters..."
        $NETWORK_CMD -ha -res "$RES_FILE" "$CIF_FILE" 2>/dev/null
        
        echo "  - Calculating surface area..."
        $NETWORK_CMD -ha -sa $PROBE_RADIUS $PROBE_RADIUS $SA_SAMPLES "$SA_FILE" "$CIF_FILE" 2>/dev/null
        
        echo "  - Calculating accessible volume..."
        $NETWORK_CMD -ha -vol $PROBE_RADIUS $PROBE_RADIUS $VOL_SAMPLES "$VOL_FILE" "$CIF_FILE" 2>/dev/null
        
        echo "  - Calculating probe-occupiable volume..."
        $NETWORK_CMD -ha -volpo $PROBE_RADIUS $PROBE_RADIUS $VOL_SAMPLES "$VOLPO_FILE" "$CIF_FILE" 2>/dev/null
        
        echo "  - Calculating pore size distribution..."
        $NETWORK_CMD -ha -psd $PROBE_RADIUS $PROBE_RADIUS $PSD_SAMPLES "$PSD_FILE" "$CIF_FILE" 2>/dev/null
        
        echo "  - Analyzing channel dimensionality..."
        $NETWORK_CMD -ha -chan $PROBE_RADIUS "$CIF_FILE" > "$CHAN_FILE" 2>/dev/null
        
        # Extract values from output files
        # Pore diameters from .res file
        if [ -f "$RES_FILE" ]; then
            # The .res file format: filename Di Df Dif
            read -r _ Di Df Dif < "$RES_FILE"
        else
            Di="NA"; Df="NA"; Dif="NA"
        fi
        
        # Surface area from .sa file
        if [ -f "$SA_FILE" ]; then
            # Extract values after @ symbol
            SA_LINE=$(grep "^@" "$SA_FILE" | head -1)
            SA_M2_G=$(echo "$SA_LINE" | awk '{print $5}')
            SA_M2_CM3=$(echo "$SA_LINE" | awk '{print $4}')
            SA_A2_CELL=$(echo "$SA_LINE" | awk '{print $3}')
        else
            SA_M2_G="NA"; SA_M2_CM3="NA"; SA_A2_CELL="NA"
        fi
        
        # Accessible volume from .vol file
        if [ -f "$VOL_FILE" ]; then
            # Extract accessible volume fraction and A^3
            AV_LINE=$(grep "^@" "$VOL_FILE" | head -1)
            AV_FRAC=$(echo "$AV_LINE" | awk '{print $4}')
            AV_A3=$(echo "$AV_LINE" | awk '{print $3}')
        else
            AV_FRAC="NA"; AV_A3="NA"
        fi
        
        # Probe-occupiable volume from .volpo file
        if [ -f "$VOLPO_FILE" ]; then
            # Extract POAV values
            POV_LINE=$(grep "^@" "$VOLPO_FILE" | grep "POAV" | head -1)
            if [ -n "$POV_LINE" ]; then
                POV_FRAC=$(echo "$POV_LINE" | awk '{print $5}')
                POV_A3=$(echo "$POV_LINE" | awk '{print $4}')
            else
                POV_FRAC="NA"; POV_A3="NA"
            fi
        else
            POV_FRAC="NA"; POV_A3="NA"
        fi
        
        # Channel dimensionality
        CHAN_DIM=$(extract_channel_dim "$CHAN_FILE")
        
        # PSD statistics (calculate mean and std from histogram)
        if [ -f "$PSD_FILE" ]; then
            # Simple mean calculation from PSD histogram
            # Format: bin_center count
            PSD_MEAN=$(awk 'NR>1 {sum += $1 * $2; count += $2} END {if(count>0) print sum/count; else print "NA"}' "$PSD_FILE")
            # For std dev, we'd need more complex calculation
            PSD_STD="NA"  # Placeholder
        else
            PSD_MEAN="NA"; PSD_STD="NA"
        fi
        
        # Append results to CSV
        echo "$BASENAME,$Di,$Df,$Dif,$SA_M2_G,$SA_M2_CM3,$SA_A2_CELL,$AV_FRAC,$AV_A3,$POV_FRAC,$POV_A3,$CHAN_DIM,$PSD_MEAN,$PSD_STD" >> "$OUTPUT_CSV"
        
        echo "  - Done"
        echo ""
    fi
done

# Cleanup temporary directory
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "Analysis complete! Results saved to: $OUTPUT_CSV"
echo ""
echo "Summary:"
echo "- Processed $COUNTER CIF files"
echo "- Probe radius used: $PROBE_RADIUS Å"
echo "- Output columns: filename, Di, Df, Dif, surface areas, volumes, channel dimensionality, PSD stats"