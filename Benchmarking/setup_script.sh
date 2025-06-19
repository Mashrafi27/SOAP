#!/bin/bash
# NYUAD HPC Zeo++ Parallel Job Array Setup
# This creates the command list for slurm_parallel_ja_submit.sh

# Parameters
INPUT_DIR="${1:-.}"
OUTPUT_DIR="${2:-zeopp_results}"
PROBE_RADIUS="${3:-1.9}"
NETWORK_CMD="network"

# Monte Carlo samples
SA_SAMPLES=500
VOL_SAMPLES=5000

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create commands file for parallel job array
COMMANDS_FILE="zeopp_commands.txt"
PROCESS_SCRIPT="process_single_cif.sh"

echo "Creating parallel job array setup for NYUAD HPC..."

# Count CIF files
count=$(find "$INPUT_DIR" -maxdepth 1 -name "*.cif" -type f | wc -l)
if [ "$count" -eq 0 ]; then
  echo "No CIF files in $INPUT_DIR"; exit 1
fi

echo "Found $count CIF files to process"

# Create the processing script for individual CIF files
cat > "$PROCESS_SCRIPT" << 'SCRIPT_EOF'
#!/bin/bash
# Individual CIF file processing script

CIF_FILE="$1"
OUTPUT_DIR="$2"
PROBE_RADIUS="$3"
SA_SAMPLES="$4"
VOL_SAMPLES="$5"
NETWORK_CMD="$6"

# Get basename
base=$(basename "$CIF_FILE" .cif)

# Create unique temporary directory for this job
tmpdir="zeopp_temp_${base}_$$"
mkdir -p "$tmpdir"

# Define output files
resf="$tmpdir/$base.res"
saf="$tmpdir/$base.sa"
volf="$tmpdir/$base.vol"
chf="$tmpdir/$base.chan"
result_file="$OUTPUT_DIR/${base}_result.csv"

# Initialize defaults
Di=Df=Dif=density=unitvol=SA_A2=SA_m2_cm3=SA_m2_g=AV_A3=AV_frac=AV_cm3_g=numchan=numpock="NA"

# 1) Pore diameters
$NETWORK_CMD -ha -res "$resf" "$CIF_FILE" >/dev/null 2>&1
if [ -s "$resf" ]; then
  read -r _ Di Df Dif < "$resf"
fi

# 2) Surface area
$NETWORK_CMD -ha -sa $PROBE_RADIUS $PROBE_RADIUS $SA_SAMPLES "$saf" "$CIF_FILE" >/dev/null 2>&1
if [ -s "$saf" ]; then
  unitvol=$(grep -oP 'Unitcell_volume:\s*\K[0-9.]+' "$saf" | head -1)
  density=$(grep -oP 'Density:\s*\K[0-9.]+' "$saf" | head -1)
  SA_A2=$(grep -oP '(?<!N)ASA_A\^2:\s*\K[0-9.]+' "$saf" | head -1)
  SA_m2_cm3=$(grep -oP '(?<!N)ASA_m\^2/cm\^3:\s*\K[0-9.]+' "$saf" | head -1)
  SA_m2_g=$(grep -oP '(?<!N)ASA_m\^2/g:\s*\K[0-9.]+' "$saf" | head -1)
  numpock=$(grep -oP 'Number_of_pockets:\s*\K[0-9]+' "$saf" | head -1)
  # Set fallbacks
  [ -z "$unitvol" ] && unitvol="NA"
  [ -z "$density" ] && density="NA"
  [ -z "$SA_A2" ] && SA_A2="NA"
  [ -z "$SA_m2_cm3" ] && SA_m2_cm3="NA"
  [ -z "$SA_m2_g" ] && SA_m2_g="NA"
  [ -z "$numpock" ] && numpock="NA"
fi

# 3) Accessible volume
$NETWORK_CMD -ha -vol $PROBE_RADIUS $PROBE_RADIUS $VOL_SAMPLES "$volf" "$CIF_FILE" >/dev/null 2>&1
if [ -s "$volf" ]; then
  AV_A3=$(grep -oP '(?<!N)AV_A\^3:\s*\K[0-9.]+' "$volf" | head -1)
  AV_frac=$(grep -oP '(?<!N)AV_Volume_fraction:\s*\K[0-9.]+' "$volf" | head -1)
  AV_cm3_g=$(grep -oP '(?<!N)AV_cm\^3/g:\s*\K[0-9.]+' "$volf" | head -1)
  [ -z "$AV_A3" ] && AV_A3="NA"
  [ -z "$AV_frac" ] && AV_frac="NA"
  [ -z "$AV_cm3_g" ] && AV_cm3_g="NA"
fi

# 4) Channels
$NETWORK_CMD -ha -chan $PROBE_RADIUS "$CIF_FILE" > "$chf" 2>&1
if [ -s "$chf" ]; then
  numchan=$(grep -oP 'Identified\s*\K[0-9]+(?=\s*channels)' "$chf" | head -1)
  [ -z "$numchan" ] && numchan=$(grep -oP 'Number_of_channels:\s*\K[0-9]+' "$chf" | head -1)
  [ -z "$numchan" ] && numchan="NA"
fi

# Write individual result file
echo "$base,$Di,$Df,$Dif,$density,$unitvol,$SA_A2,$SA_m2_cm3,$SA_m2_g,$AV_A3,$AV_frac,$AV_cm3_g,$numchan,$numpock" > "$result_file"

# Clean up temporary files
rm -rf "$tmpdir"

echo "Completed: $base"
SCRIPT_EOF

# Make the processing script executable
chmod +x "$PROCESS_SCRIPT"

# Create the commands file
echo "Creating commands file: $COMMANDS_FILE"
> "$COMMANDS_FILE"  # Clear the file

# Add a command for each CIF file
find "$INPUT_DIR" -maxdepth 1 -name "*.cif" -type f | while read -r cif_file; do
  echo "./process_single_cif.sh \"$cif_file\" \"$OUTPUT_DIR\" $PROBE_RADIUS $SA_SAMPLES $VOL_SAMPLES $NETWORK_CMD" >> "$COMMANDS_FILE"
done

echo "Created $COMMANDS_FILE with $count commands"

# Instructions for submission
cat << EOF

=====================================
HPC PARALLEL JOB ARRAY SETUP COMPLETE
=====================================

Files created:
- $COMMANDS_FILE: Contains $count commands for parallel processing
- $PROCESS_SCRIPT: Individual CIF processing script

TO SUBMIT THE JOB:
==================

1. Submit the parallel job array:
   slurm_parallel_ja_submit.sh -N 20 -t 08:00:00 $COMMANDS_FILE

   Options:
   -N 20     : Use 20 nodes (groups) for parallel processing
   -t 08:00:00 : 8 hour time limit per node group
   
   Adjust -N based on your needs:
   - More nodes = faster completion but more resources
   - Each node group will process ~$(($count/20)) files in parallel

2. After job completion, aggregate results:
   ./aggregate_results.sh $OUTPUT_DIR zeopp_final_results.csv

MONITORING:
===========
- Check job status: squeue -u \$USER
- Check progress: ls $OUTPUT_DIR/*_result.csv | wc -l
- View logs: Check the .out files created by the parallel job array

EOF