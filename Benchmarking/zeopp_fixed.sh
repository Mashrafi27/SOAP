#!/bin/bash
#SBATCH --job-name=zeopp_batch
#SBATCH --output=zeopp_%j.out
#SBATCH --error=zeopp_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mashrafi@nyu.edu

# Fixed Zeo++ batch analysis script
# Version 5 - Fixed regex patterns to avoid matching NAV/NASA values

# Parameters
INPUT_DIR="${1:-.}"
OUTPUT_CSV="${2:-zeopp_results.csv}"
PROBE_RADIUS="${3:-1.9}"
NETWORK_CMD="network"

# Monte Carlo samples (reduced for speed)
SA_SAMPLES=500
VOL_SAMPLES=5000

# Parallel processing - detect available cores
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  DEFAULT_PROCS=$SLURM_CPUS_PER_TASK
elif command -v nproc >/dev/null 2>&1; then
  DEFAULT_PROCS=$(nproc)
else
  # Fallback methods for systems without nproc
  DEFAULT_PROCS=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || echo "4")
fi

NPROCS="${4:-$DEFAULT_PROCS}"

# Create temporary directory
tmpdir="zeopp_temp_$$"
mkdir -p "$tmpdir"

# Count CIF files
count=$(find "$INPUT_DIR" -maxdepth 1 -name "*.cif" -type f | wc -l)
if [ "$count" -eq 0 ]; then
  echo "No CIF files in $INPUT_DIR"; exit 1
fi

echo "Processing $count CIF files using $NPROCS parallel processes..."
# CSV header
echo "filename,Di_A,Df_A,Dif_A,density_gcc,unit_cell_volume_A3,SA_A2,SA_m2_cm3,SA_m2_g,AV_A3,AV_fraction,AV_cm3_g,num_channels,num_pockets" > "$OUTPUT_CSV"

# Function to process a single CIF file
process_cif() {
  local cif="$1"
  local tmpdir="$2"
  local base=$(basename "$cif" .cif)
  
  # define files
  local resf="$tmpdir/${base}_$.res"
  local saf="$tmpdir/${base}_$.sa"
  local volf="$tmpdir/${base}_$.vol"
  local chf="$tmpdir/${base}_$.chan"

  # defaults
  local Di=Df=Dif=density=unitvol=SA_A2=SA_m2_cm3=SA_m2_g=AV_A3=AV_frac=AV_cm3_g=numchan=numpock="NA"

  # 1) pore diameters
  $NETWORK_CMD -ha -res "$resf" "$cif" >/dev/null 2>&1
  if [ -s "$resf" ]; then
    read -r _ Di Df Dif < "$resf"
  fi

  # 2) surface area
  $NETWORK_CMD -ha -sa $PROBE_RADIUS $PROBE_RADIUS $SA_SAMPLES "$saf" "$cif" >/dev/null 2>&1
  if [ -s "$saf" ]; then
    unitvol=$(grep -oP 'Unitcell_volume:\s*\K[0-9.]+' "$saf" | head -1)
    density=$(grep -oP 'Density:\s*\K[0-9.]+' "$saf" | head -1)
    SA_A2=$(grep -oP '(?<!N)ASA_A\^2:\s*\K[0-9.]+' "$saf" | head -1)
    SA_m2_cm3=$(grep -oP '(?<!N)ASA_m\^2/cm\^3:\s*\K[0-9.]+' "$saf" | head -1)
    SA_m2_g=$(grep -oP '(?<!N)ASA_m\^2/g:\s*\K[0-9.]+' "$saf" | head -1)
    numpock=$(grep -oP 'Number_of_pockets:\s*\K[0-9]+' "$saf" | head -1)
    # fallbacks
    [ -z "$unitvol" ] && unitvol="NA"
    [ -z "$density" ] && density="NA"
    [ -z "$SA_A2" ] && SA_A2="NA"
    [ -z "$SA_m2_cm3" ] && SA_m2_cm3="NA"
    [ -z "$SA_m2_g" ] && SA_m2_g="NA"
    [ -z "$numpock" ] && numpock="NA"
  fi

  # 3) accessible volume
  $NETWORK_CMD -ha -vol $PROBE_RADIUS $PROBE_RADIUS $VOL_SAMPLES "$volf" "$cif" >/dev/null 2>&1
  if [ -s "$volf" ]; then
    AV_A3=$(grep -oP '(?<!N)AV_A\^3:\s*\K[0-9.]+' "$volf" | head -1)
    AV_frac=$(grep -oP '(?<!N)AV_Volume_fraction:\s*\K[0-9.]+' "$volf" | head -1)
    AV_cm3_g=$(grep -oP '(?<!N)AV_cm\^3/g:\s*\K[0-9.]+' "$volf" | head -1)
    [ -z "$AV_A3" ] && AV_A3="NA"
    [ -z "$AV_frac" ] && AV_frac="NA"
    [ -z "$AV_cm3_g" ] && AV_cm3_g="NA"
  fi

  # 4) channels
  $NETWORK_CMD -ha -chan $PROBE_RADIUS "$cif" > "$chf" 2>&1
  if [ -s "$chf" ]; then
    numchan=$(grep -oP 'Identified\s*\K[0-9]+(?=\s*channels)' "$chf" | head -1)
    [ -z "$numchan" ] && numchan=$(grep -oP 'Number_of_channels:\s*\K[0-9]+' "$chf" | head -1)
    [ -z "$numchan" ] && numchan="NA"
  fi

  # output result
  echo "$base,$Di,$Df,$Dif,$density,$unitvol,$SA_A2,$SA_m2_cm3,$SA_m2_g,$AV_A3,$AV_frac,$AV_cm3_g,$numchan,$numpock"

  # clean
  rm -f "$resf" "$saf" "$volf" "$chf"
}

# Export function and variables for parallel processing
export -f process_cif
export NETWORK_CMD PROBE_RADIUS SA_SAMPLES VOL_SAMPLES

start=$(date +%s)

# Alternative: Use background processes with job control
echo "Using background processes with $NPROCS parallel jobs..."

# Create array of CIF files
mapfile -t cif_files < <(find "$INPUT_DIR" -maxdepth 1 -name "*.cif" -type f)
total_files=${#cif_files[@]}

# Process files in batches
for ((i=0; i<total_files; i+=NPROCS)); do
  # Start batch of background processes
  for ((j=0; j<NPROCS && i+j<total_files; j++)); do
    cif_file="${cif_files[i+j]}"
    {
      result=$(process_cif "$cif_file" "$tmpdir")
      echo "$result" >> "$OUTPUT_CSV"
    } &
  done
  
  # Wait for this batch to complete
  wait
  
  # Progress update
  completed=$((i + NPROCS))
  if [ $completed -gt $total_files ]; then
    completed=$total_files
  fi
  echo "Completed $completed/$total_files files..."
done

# final cleanup
rm -rf "$tmpdir"
end=$(date +%s)
dur=$((end-start))
echo "Finished $count files in $((dur/60))m$((dur%60))s using $NPROCS processes"