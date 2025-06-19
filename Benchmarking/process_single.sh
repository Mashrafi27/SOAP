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