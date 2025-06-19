#!/bin/bash

# Diagnostic script to check zeopp output format
# Run this on a single CIF file to see the actual output

CIF_FILE="${1:-test.cif}"
PROBE_RADIUS="${2:-1.9}"

if [ ! -f "$CIF_FILE" ]; then
    echo "Usage: $0 <cif_file> [probe_radius]"
    echo "Error: CIF file not found: $CIF_FILE"
    exit 1
fi

echo "=== Testing zeopp output format for: $CIF_FILE ==="
echo "Probe radius: $PROBE_RADIUS Å"
echo ""

BASENAME=$(basename "$CIF_FILE" .cif)
TEMP_DIR="temp_diagnostic_$$"
mkdir -p "$TEMP_DIR"

# Load module if needed
module load zeopp-lsmo 2>/dev/null

echo "=== Running network commands ==="

# 1. Test RES file output
echo -e "\n--- Testing .res file (pore diameters) ---"
network -ha -res "$TEMP_DIR/test.res" "$CIF_FILE" 2>&1
echo "RES file content:"
cat "$TEMP_DIR/test.res" 2>/dev/null || echo "No .res file generated"

# 2. Test SA file output
echo -e "\n--- Testing .sa file (surface area) ---"
network -ha -sa $PROBE_RADIUS $PROBE_RADIUS 100 "$TEMP_DIR/test.sa" "$CIF_FILE" 2>&1
echo "SA file content:"
cat "$TEMP_DIR/test.sa" 2>/dev/null || echo "No .sa file generated"
echo -e "\nLines starting with @:"
grep "^@" "$TEMP_DIR/test.sa" 2>/dev/null || echo "No @ lines found"

# 3. Test VOL file output
echo -e "\n--- Testing .vol file (accessible volume) ---"
network -ha -vol $PROBE_RADIUS $PROBE_RADIUS 1000 "$TEMP_DIR/test.vol" "$CIF_FILE" 2>&1
echo "VOL file content:"
cat "$TEMP_DIR/test.vol" 2>/dev/null || echo "No .vol file generated"
echo -e "\nLines starting with @:"
grep "^@" "$TEMP_DIR/test.vol" 2>/dev/null || echo "No @ lines found"

# 4. Test VOLPO file output
echo -e "\n--- Testing .volpo file (probe-occupiable volume) ---"
network -ha -volpo $PROBE_RADIUS $PROBE_RADIUS 1000 "$TEMP_DIR/test.volpo" "$CIF_FILE" 2>&1
echo "VOLPO file content:"
cat "$TEMP_DIR/test.volpo" 2>/dev/null || echo "No .volpo file generated"
echo -e "\nLines starting with @:"
grep "^@" "$TEMP_DIR/test.volpo" 2>/dev/null || echo "No @ lines found"

# 5. Test CHAN output
echo -e "\n--- Testing channel analysis ---"
network -ha -chan $PROBE_RADIUS "$CIF_FILE" > "$TEMP_DIR/test.chan" 2>&1
echo "CHAN file content (first 20 lines):"
head -20 "$TEMP_DIR/test.chan" 2>/dev/null || echo "No .chan file generated"

# Try alternative output format - check if data goes to stdout instead
echo -e "\n=== Testing stdout output ==="
echo -e "\n--- SA calculation stdout ---"
network -ha -sa $PROBE_RADIUS $PROBE_RADIUS 100 "$CIF_FILE" 2>&1 | head -20

echo -e "\n--- VOL calculation stdout ---"
network -ha -vol $PROBE_RADIUS $PROBE_RADIUS 1000 "$CIF_FILE" 2>&1 | head -20

# Cleanup
rm -rf "$TEMP_DIR"

echo -e "\n=== Diagnostic complete ==="
echo "If output files are empty, zeopp might be outputting to stdout instead of files."
echo "Check the output above to see the actual format."