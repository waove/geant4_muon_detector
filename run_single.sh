#!/bin/bash

EXECUTABLE="./mingle_master_production"
REFLECTIVITIES=("0.98")
FINISHES=("groundbackpainted")
SIGMAS=("0.2")
LOBE_FRAC="0.5"

# Set ENABLE_PATH_TRACKING=1 to enable photon path recording (slow)
PATH_TRACKING="1"


for REFL in "${REFLECTIVITIES[@]}"; do
    for FINISH in "${FINISHES[@]}"; do
        for SIGMA in "${SIGMAS[@]}"; do

            # SigmaAlpha only meaningful for ground finishes — skip for polished
            if [[ "$FINISH" == polished* ]] && [[ "$SIGMA" != "0.0" ]]; then
                echo "--- Skipping sigma=$SIGMA for polished finish $FINISH ---"
                continue
            fi

            echo "--- Reflectivity: $REFL  Finish: $FINISH  Sigma: $SIGMA  Lobe: $LOBE_FRAC  PathTracking: $PATH_TRACKING ---"
            ENABLE_PATH_TRACKING=0 \
            ENABLE_PARENTAGE=0 \
            WRAPPING_REFLECTIVITY=$REFL \
            WRAPPING_FINISH=$FINISH \
            WRAPPING_SIGMA=$SIGMA \
            WRAPPING_LOBE=$LOBE_FRAC \
            ENABLE_PATH_TRACKING=$PATH_TRACKING \
                $EXECUTABLE run.mac > /dev/null 2>&1

            if [ -f "scoring.root" ]; then
                mv scoring.root "scoring_test.root"
            else
                echo "  Warning: no output for refl=$REFL finish=$FINISH sigma=$SIGMA"
            fi

        done
    done
done

echo "Done. Total runs: $((${#REFLECTIVITIES[@]} * ${#FINISHES[@]} * ${#SIGMAS[@]}))"