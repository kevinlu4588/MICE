#!/bin/bash

# === CONFIGURATION ===
PASSWORD="temppass"
USER="kevin"
HOSTS=(kobe ei macondo karasuno karakuri hawaii tokyo umibozu kyoto saitama bippu osaka hamada kumamoto fukuyama sendai andromeda hokkaido cancun kameoka)
SSHPASS="$HOME/.local/bin/sshpass"

# === RESULT STORAGE ===
declare -A FREE_GPUS

# === CHECK HOSTS ===
for HOST in "${HOSTS[@]}"; do
    echo -e "\n=== Checking $HOST ==="

    OUTPUT=$($SSHPASS -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$USER@$HOST" "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits" 2>/dev/null)

    if [ $? -ne 0 ]; then
        echo "$HOST is unreachable ❌"
        continue
    fi

    IFS=$'\n' read -rd '' -a LINES <<<"$OUTPUT"

    FREE_COUNT=0
    DETAILS=()

    for LINE in "${LINES[@]}"; do
        MEM=$(echo "$LINE" | cut -d',' -f1 | tr -d ' ')
        UTIL=$(echo "$LINE" | cut -d',' -f2 | tr -d ' ')
        
        if [ "$MEM" -le 8000 ]; then
            DETAILS+=("Memory: ${MEM}MiB, Utilization: ${UTIL}%")
            ((FREE_COUNT++))
        fi
    done

    if [ "$FREE_COUNT" -gt 0 ]; then
        FREE_GPUS["$HOST"]="${DETAILS[*]}"
        echo "$HOST has $FREE_COUNT free GPU(s) ✅"
    else
        echo "$HOST is busy ❌"
    fi
done

# === SUMMARY ===
echo -e "\n========== SUMMARY OF AVAILABLE HOSTS =========="
for HOST in "${!FREE_GPUS[@]}"; do
    echo -e "$HOST:\n  ${FREE_GPUS[$HOST]}"
done
