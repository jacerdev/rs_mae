SECRETS_PATH="/home/users/j/jacer/store/personal.env"

get_value() {
    # usage: get_value "file_path" "var_name"
    local file_path="$1"
    local var_name="$2"

    if [[ ! -f "$file_path" ]]; then
        echo "Error: File '$file_path' not found" >&2
        return 1
    fi

    grep -E "^${var_name}=" "$file_path" | sed -E 's/^'"$var_name"'=//; s/^"(.*)"$/\1/'
}

send_telegram() {
    # usage: send_telegram "message"
    local BOT_TOKEN=$(get_value "$SECRETS_PATH" "BOT_TOKEN")
    local CHAT_ID=$(get_value "$SECRETS_PATH" "CHAT_ID")
    local message="$1"
    echo "📣: $message"
    
    # URL encode special characters ('&', '+') that break x-www-form-urlencoded
    local encoded_message="${message//&/%26}"
    encoded_message="${encoded_message//+/%2B}"
    
    wget --quiet --spider --post-data="chat_id=$CHAT_ID&text=$encoded_message" \
    "https://api.telegram.org/bot$BOT_TOKEN/sendMessage"
}

download_extract_reben() {
    URL="$1"
    OUTPUT_DIR="$2"

    mkdir -p "$OUTPUT_DIR"

    # download the file
    wget --content-disposition -c "$URL" -P "$OUTPUT_DIR"

    # get the downloaded filename
    DOWNLOADED_FILE="$OUTPUT_DIR/$(basename "$URL")"
    echo "Downloaded $DOWNLOADED_FILE to $OUTPUT_DIR"

    # extract
    pv "$DOWNLOADED_FILE" | tar -I zstd -xf - -C "$OUTPUT_DIR"
    echo "Extraction completed to $OUTPUT_DIR"
    
    # ./rico-hdl.AppImage bigearthnet --bigearthnet-s2-dir BigEarthNet-S2/ --bigearthnet-reference-maps-dir Reference_Maps/ --target-dir BigEarthNet-S2_lmdb
}

on_exit() {
    local exit_code=$?
    # Defuse the traps immediately to prevent recursion
    trap - EXIT TERM INT

    # Kill background python process if it's still running
    if [ -n "$child_pid" ]; then kill "$child_pid" 2>/dev/null || true; fi

    set +e  # Disable strict exit for the cleanup
    
    if [ "$exit_code" -eq 0 ]; then send_telegram "✅ Finished training successfully at $(date "+%Y-%m-%d %H:%M:%S")"
    else send_telegram "❌ Training failed (exit code $exit_code) on $(hostname) at $(date "+%Y-%m-%d %H:%M:%S")"
    fi
    #rsync -av /scratch/jacer/outputs/ /home/users/j/jacer/outputs/
    #send_telegram "📁 Finished copying outputs to home directory"
    exit $exit_code
}
