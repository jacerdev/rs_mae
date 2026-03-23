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
    # the message should never contain an ampersand (&) or a plus (+)
    local BOT_TOKEN=$(get_value "$SECRETS_PATH" "BOT_TOKEN")
    local CHAT_ID=$(get_value "$SECRETS_PATH" "CHAT_ID")
    local message="$1"
    echo "📣: $message"
    wget --quiet --spider --post-data="chat_id=$CHAT_ID&text=$message" \
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
