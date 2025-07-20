#!/bin/bash

# Script to download files while bypassing Cloudflare 403 errors
# Usage: ./download.sh <URL> <output_filename>

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <URL> <output_filename>"
    echo "Example: $0 'https://example.com/file.pdf' 'document.pdf'"
    exit 1
fi

URL="$1"
OUTPUT_FILE="$2"

echo "Downloading: $URL"
echo "Output file: $OUTPUT_FILE"

# Download with curl using browser-like headers to bypass Cloudflare
curl -L -o "$OUTPUT_FILE" \
     -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \
     -H "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8" \
     -H "Accept-Language: en-US,en;q=0.9" \
     -H "Accept-Encoding: gzip, deflate, br" \
     -H "Connection: keep-alive" \
     -H "Upgrade-Insecure-Requests: 1" \
     -H "Sec-Fetch-Dest: document" \
     -H "Sec-Fetch-Mode: navigate" \
     -H "Sec-Fetch-Site: none" \
     -H "Cache-Control: max-age=0" \
     --compressed \
     --location \
     --max-redirs 10 \
     --retry 3 \
     --retry-delay 2 \
     --connect-timeout 30 \
     --max-time 300 \
     "$URL"

# Check if download was successful
if [ $? -eq 0 ] && [ -f "$OUTPUT_FILE" ] && [ -s "$OUTPUT_FILE" ]; then
    echo "Download completed successfully: $OUTPUT_FILE"
    echo "File size: $(ls -lh "$OUTPUT_FILE" | awk '{print $5}')"
else
    echo "Download failed or file is empty"
    # Clean up empty file if it exists
    [ -f "$OUTPUT_FILE" ] && [ ! -s "$OUTPUT_FILE" ] && rm "$OUTPUT_FILE"
    exit 1
fi
