#!/usr/bin/env bash
set -euo pipefail

INTERVAL="${INTERVAL:-1}"
DURATION="${DURATION:-0}"
PID="${PID:-}"
OUT="${OUT:-}"
DISK="${DISK:-disk0}"

if [[ -z "$PID" ]]; then
  PID="$(pgrep -n llama-cli 2>/dev/null || true)"
fi

page_size="$(vm_stat | awk '/page size of/ { gsub("\\.", "", $8); print $8; exit }')"
page_size="${page_size:-16384}"

disk0_line="$(iostat -Id "$DISK" 1 1 | tail -n 1 || true)"
prev_disk_mb="$(awk '{ print $3 + 0 }' <<<"$disk0_line")"
prev_pageins="$(vm_stat | awk '/Pageins/ { gsub("\\.", "", $2); print $2 + 0; exit }')"
start_epoch="$(date +%s)"

if [[ -n "$OUT" ]]; then
  mkdir -p "$(dirname "$OUT")"
  printf "timestamp,elapsed_s,free_pct,app_rss_gib,disk_read_mib_s,pagein_mib_s,pid\n" > "$OUT"
fi

bar() {
  local pct="$1"
  local width=24
  local filled
  local empty
  filled="$(awk -v p="$pct" -v w="$width" 'BEGIN { v=int(p*w/100); if (v<0) v=0; if (v>w) v=w; print v }')"
  empty=$((width - filled))
  printf "["
  printf "%${filled}s" "" | tr " " "#"
  printf "%${empty}s" "" | tr " " "."
  printf "]"
}

rss_gib_for_pid() {
  local target="$1"
  if [[ -z "$target" ]]; then
    printf "0.00"
    return
  fi
  ps -o rss= -p "$target" 2>/dev/null | awk '{ printf "%.2f", ($1 * 1024) / 1024 / 1024 / 1024 }'
}

while true; do
  now_epoch="$(date +%s)"
  elapsed=$((now_epoch - start_epoch))

  if [[ "$DURATION" != "0" && "$elapsed" -gt "$DURATION" ]]; then
    break
  fi

  if [[ -z "$PID" ]]; then
    PID="$(pgrep -n llama-cli 2>/dev/null || true)"
  fi

  free_pct="$(memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/ { gsub("%", "", $2); print $2 + 0; exit }')"
  free_pct="${free_pct:-0}"
  used_pct="$(awk -v f="$free_pct" 'BEGIN { printf "%.0f", 100 - f }')"
  rss_gib="$(rss_gib_for_pid "$PID")"

  current_pageins="$(vm_stat | awk '/Pageins/ { gsub("\\.", "", $2); print $2 + 0; exit }')"
  pagein_delta=$((current_pageins - prev_pageins))
  pagein_mib_s="$(awk -v p="$pagein_delta" -v s="$page_size" -v i="$INTERVAL" 'BEGIN { printf "%.1f", (p*s)/1024/1024/i }')"
  prev_pageins="$current_pageins"

  disk_line="$(iostat -Id "$DISK" 1 1 | tail -n 1 || true)"
  current_disk_mb="$(awk '{ print $3 + 0 }' <<<"$disk_line")"
  disk_read_mib_s="$(awk -v c="$current_disk_mb" -v p="$prev_disk_mb" -v i="$INTERVAL" 'BEGIN { d=(c-p); if (d<0) d=0; printf "%.1f", d/i }')"
  prev_disk_mb="$current_disk_mb"

  clear 2>/dev/null || printf "\n\n"
  printf "MoE-MADV live monitor\n"
  printf "Clock:   %s\n" "$(date "+%H:%M:%S")"
  printf "Elapsed: %02d:%02d\n" $((elapsed / 60)) $((elapsed % 60))
  printf "Memory:  %s %s%% used, %s%% free\n" "$(bar "$used_pct")" "$used_pct" "$free_pct"
  printf "llama:   PID=%s RSS=%s GiB\n" "${PID:-not-running}" "$rss_gib"
  printf "Disk:    %s read %.1f MiB/s\n" "$DISK" "$disk_read_mib_s"
  printf "Page-in: %.1f MiB/s\n" "$pagein_mib_s"
  printf "\nTip: show Activity Monitor GPU History next to this terminal for GPU charts.\n"

  if [[ -n "$OUT" ]]; then
    printf "%s,%s,%s,%s,%s,%s,%s\n" \
      "$(date -u "+%Y-%m-%dT%H:%M:%SZ")" \
      "$elapsed" "$free_pct" "$rss_gib" "$disk_read_mib_s" "$pagein_mib_s" "${PID:-}" >> "$OUT"
  fi

  sleep "$INTERVAL"
done
