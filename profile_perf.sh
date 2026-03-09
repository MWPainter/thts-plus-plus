#!/bin/bash
if [ $# -eq 0 ]; then
  echo "Usage: $0 <command>"
  echo "Example: $0 \"./moexpr eval 014\""
  exit 1
fi
sudo sysctl kernel.perf_event_paranoid=-1
eval "perf record -g --call-graph dwarf $1"