#!/bin/bash
if [ $# -eq 0 ]; then
  echo "Usage: $0 <command>"
  echo "Example: $0 \"./moexpr eval 014\""
  exit 1
fi
eval "valgrind --tool=callgrind --callgrind-out-file=callgrind.out $1"
