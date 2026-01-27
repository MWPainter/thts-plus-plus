#!/bin/bash
gprof2dot -f callgrind -n 1.0 -e 0.5 callgrind.out | dot -Tsvg -o profile.svg
