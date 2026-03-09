#!/bin/bash
perf script | c++filt | gprof2dot -f perf -n 1.0 -e 0.5 | dot -Tsvg -o profile.svg