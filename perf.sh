#!/bin/bash
sudo sysctl kernel.perf_event_paranoid=-1
perf record -g --call-graph dwarf ./moexpr eval 014