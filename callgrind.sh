#!/bin/bash
valgrind --tool=callgrind --callgrind-out-file=callgrind.out ./moexpr eval 014
