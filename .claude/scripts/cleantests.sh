#!/bin/bash

echo_and_run() {
    local cmd="$*"
    echo "$ $cmd"
    eval "$cmd"
    return $?
}

echo_and_run './build/mad_escape_tests --gtest_color=no 2>&1 | grep -v "^\[" | grep -v "^Running\|^Global\|^Note:" | grep -v ": Skipped$" | grep -v "^$" | wc -l'