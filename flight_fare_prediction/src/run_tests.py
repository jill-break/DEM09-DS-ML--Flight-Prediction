#!/usr/bin/env python3
"""
Test Runner Script

Convenient script to run tests with different configurations.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --unit       # Run unit tests only
    python run_tests.py --coverage   # Run with coverage report
    python run_tests.py --fast       # Skip slow tests
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd: list, description: str) -> int:
    """Run a command and print status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"{'='*70}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run project tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--fast', action='store_true', help='Skip slow tests')
    parser.add_argument('--html', action='store_true', help='Generate HTML coverage report')
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Base command
    cmd = ['pytest', 'tests/']
    
    # Add verbosity
    if args.verbose:
        cmd.append('-v')
    
    # Add markers
    if args.unit:
        cmd.extend(['-m', 'unit'])
    elif args.integration:
        cmd.extend(['-m', 'integration'])
    
    # Skip slow tests
    if args.fast:
        cmd.extend(['-m', 'not slow'])
    
    # Add coverage
    if args.coverage or args.html:
        cmd.extend(['--cov=src', '--cov-report=term-missing'])
        if args.html:
            cmd.append('--cov-report=html')
    
    # Parallel execution
    if args.parallel:
        cmd.extend(['-n', 'auto'])
    
    # Run tests
    returncode = run_command(cmd, 'Test Suite')
    
    if returncode == 0:
        print(f"\n{'='*70}")
        print("All tests passed!")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print("Some tests failed!")
        print(f"{'='*70}\n")
    
    return returncode


if __name__ == '__main__':
    sys.exit(main())