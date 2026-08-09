"""Reproducible wall-clock benchmark for the 1D native engine."""

from __future__ import annotations

import argparse
from statistics import median
from time import perf_counter

import uc_sgsim as uc

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid-size', type=int, default=400)
    parser.add_argument('--realizations', type=int, default=250)
    parser.add_argument('--neighbors', type=int, default=12)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--include-python', action='store_true')
    return parser.parse_args()


def measure(
    args: argparse.Namespace,
    backend: str,
    kriging: str,
    cache: bool,
) -> float:
    durations = []
    for repeat in range(args.repeats + 1):
        simulator = uc.SequentialGaussianSimulator(
            args.grid_size,
            uc.Exponential(40, 1, 18, sill=1.7, nugget=0.2),
            backend=backend,
            kriging=kriging,
            max_neighbors=args.neighbors,
            constant_path=cache,
            covariance_cache=cache,
        )
        started = perf_counter()
        simulator.simulate(args.realizations, seed=2026, workers=args.workers)
        duration = perf_counter() - started
        if repeat > 0:
            durations.append(duration)
    return median(durations)


def main() -> None:
    args = parse_args()
    backends = ['c', 'python'] if args.include_python else ['c']
    print(f'workers={args.workers}')
    print('backend  kriging   cache  median_seconds')
    for backend in backends:
        for kriging in ('simple',):
            for cache in (False, True):
                duration = measure(args, backend, kriging, cache)
                print(
                    '{:<8} {:<9} {:<6} {:.6f}'.format(
                        backend,
                        kriging,
                        str(cache),
                        duration,
                    ),
                )
    if resource is not None:
        print(f'peak_rss_kib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')


if __name__ == '__main__':
    main()
