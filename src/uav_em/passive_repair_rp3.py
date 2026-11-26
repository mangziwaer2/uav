"""Independent RP3 batch after RP2's final-message phase validation failure."""

import argparse
from pathlib import Path

from .passive_repair_rp2 import ROOT, freeze, run


OUTPUT = ROOT / 'outputs/passive_repair_RP3_20261126'
CONTRACT = ROOT / 'docs/passive_repair_RP3_contract.md'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('freeze')
    execute = commands.add_parser('run')
    execute.add_argument('digest')
    args = parser.parse_args()
    if args.command == 'freeze':
        print(freeze(args.out, label='RP3', contract=CONTRACT), flush=True)
    else:
        import sys
        if sys.stdin.isatty():
            import termios
            settings = termios.tcgetattr(sys.stdin)
            settings[3] &= ~(termios.ECHO | termios.ICANON)
            settings[6][termios.VMIN], settings[6][termios.VTIME] = 1, 0
            termios.tcsetattr(sys.stdin, termios.TCSANOW, settings)
        run(args.out, args.digest, contract=CONTRACT)


if __name__ == '__main__':
    main()
