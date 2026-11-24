"""Project status entry. New scientific runs require a separately frozen protocol."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('describe', help='Report current status without running an experiment')
    parser.parse_args()
    print('研究：复杂电磁环境下LLM辅助UAV动态路径重规划；尚未证明LLM任务收益。')
    print('当前：E1 两场事件选路已关闭；LLM 四次实际选路，均与同菜单强 N 同轨。')
    print('下一步：审计持续预警中的有效首分岔，计划图外中间目标点与同资源强 N 对照。')
    print('不开放新运行；接手见 docs/AGENT_HANDOFF.md，双图及归档见 outputs/index.html。')


if __name__ == '__main__':
    main()
