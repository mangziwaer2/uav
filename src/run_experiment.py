"""Project status entry. New scientific runs require a separately frozen protocol."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('describe', help='Report current status without running an experiment')
    parser.parse_args()
    print('研究：复杂电磁环境下LLM辅助UAV动态路径重规划；尚未证明LLM任务收益。')
    print('当前：无功率延续源池/飞行库；独立多阶段条件策略组件，强N已解决该有限域。')
    print('下一步：任务相关模型结构诊断/修订的小型强N对照，先冻结合同与资源。')
    print('旧实验预算全部关闭，历史驱动已删除；接手见 docs/AGENT_HANDOFF.md。')


if __name__ == '__main__':
    main()
