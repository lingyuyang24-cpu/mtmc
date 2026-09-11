"""本地工作台入口，模型在任务启动后加载。"""
import argparse
from pathlib import Path
import os


def main():
    parser = argparse.ArgumentParser(description='MTMC 多摄像头追踪工作台')
    parser.add_argument('--host', default='127.0.0.1', help='默认仅允许本机访问')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    import uvicorn
    uvicorn.run('web.app:app', host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == '__main__':
    main()
