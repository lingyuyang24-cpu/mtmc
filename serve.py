"""本地工作台入口，模型在任务启动后加载。"""
import argparse
from pathlib import Path
import os


def main():
    parser = argparse.ArgumentParser(description='MTMC 多摄像头追踪工作台')
    parser.add_argument('--host', default='127.0.0.1', help='默认仅允许本机访问')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--showroom-url', help='可选：显示独立展车服务入口，例如 http://127.0.0.1:8766')
    args = parser.parse_args()
    if args.showroom_url:
        os.environ['MTMC_SHOWROOM_URL'] = args.showroom_url
    os.chdir(Path(__file__).resolve().parent)
    import uvicorn
    uvicorn.run('web.app:app', host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == '__main__':
    main()
