"""独立启动展车分析服务；不加载模型、不控制追踪进程。"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="MTMC 独立展车分析")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--tracking-url",
        default=os.environ.get("MTMC_TRACKING_URL", "http://127.0.0.1:8765"),
    )
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    os.environ["MTMC_TRACKING_URL"] = args.tracking_url
    import uvicorn

    uvicorn.run(
        "modules.showroom.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=1,
        access_log=False,
    )


if __name__ == "__main__":
    main()
