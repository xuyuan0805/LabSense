from __future__ import annotations

import threading
import time
import urllib.request
import webbrowser

import uvicorn


HOST = "127.0.0.1"
PORT = 8765
URL = f"http://{HOST}:{PORT}"


def open_when_ready() -> None:
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"{URL}/api/health", timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(URL)
                    return
        except Exception:
            time.sleep(0.25)
    print(f"服务已启动，但浏览器未自动打开。请访问：{URL}")


if __name__ == "__main__":
    print("LabSense 正在本地启动……")
    print(f"课堂地址：{URL}")
    print("按 Ctrl+C 可结束服务。")
    threading.Thread(target=open_when_ready, daemon=True).start()
    uvicorn.run("app.main:app", host=HOST, port=PORT, log_level="warning")
