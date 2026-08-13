#!/usr/bin/env python3
import os
import csv
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# Đường dẫn file CSV thực tế
CSV_FILE_PATH = (
    r"C:\Workspace - Copy\soict_pro\edge-uav-traffic\results\runtime_telemetry.csv"
)

# Đường dẫn frame ảnh realtime
FRAME_IMG_PATH = (
    r"C:\Workspace - Copy\soict_pro\edge-uav-traffic\results\frames\latest_frame.jpg"
)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass  # Tắt log request để terminal không bị rác

    def _json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def _file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except Exception:
            self.send_error(404)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._file(
                os.path.join(os.path.dirname(__file__), "dashboard.html"),
                "text/html; charset=utf-8",
            )
        elif u.path == "/api/telemetry":
            self._json(self.read_csv())
        elif u.path == "/api/video_feed":
            # Trả về ảnh frame mới nhất
            self._file(FRAME_IMG_PATH, "image/jpeg")
        else:
            self.send_error(404)

    def read_csv(self):
        data = []
        if os.path.exists(CSV_FILE_PATH):
            try:
                with open(CSV_FILE_PATH, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.startswith("\ufeff"):
                        content = content[1:]
                    reader = csv.DictReader(io.StringIO(content))
                    rows = list(reader)
                    for row in rows[-100:]:
                        try:
                            data.append(
                                {
                                    "frame_id": int(row.get("frame_id", 0)),
                                    "timestamp_ms": float(row.get("timestamp_ms", 0)),
                                    "n_tracks": int(row.get("n_tracks", 0)),
                                    "vehicle_count": int(row.get("vehicle_count", 0)),
                                    "person_count": int(row.get("person_count", 0)),
                                    "congestion_level": row.get(
                                        "congestion_level", "normal"
                                    ),
                                    "cls": {
                                        "pedestrian": int(row.get("cls_pedestrian", 0)),
                                        "people": int(row.get("cls_people", 0)),
                                        "bicycle": int(row.get("cls_bicycle", 0)),
                                        "car": int(row.get("cls_car", 0)),
                                        "van": int(row.get("cls_van", 0)),
                                        "truck": int(row.get("cls_truck", 0)),
                                        "tricycle": int(row.get("cls_tricycle", 0)),
                                        "awning-tricycle": int(
                                            row.get("cls_awning-tricycle", 0)
                                        ),
                                        "bus": int(row.get("cls_bus", 0)),
                                        "motor": int(row.get("cls_motor", 0)),
                                    },
                                    "roi_intersection": int(
                                        row.get("roi_intersection", 0)
                                    ),
                                    "roi_full_frame": int(row.get("roi_full_frame", 0)),
                                    "line_fwd": int(row.get("line_main_road_fwd", 0)),
                                    "line_bwd": int(row.get("line_main_road_bwd", 0)),
                                    "pre": float(row.get("pre_ms", 0)),
                                    "infer": float(row.get("infer_ms", 0)),
                                    "post": float(row.get("post_ms", 0)),
                                    "track": float(row.get("track_ms", 0)),
                                    "total": float(row.get("total_ms", 0)),
                                    "session_id": row.get("session_id", ""),
                                }
                            )
                        except Exception:
                            continue
            except Exception:
                pass
        return data


if __name__ == "__main__":
    print(f"Đang theo dõi file CSV: {CSV_FILE_PATH}")
    print(f"Đang theo dõi frame ảnh: {FRAME_IMG_PATH}")
    print("Dashboard chạy tại: http://127.0.0.1:8000/")
    srv = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    srv.serve_forever()
