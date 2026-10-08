import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from mlops_ref.drift import drift_report, send_alert
from mlops_ref.features import MODEL_FEATURES, simulate_current_batch


def test_no_intended_drift_stays_below_threshold(featured, params, tmp_path):
    tr, te = featured
    r = drift_report(tr, simulate_current_batch(te, strength=0), MODEL_FEATURES, params["drift"]["share_threshold"], tmp_path)
    assert r["alert"] is False and r["drift_share"] <= params["drift"]["share_threshold"]
    assert len(r["columns"]) == len(MODEL_FEATURES)
    assert (tmp_path / "drift_report.html").stat().st_size > 10_000
    assert json.loads((tmp_path / "drift_summary.json").read_text())["alert"] is False
    assert send_alert(r) is False


def test_simulated_shift_fires_alert(featured, params):
    tr, te = featured
    r = drift_report(tr, simulate_current_batch(te, strength=1), MODEL_FEATURES, params["drift"]["share_threshold"])
    assert r["alert"] is True and r["drift_share"] > params["drift"]["share_threshold"]
    assert {"credit_amount", "duration_months", "age_years", "log_credit_amount"} <= set(r["drifted_columns"])


def test_alert_posts_to_webhook():
    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            got.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.handle_request)
    t.start()
    result = {"alert": True, "drift_share": 0.5, "share_threshold": 0.3, "drifted_columns": ["age_years", "credit_amount"]}
    assert send_alert(result, f"http://127.0.0.1:{srv.server_port}/hook") is True
    t.join(5)
    srv.server_close()
    assert got and "50% of columns drifted" in got[0]["text"] and "age_years" in got[0]["text"]
