"""HttpPriceSource (the api's client of the prices service) against a tiny real HTTP server, and the switch in
make_agent. The stub serves exactly what the prices service serves, using the prices service's own payload code."""
import json
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from energy_agent.sources import HttpPriceSource
from energy_agent.tools import EnergyTools
from prices_service.app import price_payload
from prices_service.sources import DemoPriceSource

DAYS = [date(2026, 11, 16), date(2026, 3, 29), date(2026, 10, 25)]  # normal, 23-hour and 25-hour days


# ---- HttpPriceSource against a tiny real HTTP server -------------------------------------------

class Stub(BaseHTTPRequestHandler):
    mode = "ok"

    def do_GET(self):
        if Stub.mode == "error":
            status, body = 502, {"detail": "The price source is unavailable: boom"}
        else:
            day = date.fromisoformat(parse_qs(urlparse(self.path).query)["date"][0])
            status, body = 200, price_payload(DemoPriceSource().get_prices(day), day, "DE-LU", "synthetic_demo_data")
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def service_url():
    Stub.mode = "ok"
    server = HTTPServer(("127.0.0.1", 0), Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("day", DAYS)
def test_http_source_returns_exactly_what_the_service_serves(service_url, day):
    direct = DemoPriceSource().get_prices(day)
    remote = HttpPriceSource(service_url).get_prices(day)
    assert [(p.timestamp.timestamp(), p.price_eur_mwh) for p in remote] == [(p.timestamp.timestamp(), p.price_eur_mwh) for p in direct]
    assert [p.timestamp.hour for p in remote] == [p.timestamp.hour for p in direct]   # local hours survive, DST included


def test_tools_give_identical_answers_whether_prices_come_from_the_service_or_directly(service_url):
    remote, direct = EnergyTools(HttpPriceSource(service_url)), EnergyTools(DemoPriceSource())
    for day in DAYS:
        iso = day.isoformat()
        assert remote.analyze_prices(iso) == direct.analyze_prices(iso)
        assert remote.find_cheapest_window(iso, 3, 7, 22) == direct.find_cheapest_window(iso, 3, 7, 22)
        assert remote.estimate_run_cost(iso, 11, 3, 2.0) == direct.estimate_run_cost(iso, 11, 3, 2.0)
        assert remote.get_hourly_prices(iso) == direct.get_hourly_prices(iso)


def test_tool_results_are_labelled_with_the_source_the_service_reports(service_url):
    result = EnergyTools(HttpPriceSource(service_url)).get_hourly_prices("2026-11-16")
    assert result["source"] == "synthetic_demo_data"
    assert "synthetic" in result["note"].lower()


def test_service_errors_surface_as_readable_runtime_errors(service_url):
    Stub.mode = "error"
    with pytest.raises(RuntimeError) as caught:
        HttpPriceSource(service_url).get_prices(DAYS[0])
    assert "502" in str(caught.value) and "boom" in str(caught.value)


def test_unreachable_service_is_a_readable_runtime_error():
    with pytest.raises(RuntimeError, match="unreachable"):
        HttpPriceSource("http://127.0.0.1:9", timeout_seconds=2).get_prices(DAYS[0])


def test_api_uses_the_prices_service_only_when_prices_url_is_set(monkeypatch):
    import api
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    monkeypatch.setenv("PRICES_URL", "http://prices:8001/")
    source = api.make_agent().tools.price_source
    assert isinstance(source, HttpPriceSource) and source.base_url == "http://prices:8001"
    monkeypatch.delenv("PRICES_URL")
    monkeypatch.setenv("PRICE_SOURCE", "demo")
    assert isinstance(api.make_agent().tools.price_source, DemoPriceSource)   # unchanged behaviour without it
