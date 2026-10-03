from topdon_duo.web import LiveStream, create_app


def test_index_and_status_routes():
    stream = LiveStream()
    client = create_app(stream).test_client()
    assert client.get("/").status_code == 200
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json == {
        "discarded_frames": 0,
        "error": None,
        "frames": 0,
        "rotation": 0,
        "stats": None,
        "warming_up": True,
    }


def test_status_route_reports_capture_error():
    stream = LiveStream()
    stream.error = "camera unavailable"
    client = create_app(stream).test_client()
    assert client.get("/api/status").json["error"] == "camera unavailable"


def test_rotate_route_cycles_clockwise():
    stream = LiveStream()
    client = create_app(stream).test_client()
    assert client.post("/api/rotate").json == {"rotation": 90}
    assert client.post("/api/rotate").json == {"rotation": 180}
