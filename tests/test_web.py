from topdon_duo.web import LiveStream, create_app


def test_index_and_status_routes():
    stream = LiveStream()
    client = create_app(stream).test_client()
    assert client.get("/").status_code == 200
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json == {"error": None, "frames": 0, "stats": None}
