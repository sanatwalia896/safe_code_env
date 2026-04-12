def test_health_endpoint_shape(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "safe-code-api"
    assert data["sqlite_ready"] is True
