from app import app


def test_index_route_still_works():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert response.get_json() == {"message": "starter app"}


def test_health_route_exists():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
