def test_create_user_returns_flat_user_payload(client):
    response = client.post("/users", json={"email": "dev@example.com", "display_name": "Dev"})
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"id", "email", "display_name"}
    assert data["email"] == "dev@example.com"


def test_create_user_duplicate_is_rejected(client):
    first = client.post("/users", json={"email": "dup@example.com", "display_name": "One"})
    assert first.status_code == 200
    second = client.post("/users", json={"email": "dup@example.com", "display_name": "Two"})
    assert second.status_code == 400
    assert second.json()["detail"] == "User already exists"
