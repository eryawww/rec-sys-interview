from fastapi.testclient import TestClient

from recsys.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_users_listing_shape():
    body = client.get("/users").json()
    assert len(body) == 220
    assert set(body[0]) == {"user_id", "name", "region"}


def test_popular_returns_k_items():
    body = client.get("/popular?k=5").json()
    assert body["k"] == 5
    assert len(body["items"]) == 5
    assert body["algorithm"] in {"popularity", "item_knn"}
    assert set(body["items"][0]) == {
        "item_id",
        "title",
        "content_type",
        "genre",
        "score",
        "reason",
    }


def test_popular_is_the_same_for_every_request():
    first = client.get("/popular?k=10").json()["items"]
    second = client.get("/popular?k=10").json()["items"]
    assert first == second


def test_recommendations_for_known_user():
    body = client.get("/recommendations?user_id=u1&k=5").json()
    assert body["user_id"] == "u1"
    assert body["fallback_used"] is False
    assert len(body["items"]) == 5
    assert all(item["reason"] for item in body["items"])


def test_unknown_user_returns_200_with_fallback_flag():
    response = client.get("/recommendations?user_id=u999&k=5")
    assert response.status_code == 200
    body = response.json()
    assert body["fallback_used"] is True
    assert body["items"] == client.get("/popular?k=5").json()["items"]


def test_k_out_of_range_is_rejected():
    assert client.get("/popular?k=0").status_code == 422
    assert client.get("/popular?k=51").status_code == 422
    assert client.get("/recommendations?user_id=u1&k=0").status_code == 422


def test_missing_user_id_is_rejected():
    assert client.get("/recommendations").status_code == 422


def test_root_serves_the_frontend():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
