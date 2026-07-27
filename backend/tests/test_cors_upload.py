def test_preview_origin_can_upload_directly_without_via_vercel_proxy():
    from app import app

    origin = "https://hawkeye-labeling-tool-etzjb13em-22s-projects-de7c705f.vercel.app"
    response = app.test_client().options(
        "/api/analyze",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == origin
    assert "POST" in response.headers["Access-Control-Allow-Methods"]


def test_parkicheck_origin_receives_cors_header_on_upload_error():
    from app import app

    origin = "https://finger-tap-fmsen05gj-22s-projects-de7c705f.vercel.app"
    response = app.test_client().post(
        "/api/analyze",
        headers={"Origin": origin},
    )

    assert response.status_code == 400
    assert response.headers["Access-Control-Allow-Origin"] == origin


def test_unrelated_origin_is_not_granted_browser_upload_access():
    from app import app

    response = app.test_client().options(
        "/api/analyze",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert "Access-Control-Allow-Origin" not in response.headers
