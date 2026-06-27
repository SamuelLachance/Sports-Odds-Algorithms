"""Sports database normalization tests."""

from web.sports_db.normalize import parse_news, parse_standings, parse_team_statistics


def test_parse_standings_groups() -> None:
    payload = {
        "children": [
            {
                "name": "East",
                "standings": {
                    "entries": [
                        {
                            "team": {"abbreviation": "BOS", "displayName": "Boston Celtics"},
                            "stats": [
                                {"name": "wins", "value": 50.0, "displayValue": "50"},
                                {"name": "losses", "value": 20.0, "displayValue": "20"},
                                {"name": "winPercent", "value": 0.714, "displayValue": ".714"},
                                {"name": "rank", "value": 2.0, "displayValue": "2"},
                            ],
                        }
                    ]
                },
            }
        ]
    }
    parsed = parse_standings(payload)
    assert len(parsed["groups"]) == 1
    assert parsed["teams"][0]["abbr"] == "bos"
    assert parsed["teams"][0]["wins"] == 50.0


def test_parse_news_headlines() -> None:
    payload = {
        "articles": [
            {
                "id": "1",
                "headline": "Test headline",
                "description": "Body",
                "published": "2026-06-27T12:00:00Z",
                "links": {"web": {"href": "https://example.com"}},
            }
        ]
    }
    rows = parse_news(payload)
    assert len(rows) == 1
    assert rows[0]["headline"] == "Test headline"


def test_parse_team_statistics_categories() -> None:
    payload = {
        "results": {
            "stats": {
                "categories": [
                    {
                        "name": "general",
                        "displayName": "General",
                        "stats": [
                            {"name": "avgPoints", "displayName": "Points", "value": 110.0, "displayValue": "110.0"}
                        ],
                    }
                ]
            }
        }
    }
    parsed = parse_team_statistics(payload)
    assert parsed["categories"][0]["name"] == "General"
    assert parsed["categories"][0]["stats"][0]["display"] == "110.0"
