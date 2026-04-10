from db_utils import get_user


class RecordingCursor:
    def __init__(self):
        self.calls = []

    def execute(self, query, params=None):
        self.calls.append((query, params))

    def fetchone(self):
        return {"name": "alice"}


def test_parameterized_query_is_used():
    cursor = RecordingCursor()
    get_user(cursor, "alice")
    query, params = cursor.calls[-1]
    assert "?" in query
    assert params == ("alice",)


def test_username_is_not_interpolated_into_query():
    cursor = RecordingCursor()
    malicious = "' OR 1=1 --"
    get_user(cursor, malicious)
    query, params = cursor.calls[-1]
    assert malicious not in query
    assert params == (malicious,)
