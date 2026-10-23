from hypothesis import given
from hypothesis import strategies as st

from eroll.extraction.records import FIELD_KEYS
from eroll.service.api import changed_fields, key_matches
from eroll.service.workers import (
    IDLE_TERMINATE_MINUTES,
    idle_since,
    should_terminate,
)

KEY = "s3cret-key"
IDLE_LIMIT_SECONDS = IDLE_TERMINATE_MINUTES * 60


def test_key_matches_rejects_missing_and_empty() -> None:
    assert not key_matches(None, KEY)
    assert not key_matches("", KEY)
    # An unset expected key must never match, even an empty header.
    assert not key_matches("", "")


def test_key_matches_compares_key() -> None:
    assert not key_matches("wrong-key", KEY)
    assert key_matches(KEY, KEY)


def test_key_matches_non_ascii_does_not_raise() -> None:
    # compare_digest raises TypeError on non-ASCII str; key_matches encodes first.
    assert not key_matches("clé-ünïcode", KEY)
    assert key_matches("clé", "clé")


# Only changed fields become corrections.
@given(st.data())
def test_changed_fields_keeps_only_stripped_differences(data) -> None:
    prefilled = data.draw(st.fixed_dictionaries({key: st.text(max_size=12) for key in FIELD_KEYS}))
    padding = st.text(alphabet=" \t\n", max_size=3)
    submitted = {}
    for key in FIELD_KEYS:
        if data.draw(st.booleans()):
            # The pre-filled value sent back, possibly with whitespace around it.
            submitted[key] = data.draw(padding) + prefilled[key] + data.draw(padding)
        else:
            submitted[key] = data.draw(st.text(max_size=12))

    expected = {
        key: value.strip() for key, value in submitted.items() if value.strip() != prefilled[key]
    }
    assert changed_fields(prefilled, submitted) == expected


def test_changed_fields_ignores_surrounding_whitespace() -> None:
    prefilled = dict.fromkeys(FIELD_KEYS, "x")
    submitted = dict.fromkeys(FIELD_KEYS, "  x\t")
    submitted["age"] = " 41 "
    assert changed_fields(prefilled, submitted) == {"age": "41"}


# The idle timer fires at the first observation 15+ minutes after the last busy one.
@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=2 * IDLE_LIMIT_SECONDS),
            st.sampled_from(["busy", "idle", "unreachable"]),
        ),
        max_size=40,
    )
)
def test_idle_timer_terminates_at_first_long_idle_observation(observations) -> None:
    process_start = 0
    now = process_start
    last_busy_end = None  # what the worker loop tracks
    last_busy = process_start  # the test's own reckoning: max(start, last busy)
    first_requested = None
    first_expected = None
    for index, (gap, kind) in enumerate(observations):
        now += gap
        busy = kind == "busy"
        # Idle and unreachable observations never move the start of the idle span.
        assert idle_since(process_start, last_busy_end) == last_busy

        requested = should_terminate(idle_since(process_start, last_busy_end), now, busy)
        if busy:
            assert not requested
        if requested and first_requested is None:
            first_requested = index
        if not busy and now - last_busy >= IDLE_LIMIT_SECONDS and first_expected is None:
            first_expected = index

        if busy:
            last_busy_end = now
            last_busy = now
    assert first_requested == first_expected


def test_idle_timer_boundary() -> None:
    assert should_terminate(0, IDLE_LIMIT_SECONDS, busy=False)
    assert not should_terminate(0, IDLE_LIMIT_SECONDS - 1, busy=False)
    assert not should_terminate(0, 10 * IDLE_LIMIT_SECONDS, busy=True)
    # A job that ended after process start restarts the span from its end.
    assert idle_since(100, 500) == 500
    assert not should_terminate(idle_since(100, 500), 500 + IDLE_LIMIT_SECONDS - 1, busy=False)
