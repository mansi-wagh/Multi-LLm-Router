import unittest
from unittest.mock import patch

import httpx

from app import main
from app.config import CONFIG
from app.main import ChatMessage, ChatRequest
from app.label_jev import Labels, label_with_jev
from app.route import decide, label_with_llm, resolve_route, select
from app.rules import looks_private, route_by_rules


class RoutingTests(unittest.TestCase):
    def test_private_patterns_are_detected_by_rules(self):
        for text in (
            "Email me at person@example.com",
            "Call me at +1 (415) 555-0199",
            "Use account 123456789012",
        ):
            with self.subTest(text=text):
                self.assertTrue(looks_private(text))

    def test_greeting_is_a_rules_chitchat_label(self):
        self.assertEqual(route_by_rules("Hi there!"), "chitchat")
        route = resolve_route("Hi there!")
        self.assertEqual(
            (route.lane, route.rule, route.label_source),
            ("chitchat", "greeting", "rules"),
        )

    def test_keyword_labels_and_default_lane_are_configured(self):
        self.assertEqual(route_by_rules("Please debug this Python function"), "code")
        self.assertEqual(decide("unrecognized"), CONFIG["labels"]["general"])

    def test_labeler_uses_local_model_from_root_config(self):
        response = httpx.Response(
            200,
            json={"message": {"content": "reasoning"}},
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )
        with patch("app.route.httpx.post", return_value=response) as post:
            self.assertEqual(label_with_llm("Compare these approaches"), "reasoning")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(
            payload["model"],
            CONFIG["models"][CONFIG["labeler"]["model"]]["model"],
        )
        self.assertEqual(post.call_args.args[0], "http://localhost:11434/api/chat")

    def test_private_route_never_calls_labeler(self):
        with patch("app.route.label_with_llm", side_effect=AssertionError):
            route = resolve_route("my number is 123456789012", private=True)
        self.assertEqual(
            (route.label, route.lane, route.label_source),
            ("private", "private", "rules"),
        )

    def test_private_selection_filters_cloud_in_every_lane(self):
        original_lane = CONFIG["lanes"]["private"]
        original_enabled = CONFIG["models"]["local"]["enabled"]
        try:
            CONFIG["lanes"]["private"] = ["groq", "local"]
            CONFIG["models"]["local"]["enabled"] = False
            self.assertIsNone(select("private", private=False))
            CONFIG["models"]["local"]["enabled"] = True
            self.assertEqual(select("private", private=True), "local")
        finally:
            CONFIG["lanes"]["private"] = original_lane
            CONFIG["models"]["local"]["enabled"] = original_enabled

    def test_private_route_uses_rules_even_if_pattern_is_in_prior_user_message(self):
        route = resolve_route("What is the weather?", private=True)
        self.assertEqual(route.label_source, "rules")
        self.assertEqual(route.lane, CONFIG["labels"][CONFIG["private_label"]])

    def test_jev_success_parses_labels_and_sends_one_configured_request(self):
        response = httpx.Response(
            200,
            json={
                "answers": {
                    "task": {"choice": "code"},
                    "difficulty": {"score": 1.0, "confidence": 0.8},
                    "private": {"noul": 0.7},
                    "needs_web": {"noul": 0.6},
                }
            },
            request=httpx.Request(
                "POST", "https://opencode.ai/zen/v1/systemone"
            ),
        )
        with patch("app.label_jev.get_api_key", return_value="test-key"):
            with patch("app.label_jev.httpx.post", return_value=response) as post:
                labels = label_with_jev("Please classify this")

        self.assertEqual(
            labels,
            Labels("code", "easy", True, True),
        )
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["timeout"], 3)
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key"
        )
        self.assertEqual(
            post.call_args.kwargs["json"]["state"], "Please classify this"
        )

    def test_jev_calls_without_authorization_when_key_is_missing(self):
        response = httpx.Response(
            200,
            json={
                "answers": {
                    "task": {"choice": "code"},
                    "difficulty": {"score": 2, "confidence": 1},
                    "private": {"noul": 0},
                    "needs_web": {"noul": 0},
                }
            },
            request=httpx.Request(
                "POST", "https://opencode.ai/zen/v1/systemone"
            ),
        )
        with patch("app.label_jev.get_api_key", return_value=None):
            with patch("app.label_jev.httpx.post", return_value=response) as post:
                self.assertIsNotNone(label_with_jev("Please classify this"))
        self.assertEqual(post.call_args.kwargs["headers"], {})

    def test_jev_route_success_uses_jev_labels(self):
        with patch(
            "app.route.label_with_jev",
            return_value=Labels("code", "hard", False, False),
        ):
            route = resolve_route("Please classify this", router_mode="jev")
        self.assertEqual((route.lane, route.label_source), ("code", "jev"))

    def test_jev_private_label_selects_local_lane(self):
        with patch(
            "app.route.label_with_jev",
            return_value=Labels("code", "moderate", True, False),
        ):
            route = resolve_route("Please classify this", router_mode="jev")
        self.assertTrue(route.private)
        self.assertEqual(route.lane, "private")
        self.assertEqual(select(route.lane, route.private), "local")

    def test_jev_failure_falls_back_to_local_labeller(self):
        with patch("app.route.label_with_jev", return_value=None):
            with patch("app.route.label_with_llm", return_value="code") as local:
                route = resolve_route("Please classify this", router_mode="jev")
        local.assert_called_once_with("Please classify this")
        self.assertEqual(
            (route.lane, route.label_source),
            ("code", "local-llm (jev failed)"),
        )

    def test_private_text_never_calls_jev_http_endpoint(self):
        with patch("app.label_jev.httpx.post") as post:
            route = resolve_route("Contact test@example.com", router_mode="jev")
        post.assert_not_called()
        self.assertEqual((route.lane, route.label_source), ("private", "rules"))

    def test_password_keyword_selects_private_lane_and_local_model(self):
        with patch("app.route.label_with_jev") as jev:
            with patch("app.route.label_with_llm") as local:
                route = resolve_route("I forgot my password", router_mode="jev")
        jev.assert_not_called()
        local.assert_not_called()
        self.assertEqual(route.lane, "private")
        self.assertEqual(select(route.lane), "local")

    def test_router_mode_switch_updates_mode_and_local_labeller(self):
        from fastapi.testclient import TestClient

        original_mode = main.ROUTER_MODE
        original_local_llm = main.LOCAL_LLM_ENABLED
        try:
            response = TestClient(main.app).post(
                "/router-mode",
                json={"mode": "jev", "local_llm": False},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(), {"mode": "jev", "local_llm": False}
            )
            self.assertEqual(main.ROUTER_MODE, "jev")
            self.assertFalse(main.LOCAL_LLM_ENABLED)
        finally:
            main.ROUTER_MODE = original_mode
            main.LOCAL_LLM_ENABLED = original_local_llm


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_conversation_flag_keeps_later_request_local(self):
        original_count = main.REQUEST_COUNT
        original_model_counts = main.MODEL_REQUEST_COUNTS.copy()
        attempted = []

        async def fake_stream(model_name, *_args):
            attempted.append(model_name)
            yield "local response"

        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Tell me about weather")],
            private_conversation=True,
        )
        try:
            with patch("app.main.stream_model", side_effect=fake_stream):
                events = [event async for event in main._chat_events(request)]
        finally:
            main.REQUEST_COUNT = original_count
            main.MODEL_REQUEST_COUNTS.update(original_model_counts)

        self.assertEqual(attempted, ["local"])
        self.assertIn('"private": true', events[0])
        self.assertIn('"model": "local"', events[0])

    async def test_pre_token_failure_tries_next_model_and_emits_route_first(self):
        original_enabled = {
            name: model["enabled"] for name, model in CONFIG["models"].items()
        }
        original_counts = main.MODEL_REQUEST_COUNTS.copy()
        original_request_count = main.REQUEST_COUNT

        try:
            CONFIG["models"]["groq"]["enabled"] = True
            CONFIG["models"]["gemini"]["enabled"] = False
            CONFIG["models"]["local"]["enabled"] = True
            main.REQUEST_COUNT = 0
            for name in main.MODEL_REQUEST_COUNTS:
                main.MODEL_REQUEST_COUNTS[name] = 0

            async def fake_stream(model_name, *_args):
                if model_name == "groq":
                    raise httpx.ConnectError("offline")
                yield "answer"

            request = ChatRequest(
                messages=[ChatMessage(role="user", content="Debug this code")]
            )
            with patch("app.main.stream_model", side_effect=fake_stream):
                events = [event async for event in main._chat_events(request)]

            self.assertTrue(events[0].startswith("event: route"))
            route_events = [event for event in events if event.startswith("event: route")]
            self.assertIn('"model": "groq"', route_events[0])
            self.assertIn('"model": "local"', route_events[1])
            self.assertTrue(any(event.startswith("event: token") for event in events))
            self.assertTrue(events[-1].startswith("event: done"))
        finally:
            for name, enabled in original_enabled.items():
                CONFIG["models"][name]["enabled"] = enabled
            for name, count in original_counts.items():
                main.MODEL_REQUEST_COUNTS[name] = count
            main.REQUEST_COUNT = original_request_count

    async def test_private_stream_never_attempts_cloud_fallback(self):
        attempted = []
        original_request_count = main.REQUEST_COUNT
        original_counts = main.MODEL_REQUEST_COUNTS.copy()

        async def fail_local(model_name, *_args):
            attempted.append(model_name)
            raise httpx.ConnectError("offline")
            yield ""

        request = ChatRequest(
            messages=[
                ChatMessage(role="user", content="Email me at person@example.com"),
                ChatMessage(role="assistant", content="Understood."),
                ChatMessage(role="user", content="What is the weather?"),
            ]
        )
        try:
            with patch("app.route.label_with_llm", side_effect=AssertionError):
                with patch("app.main.stream_model", side_effect=fail_local):
                    events = [event async for event in main._chat_events(request)]
        finally:
            main.REQUEST_COUNT = original_request_count
            for name, count in original_counts.items():
                main.MODEL_REQUEST_COUNTS[name] = count
        self.assertEqual(attempted, ["local"])
        self.assertTrue(any("All enabled local models failed" in event for event in events))

    async def test_private_request_without_enabled_local_returns_clear_error(self):
        original_enabled = CONFIG["models"]["local"]["enabled"]
        original_request_count = main.REQUEST_COUNT
        original_counts = main.MODEL_REQUEST_COUNTS.copy()
        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Email me at person@example.com")]
        )
        try:
            CONFIG["models"]["local"]["enabled"] = False
            with patch("app.route.label_with_llm", side_effect=AssertionError):
                events = [event async for event in main._chat_events(request)]
        finally:
            CONFIG["models"]["local"]["enabled"] = original_enabled
            main.REQUEST_COUNT = original_request_count
            for name, count in original_counts.items():
                main.MODEL_REQUEST_COUNTS[name] = count
        self.assertTrue(events[0].startswith("event: route"))
        self.assertTrue(
            any("No enabled local model is available for this private request" in event
                for event in events)
        )


if __name__ == "__main__":
    unittest.main()
