"""The OpenAI-compatible server, with a fake backend and a one-letter-per-token tokenizer (no models, no network)."""
import http.client
import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from commonnet.mesh.placement import NoRoute
from commonnet.net.client import Unreachable
from commonnet.serve import ApiError, TextStream, _WorkerStream, make_server, parse_request


class LetterTokenizer:
    def decode(self, ids):
        return "".join("�" if i == 99 else chr(96 + i) for i in ids)


class FakeStream:
    def __init__(self, ids, fail_after=None, error=None, delay=0.0):
        self.ids, self.fail_after, self.error, self.closed, self.delay = list(ids), fail_after, error, False, delay

    def __iter__(self):
        return self

    def __next__(self):
        time.sleep(self.delay)
        if self.fail_after is not None and self.fail_after == 0:
            raise self.error
        if not self.ids:
            raise StopIteration
        if self.fail_after is not None:
            self.fail_after -= 1
        return self.ids.pop(0)

    def close(self):
        self.closed = True


class FakeBackend:
    model_name = "Tiny"
    tokenizer = LetterTokenizer()

    def __init__(self):
        self.script = [1, 2, 3]
        self.fail_after, self.error, self.delay = None, None, 0.0
        self.last_args, self.last_stream = None, None

    def encode_chat(self, messages, template_kwargs):
        return list(range(sum(len(m["content"]) for m in messages)))

    def stream(self, prompt_ids, max_tokens, temperature, top_p, seed):
        self.last_args = (len(prompt_ids), max_tokens, temperature, top_p, seed)
        self.last_stream = FakeStream(self.script[:max_tokens], self.fail_after, self.error, self.delay)
        return self.last_stream

    def health(self):
        return {"peers": 3}


class ServeTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.server = make_server(self.backend, 0, max_context=100)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (self.server.shutdown(), self.server.server_close()))

    def request(self, method, path, body=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        conn.request(method, path, body=data, headers={"Content-Type": "application/json"} if data else {})
        resp = conn.getresponse()
        payload = resp.read()
        conn.close()
        return resp.status, resp.getheader("Content-Type"), payload

    def chat(self, **extra):
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
        body.update(extra)
        return body

    def sse(self, payload):
        events = [e for e in payload.decode().split("\n\n") if e]
        return [e[len("data: "):] for e in events]

    def test_models_and_health(self):
        status, _, payload = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["data"][0]["id"], "Tiny")
        self.assertEqual(json.loads(self.request("GET", "/health")[2]), {"peers": 3})
        self.assertEqual(self.request("GET", "/nope")[0], 404)

    def test_plain_completion(self):
        status, ctype, payload = self.request("POST", "/v1/chat/completions", self.chat(max_tokens=10))
        out = json.loads(payload)
        self.assertEqual((status, ctype), (200, "application/json"))
        self.assertEqual(out["choices"][0]["message"], {"role": "assistant", "content": "abc"})
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")  # three tokens, 10 allowed: it ended itself
        self.assertEqual(out["usage"], {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})
        self.assertEqual(out["object"], "chat.completion")

    def test_length_finish_reason(self):
        out = json.loads(self.request("POST", "/v1/chat/completions", self.chat(max_tokens=2))[2])
        self.assertEqual(out["choices"][0]["message"]["content"], "ab")
        self.assertEqual(out["choices"][0]["finish_reason"], "length")

    def test_sampling_parameters_reach_the_backend(self):
        self.request("POST", "/v1/chat/completions", self.chat(max_tokens=5, temperature=0.2, top_p=0.9, seed=7))
        self.assertEqual(self.backend.last_args, (2, 5, 0.2, 0.9, 7))

    def test_streaming_completion(self):
        status, ctype, payload = self.request("POST", "/v1/chat/completions", self.chat(stream=True, max_tokens=10))
        self.assertEqual((status, ctype), (200, "text/event-stream"))
        events = self.sse(payload)
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(e) for e in events[:-1]]
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant", "content": ""})
        self.assertEqual("".join(c["choices"][0]["delta"].get("content", "") for c in chunks[1:]), "abc")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertTrue(all(c["object"] == "chat.completion.chunk" for c in chunks))

    def test_validation_errors_are_openai_shaped(self):
        for body, needle in [({"messages": []}, "non-empty"),
                             (self.chat(temperature=9), "temperature"),
                             (self.chat(max_tokens=0), "max_tokens"),
                             ({"messages": [{"role": "robot", "content": "x"}]}, "role")]:
            with self.subTest(body=body):
                status, _, payload = self.request("POST", "/v1/chat/completions", body)
                self.assertEqual(status, 400)
                self.assertIn(needle, json.loads(payload)["error"]["message"])
        self.assertEqual(self.request("POST", "/v1/chat/completions", raw=b"{nope")[0], 400)

    def test_context_limit(self):
        status, _, payload = self.request("POST", "/v1/chat/completions",
                                          self.chat(messages=[{"role": "user", "content": "x" * 95}]))
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(payload)["error"]["code"], "context_length_exceeded")

    def test_no_route_is_503_before_any_stream_headers(self):
        self.backend.fail_after, self.backend.error = 0, NoRoute(18)
        for stream in (False, True):
            with self.subTest(stream=stream):
                status, ctype, payload = self.request("POST", "/v1/chat/completions", self.chat(stream=stream))
                self.assertEqual((status, ctype), (503, "application/json"))
                self.assertEqual(json.loads(payload)["error"]["code"], "no_route")

    def test_a_peer_failing_mid_stream_ends_with_an_error_event_and_done(self):
        self.backend.fail_after, self.backend.error = 2, Unreachable("peer gone")
        status, ctype, payload = self.request("POST", "/v1/chat/completions", self.chat(stream=True))
        self.assertEqual((status, ctype), (200, "text/event-stream"))
        events = self.sse(payload)
        self.assertEqual(events[-1], "[DONE]")
        self.assertEqual(json.loads(events[-2])["error"]["code"], "peer_failed")
        text = "".join(json.loads(e)["choices"][0]["delta"].get("content", "") for e in events[:-2])
        self.assertEqual(text, "ab")  # what was produced before the failure was delivered

    def test_a_peer_failing_in_a_plain_request_is_502(self):
        self.backend.fail_after, self.backend.error = 1, Unreachable("peer gone")
        status, _, payload = self.request("POST", "/v1/chat/completions", self.chat())
        self.assertEqual(status, 502)

    def test_a_caller_that_hangs_up_closes_the_stream(self):
        self.backend.script = list(range(1, 90))
        self.backend.delay = 0.05
        import socket
        sock = socket.create_connection(("127.0.0.1", self.port))
        body = json.dumps(self.chat(stream=True, max_tokens=80)).encode()
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        sock.recv(200)
        sock.close()
        deadline = time.time() + 5
        while time.time() < deadline and not self.backend.last_stream.closed:
            time.sleep(0.05)
        self.assertTrue(self.backend.last_stream.closed)


class TextTests(unittest.TestCase):
    def test_an_unfinished_character_is_held_back_until_it_completes(self):
        text = TextStream(LetterTokenizer())
        self.assertEqual([text.push(1), text.push(99), text.push(2)][0], "a")
        text = TextStream(LetterTokenizer())
        text.push(1)
        self.assertEqual(text.push(99), "")  # replacement character: wait
        self.assertEqual(text.flush(), "�")

    def test_thinking_is_off_by_default_and_a_request_can_turn_it_on(self):
        msgs = [{"role": "user", "content": "hi"}]
        self.assertEqual(parse_request({"messages": msgs})["template_kwargs"], {"enable_thinking": False})
        on = parse_request({"messages": msgs, "chat_template_kwargs": {"enable_thinking": True, "x": 1}})
        self.assertEqual(on["template_kwargs"], {"enable_thinking": True, "x": 1})

    def test_parse_request_defaults(self):
        req = parse_request({"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]})
        self.assertEqual(req["messages"][0]["content"], "hi")
        self.assertEqual((req["max_tokens"], req["stream"], req["seed"]), (256, False, None))
        with self.assertRaises(ApiError):
            parse_request({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]})


class WorkerStreamTests(unittest.TestCase):
    def setUp(self):
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(self.worker.shutdown)

    def test_items_cross_threads_in_order_and_run_on_one_thread(self):
        names = set()

        def gen(n):
            for i in range(n):
                names.add(threading.current_thread().name)
                yield i

        self.assertEqual(list(_WorkerStream(self.worker, gen, (4,))), [0, 1, 2, 3])
        self.assertEqual(len(names), 1)

    def test_an_error_is_raised_to_the_reader_after_the_items_before_it(self):
        def gen():
            yield 1
            raise Unreachable("gone")

        stream = _WorkerStream(self.worker, gen, ())
        self.assertEqual(next(stream), 1)
        with self.assertRaises(Unreachable):
            next(stream)

    def test_close_stops_the_generator_and_closes_it(self):
        closed = threading.Event()

        def gen():
            try:
                for i in range(10_000):
                    time.sleep(0.001)
                    yield i
            finally:
                closed.set()

        stream = _WorkerStream(self.worker, gen, ())
        next(stream)
        stream.close()
        self.assertTrue(closed.wait(5))


if __name__ == "__main__":
    unittest.main()
