import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks/comparison"))
import profile_extended_decode as trace  # noqa: E402
import run_extended_comparison as extended  # noqa: E402


class ProfileExtendedDecodeTests(unittest.TestCase):
    def test_trace_is_one_warmup_one_measured_with_short_service_bound(self):
        self.assertEqual((trace.WARMUP_REQUESTS, trace.MEASURED_REQUESTS), (1, 1))
        self.assertEqual(trace.SERVICE_SECONDS, 15)
        self.assertLessEqual(trace.MAX_BUNDLE_SECONDS, 60)

    def test_marks_only_the_first_measured_request(self):
        events = []

        class FakeMarker:
            def push(self, name):
                events.append(("push", name))

            def pop(self):
                events.append(("pop", None))

        calls = []

        def request(label):
            calls.append(label)
            events.append(("request", label))
            return label

        wrapped, call_count, marked_count = trace.mark_first_measured_request(
            request, FakeMarker(), "v200_decode")
        self.assertEqual(wrapped("warmup"), "warmup")
        self.assertEqual(wrapped("measured"), "measured")
        self.assertEqual((call_count(), marked_count()), (2, 1))
        self.assertEqual(events, [
            ("request", "warmup"),
            ("push", "v200_decode"),
            ("request", "measured"),
            ("pop", None),
        ])

    def test_marked_range_closes_when_request_fails(self):
        events = []

        class FakeMarker:
            def push(self, name):
                events.append(("push", name))

            def pop(self):
                events.append(("pop", None))

        def request(label):
            if label == "warmup":
                return label
            raise RuntimeError("request failed")

        wrapped, _, marked_count = trace.mark_first_measured_request(
            request, FakeMarker(), "native_decode")
        self.assertEqual(wrapped("warmup"), "warmup")
        with self.assertRaisesRegex(RuntimeError, "request failed"):
            wrapped("measured")
        self.assertEqual(marked_count(), 1)
        self.assertEqual(events, [("push", "native_decode"), ("pop", None)])

    def test_case_selection_and_build_dir_override_are_narrow(self):
        self.assertEqual(trace.selected_case("v200_extended")["id"], "v200_extended")
        with self.assertRaises(trace.TraceCaseError):
            trace.selected_case("v200_anchor")
        for name in ("build-multipair", "build-perf"):
            selected = extended._selected_build_dir(name)
            self.assertEqual(selected, REPO_ROOT / name)
            self.assertEqual(extended._selected_dso_paths(selected)["v200_operators"],
                             REPO_ROOT / name / "src/operators/libv200_operators.so")
        with self.assertRaises(extended.ExtendedComparisonError):
            extended._selected_build_dir("build")

    def test_local_hbm_override_is_scoped_and_restored(self):
        key = extended.PREFER_LOCAL_HBM_ENV
        with patch.dict(os.environ, {key: "prior"}, clear=False):
            with extended._local_hbm_environment(True):
                self.assertEqual(os.environ[key], "1")
            self.assertEqual(os.environ[key], "prior")
            with extended._local_hbm_environment(False):
                self.assertEqual(os.environ[key], "0")
            self.assertEqual(os.environ[key], "prior")


if __name__ == "__main__":
    unittest.main()
