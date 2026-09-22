"""Execute only within the isolated evaluation container; data arrives on stdin."""

import json
import sys

from evalplus.eval import untrusted_check
from evalplus.gen.util import trusted_exec


def main():
    payload = json.load(sys.stdin)
    p = payload["problem"]
    source = payload["solution"]
    result = {}
    for name in ["base", "plus"]:
        inputs = p[name + "_input"]
        expected, times = trusted_exec(
            p["prompt"] + p["canonical_solution"], inputs, p["entry_point"], record_time=True
        )
        status, checks = untrusted_check(
            "humaneval",
            source,
            inputs,
            p["entry_point"],
            expected,
            p["atol"],
            times,
            fast_check=True,
        )
        result[name] = {"status": status, "n_tests": len(inputs), "n_checked": len(checks)}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
