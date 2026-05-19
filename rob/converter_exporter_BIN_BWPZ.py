import lib_webp as lib
import time
import json
import sys
import os


# =============================================================================

ret = lib.prepare_structure_from_json("config.json")
print(json.dumps(ret, indent=2))

i=0
sm = len(ret["in"])
for s_in,s_out in zip(ret["in"],ret["out"]):
    i += 1
    print(f"\n[SESSION {i}]")
    t = time.perf_counter()
    lib.convert_session(i,sm, s_in, s_out,quality=ret["quality"],method=ret["method"],max_workers=ret["workers"]) # finalnie quality=85,method=6,max_workers=None
    h, m, s = lib.seconds_to_hms(time.perf_counter()-t)
    print(f"\nCzas konwersji całej sesji: {h}h {m}m {s:.1f}s")
    print("-" * 50)

lib.wait_for_escape()

