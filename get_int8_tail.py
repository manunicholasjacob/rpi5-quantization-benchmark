import csv, os
path = os.path.expanduser("~/paper_results/static_benchmark_results.csv")
rows = list(csv.DictReader(open(path)))
int8 = [r for r in rows if r["precision"] == "int8"]
if not int8:
    print("NO INT8 ROWS in static_benchmark_results.csv")
    print("All precisions:", set(r["precision"] for r in rows))
else:
    for r in int8:
        ratio = float(r["p90_latency_ms"]) / float(r["p50_latency_ms"])
        print(f"{r['model']} b={r['batch']} t={r['threads']} {r['precision']}: p50={float(r['p50_latency_ms']):.3f} p90={float(r['p90_latency_ms']):.3f} ratio={ratio:.3f}")
