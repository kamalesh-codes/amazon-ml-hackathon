import logging, time
from tqdm import tqdm
from .io import count_rows, iter_rows, write_candidates
from .retrieval import Retriever
log=logging.getLogger(__name__)

def generate_candidates(config):
    r=Retriever(config.index_dir,config)
    start=time.time(); done=0
    with open(config.candidates_path,"w",encoding="utf-8") as f:
        f.write("entity_id\tcandidate_ids\n")
        progress=tqdm(total=count_rows(config.source1_path), desc="Generating candidates", unit="rows", disable=not config.show_progress)
        try:
            for row in iter_rows(config.source1_path,chunk_size=config.chunk_size):
                ids=r.retrieve(row); f.write(row["entity_id"]+"\t"+",".join(ids)+"\n"); done+=1
                progress.update(1)
                progress.set_postfix(rate=f"{done/max(time.time()-start,1e-9):.1f} rows/s", refresh=False)
                if done%10000==0: log.info("processed=%d rows/sec=%.1f",done,done/max(time.time()-start,1e-9))
        finally:
            progress.close()

def _split(v): return {x for x in (v or "").split(",") if x}

def evaluate(candidates_path, truth_path, output_path):
    import csv, json, statistics
    truth={r["source1_entity_id"]:_split(r["matched_entity_ids"]) for r in __import__("entity_resolution.io",fromlist=["iter_rows"]).iter_rows(truth_path,columns=("source1_entity_id","matched_entity_ids"))}
    rows=[]
    with open(candidates_path,encoding="utf-8") as f:
        for r in csv.DictReader(f,delimiter="\t"):
            g=truth.get(r["entity_id"],set()); got=_split(r.get("candidate_ids")); tp=len(g&got)
            rows.append((r["entity_id"],g,got,tp))
    counts=[len(x[2]) for x in rows]; recalls=[(x[3]/len(x[1]) if x[1] else 1.0) for x in rows]
    by_matches={}
    for _,g,_,tp in rows: by_matches.setdefault(len(g),[]).append(tp/len(g) if g else 1.0)
    out={"rows":len(rows),"mean_candidates":statistics.mean(counts) if counts else 0,"median_candidates":statistics.median(counts) if counts else 0,
         "p90_candidates":_percentile(counts,.90),"p95_candidates":_percentile(counts,.95),"p99_candidates":_percentile(counts,.99),"max_candidates":max(counts,default=0),
         "zero_retrieved_pct":100*sum(c==0 for c in counts)/max(len(rows),1),"empty_ground_truth_pct":100*sum(not x[1] for x in rows)/max(len(rows),1),
         "mean_recall":statistics.mean(recalls) if recalls else 0,"full_match_coverage_pct":100*sum(x[1]<=x[2] for x in rows)/max(len(rows),1),
         "recall_by_true_match_count":{str(k):statistics.mean(v) for k,v in by_matches.items()}}
    with open(output_path,"w",encoding="utf-8") as f: json.dump(out,f,indent=2)
    return out

def _percentile(values,p):
    if not values:return 0
    a=sorted(values); return a[min(len(a)-1,int(round((len(a)-1)*p)))]
