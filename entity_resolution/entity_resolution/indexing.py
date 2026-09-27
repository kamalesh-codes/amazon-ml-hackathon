from pathlib import Path
import json
import numpy as np
from .metadata import save_metadata, load_metadata

try:
    import faiss
except ImportError:
    faiss = None

class VectorIndex:
    def __init__(self, dim, nlist=4096, pq_m=64, pq_nbits=8, nprobe=32, exact_threshold=50000):
        self.dim, self.nlist, self.pq_m, self.pq_nbits, self.nprobe, self.exact_threshold = dim,nlist,pq_m,pq_nbits,nprobe,exact_threshold
        self.vectors=[]; self.index=None; self.metadata=[]
    def add(self, vectors, metadata):
        x=np.asarray(vectors,dtype="float32"); x /= np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
        self.vectors.append(x); self.metadata.extend(metadata)
    def _build(self):
        x=np.concatenate(self.vectors,axis=0) if self.vectors else np.empty((0,self.dim),dtype="float32")
        if faiss is None:
            self.index=x
        elif len(x) < self.exact_threshold:
            self.index=faiss.IndexFlatIP(self.dim); self.index.add(x)
        else:
            nlist=min(self.nlist,max(1,len(x)//1000)); m=min(self.pq_m,self.dim)
            quant=faiss.IndexFlatIP(self.dim)
            self.index=faiss.IndexIVFPQ(quant,self.dim,nlist,m,self.pq_nbits,faiss.METRIC_INNER_PRODUCT)
            sample=x[:min(len(x),max(10000,nlist*40))]; self.index.train(sample); self.index.add(x); self.index.nprobe=self.nprobe
        self.vectors=[]
    def search(self, queries, k):
        if self.index is None: self._build()
        q=np.asarray(queries,dtype="float32"); q /= np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12)
        if faiss is not None and hasattr(self.index,"search"): return self.index.search(q,k)
        sims=q @ self.index.T; kk=min(k,sims.shape[1]); ids=np.argsort(-sims,axis=1)[:,:kk]; return np.take_along_axis(sims,ids,1),ids
    def save(self, directory, field):
        self._build(); d=Path(directory); d.mkdir(parents=True,exist_ok=True)
        if faiss is not None and hasattr(self.index,"ntotal"): faiss.write_index(self.index,str(d/f"{field}.index"))
        else: np.save(d/f"{field}.npy",self.index)
        save_metadata(d/f"{field}.metadata.jsonl",self.metadata)
        (d/f"{field}.config.json").write_text(json.dumps({"dim":self.dim,"backend":"faiss" if faiss else "numpy"}),encoding="utf-8")
    @classmethod
    def load(cls,directory,field):
        d=Path(directory); cfg=json.loads((d/f"{field}.config.json").read_text()); obj=cls(cfg["dim"])
        if faiss is not None and (d/f"{field}.index").exists():
            obj.index=faiss.read_index(str(d/f"{field}.index"))
            if hasattr(obj.index,"nprobe"):
                obj.index.nprobe=32
        else: obj.index=np.load(d/f"{field}.npy")
        obj.metadata=load_metadata(d/f"{field}.metadata.jsonl"); return obj
