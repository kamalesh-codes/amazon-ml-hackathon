import csv, json
from pathlib import Path
import numpy as np
import pytest
from entity_resolution.config import PipelineConfig
from entity_resolution.io import read_tsv, write_candidates
from entity_resolution.preprocessing import normalize_name, normalize_address
from entity_resolution.country import normalize_country, route_country
from entity_resolution.indexing import VectorIndex
from entity_resolution.candidate_generation import evaluate
from entity_resolution.retrieval import Retriever
from entity_resolution.ranker import pair_features, make_training_data

def test_reader_chunks_and_missing(tmp_path):
 p=tmp_path/'x.tsv'; p.write_text('entity_id\tbusiness_name\tbusiness_address\tcountry\n1\t café  ltd \t\tIN\n2\t東京\t1-2\t\n',encoding='utf8')
 chunks=list(read_tsv(str(p),chunk_size=1)); assert [len(x) for x in chunks]==[1,1]; assert chunks[0][0]['business_address']==''

def test_multilingual_normalization():
 assert normalize_name('  Café　LTD！ ') == 'café ltd!'
 assert normalize_address('१२३  Main-Street') == '१२३ main-street'
 assert normalize_name('東京 株式会社') == '東京 株式会社'

def test_country_routing():
 assert normalize_country(' united states ')=='US'; assert normalize_country('IN')=='IN'; assert route_country('', ['IN','US'],'all_countries')==['IN','US']

def test_index_save_load_search(tmp_path):
 ix=VectorIndex(3,exact_threshold=100); ix.add(np.array([[1,0,0],[0,1,0]],dtype='float32'),[{'entity_id':'a'},{'entity_id':'b'}]); ix.save(tmp_path,'name'); loaded=VectorIndex.load(tmp_path,'name'); _, ids=loaded.search(np.array([[.99,.01,0]],dtype='float32'),1); assert loaded.metadata[int(ids[0,0])]['entity_id']=='a'

def test_retrieval_unions_gpu_shards(tmp_path):
 c=PipelineConfig(index_dir=str(tmp_path),backend='hash',embedding_dim=16,top_k=1)
 for gpu, entity in [(0,'r0'),(1,'r1')]:
  for field in ('name','address'):
   ix=VectorIndex(16,exact_threshold=100); ix.add(np.ones((1,16),dtype='float32'),[{'entity_id':entity}]); ix.save(tmp_path/f'shard-gpu{gpu}'/'IN',field)
 r=Retriever(str(tmp_path),c); ids=r.retrieve({'entity_id':'q','business_name':'Acme','business_address':'1 Main','country':'IN'})
 assert set(ids)=={'r0','r1'}

def test_retriever_rejects_missing_indexes(tmp_path):
 with pytest.raises(FileNotFoundError): Retriever(str(tmp_path), PipelineConfig(backend='hash', embedding_dim=16))

def test_ranker_features_and_negative_sampling(tmp_path):
 cand=tmp_path/'c.tsv'; truth=tmp_path/'g.tsv'
 cand.write_text('entity_id\tcandidate_ids\nq1\tr1,r2,r3\n',encoding='utf8')
 truth.write_text('source1_entity_id\tmatched_entity_ids\nq1\tr1\n',encoding='utf8')
 q={'q1':{'entity_id':'q1','business_name':'Acme Ltd','business_address':'1 Main','country':'IN'}}
 refs={f'r{i}':{'entity_id':f'r{i}','business_name':('Acme Ltd' if i==1 else 'Other'),'business_address':'1 Main','country':'IN'} for i in range(1,4)}
 x,y,groups=make_training_data(str(cand),q,refs,{'q1':{'r1'}},max_negatives=1)
 assert x.shape[1]==13 and groups==[2] and list(y)==[1,0]
 assert pair_features(q['q1'],refs['r1'],1)[1]==1.0

def test_evaluation_metrics(tmp_path):
 cand=tmp_path/'c.tsv'; truth=tmp_path/'g.tsv'; out=tmp_path/'o.json'
 cand.write_text('entity_id\tcandidate_ids\nq1\ta,b\nq2\t\nq3\tx\n',encoding='utf8')
 truth.write_text('source1_entity_id\tmatched_entity_ids\nq1\ta\nq2\t\nq3\ty\n',encoding='utf8')
 m=evaluate(str(cand),str(truth),str(out)); assert m['rows']==3; assert m['full_match_coverage_pct']==pytest.approx(66.66666666666666); assert m['empty_ground_truth_pct']==pytest.approx(33.33333333333333)

def test_small_end_to_end(tmp_path, monkeypatch):
 import subprocess, sys
 data=tmp_path/'data'; data.mkdir()
 hdr='entity_id\tbusiness_name\tbusiness_address\tcountry\n'
 (data/'source2.tsv').write_text(hdr+'r1\tAcme\t1 Main\tIN\n',encoding='utf8')
 (data/'source3.tsv').write_text(hdr+'r2\t東京商店\t2 Tokyo\tJP\n',encoding='utf8')
 (data/'source1.tsv').write_text(hdr+'q1\tAcme\t1 Main\tIN\nq2\t未知\t\t\n',encoding='utf8')
 (data/'ground_truth.tsv').write_text('source1_entity_id\tmatched_entity_ids\nq1\tr1\nq2\t\n',encoding='utf8')
 c=PipelineConfig(source1_path=str(data/'source1.tsv'),source2_path=str(data/'source2.tsv'),source3_path=str(data/'source3.tsv'),ground_truth_path=str(data/'ground_truth.tsv'),index_dir=str(tmp_path/'idx'),candidates_path=str(tmp_path/'c.tsv'),evaluation_path=str(tmp_path/'e.json'),backend='hash',embedding_dim=16,top_k=50)
 from entity_resolution.scripts_stub import build_indexes_for_test, generate_for_test
 build_indexes_for_test(c); generate_for_test(c); assert sum(1 for _ in open(c.candidates_path,encoding='utf8'))==3
