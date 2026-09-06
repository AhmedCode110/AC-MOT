import gzip,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from core import Config,Controller,candidates,boxes,atomic_json,sha
from experiment import make_tracker,track,replay,environment

class Tests(unittest.TestCase):
    def test_all_candidates_valid(self):
        names=[]
        for c in candidates():names.append(Config(**c).validate().name)
        self.assertEqual(len(names),len(set(names)))

    def test_stability_and_reset(self):
        c=Controller(Config('test',policy='adaptive',stable=True))
        v=dict(brightness=100,blur=300,edges=.14)
        det=[[i*3,0,i*3+2,2,.8,0] for i in range(40)]
        self.assertEqual(c.choose(1,v,det)['size'],640)
        self.assertEqual(c.choose(11,v,det)['size'],640)
        self.assertEqual(c.choose(31,v,det)['size'],832)
        self.assertEqual(Controller(Config('test')).choose(1,v,[])['size'],640)

    def test_recovery_retains_id(self):
        c=Config('test',recovery=True)
        tracker=make_tracker(c);p=dict(conf=.1,high=.25,new=.25)
        first,_=track(tracker,[[0,0,20,20,.9,0]],(100,100),p)
        weak,_=track(tracker,[[1,0,21,20,.15,0]],(100,100),p)
        self.assertEqual(first[0,4],weak[0,4])

    def test_empty_frame(self):
        t,_=track(make_tracker(Config('test')),[],(100,100),dict(conf=.1,high=.25,new=.25))
        self.assertEqual(t.shape,(0,8))

    def test_bad_boxes(self):
        with self.assertRaises(ValueError):boxes([[5,5,1,1,.9,0]])

    def test_cache_replay_end_to_end_and_guard(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);cache=root/'cache';cache.mkdir()
            cfg=dict(fingerprint='synthetic',environment=environment(),weights_sha256='fixture',
                split='development',dataset='fixture',manifest=[dict(sequence='uav001',frames=3)])
            atomic_json(cache/'cache.json',cfg)
            src=cache/'uav001.jsonl.gz'
            with gzip.open(src,'wt') as f:
                for frame in range(1,4):
                    f.write(json.dumps(dict(frame=frame,shape=[100,100],visual=dict(brightness=100,blur=300,edges=.1),
                        bank={'640_0.45':[[frame,0,frame+20,20,.9,0]]}))+'\n')
            atomic_json(cache/'uav001.complete.json',dict(sha256=sha(src)))
            config=root/'systems.json';atomic_json(config,[dict(name='test')])
            args=SimpleNamespace(cache=cache,output=root/'run',config=config,frozen=None)
            replay(args)
            with gzip.open(root/'run/test/uav001.frames.jsonl.gz','rt') as f:rows=[json.loads(s) for s in f]
            self.assertEqual(len(rows),3);self.assertEqual(rows[0]['ids'],rows[-1]['ids'])
            replay(args) # exact resume
            cfg['split']='test';atomic_json(cache/'cache.json',cfg)
            with self.assertRaises(ValueError):replay(args)

if __name__=='__main__':unittest.main()
