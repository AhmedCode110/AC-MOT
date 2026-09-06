import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess
spec=importlib.util.spec_from_file_location('runner',Path(__file__).parent/'scripts/run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class PortabilityTests(unittest.TestCase):
    def test_gpu_never_falls_back(self):
        with patch('torch.cuda.is_available',return_value=False),patch.object(runner.md,'version',return_value='8.3.200'):
            with self.assertRaisesRegex(RuntimeError,'CUDA required'):runner.environment('0',True,True)
    def test_non_t4_rejected(self):
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.get_device_name',return_value='A100'),patch.object(runner.md,'version',return_value='8.3.200'):
            with self.assertRaisesRegex(RuntimeError,'Tesla T4'):runner.environment('0',True,True)
    def test_unique_output_and_failure_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'dataset/annotations').mkdir(parents=True)
            cfg=dict(mode='evaluate_saved',dataset=str(root/'dataset'),output_root=str(root/'results'),saved_run='fixture',trackeval='fixture')
            def fake_run(*args,**kwargs):
                command=args[0]
                if 'evaluate.py' in command[1]:
                    out=Path(command[-1]);out.mkdir(exist_ok=False);(out/'preserved.txt').write_text('original')
            with patch.object(runner,'environment',return_value={}),patch.object(runner.subprocess,'check_output',return_value='fixture'),patch.object(runner.subprocess,'run',side_effect=fake_run):
                a=runner.execute(cfg);b=runner.execute(cfg)
            self.assertNotEqual(a,b)
            self.assertEqual((a/'result/preserved.txt').read_text(),'original')
            self.assertEqual(json.loads((a/'run_metadata.json').read_text())['status'],'completed')
            def fail(command,**kwargs):
                if 'evaluate.py' in command[1]:raise subprocess.CalledProcessError(1,command)
            with patch.object(runner,'environment',return_value={}),patch.object(runner.subprocess,'check_output',return_value='fixture'),patch.object(runner.subprocess,'run',side_effect=fail):
                with self.assertRaises(subprocess.CalledProcessError):runner.execute(cfg)
            receipts=[json.loads(p.read_text()) for p in (root/'results').glob('*/run_metadata.json')]
            self.assertEqual(sum(r['status']=='failed' for r in receipts),1)
    def test_notebook_credentials_not_in_url_or_file(self):
        nb=json.loads((Path(__file__).parent/'notebooks/AC_MOT_Colab.ipynb').read_text())
        source='\n'.join(''.join(c['source']) for c in nb['cells'])
        self.assertIn('getpass.getpass',source)
        self.assertIn("env.pop('ACMOT_GH_TOKEN',None)",source)
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                self.assertEqual(cell['outputs'],[])
                compile(''.join(cell['source']),'runner notebook','exec')
if __name__=='__main__':unittest.main()
