"""Synthetic CPU Conv1D runtime check only; no project data or model training."""
from pathlib import Path
import datetime
import importlib.metadata
import json
import platform
import sys
import tempfile
import traceback

ROOT=Path(__file__).resolve().parents[1]

def main():
    result=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        python=sys.executable,python_version=platform.python_version(),
        purpose='Synthetic runtime smoke test; no project records/targets accessed',
        torch_distribution=importlib.metadata.version('torch'),security_policy_changed=False)
    try:
        import torch
        torch.set_num_threads(2); torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        model=torch.nn.Sequential(torch.nn.Conv1d(1,4,3),torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),torch.nn.Flatten(),torch.nn.Linear(4,10)).double()
        x=torch.linspace(-1,1,2*195,dtype=torch.float64).reshape(2,1,195)
        y=torch.tensor([0,6]); optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        loss=torch.nn.CrossEntropyLoss()(model(x),y); loss.backward(); optimizer.step()
        assert torch.isfinite(loss) and all(torch.isfinite(p).all() for p in model.parameters())
        model.eval()
        with torch.no_grad(): expected=model(x)
        cache=ROOT/'.cache/neural_reference_review_v2';cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            path=Path(tmp)/'synthetic_state.pt';torch.save(model.state_dict(),path)
            other=torch.nn.Sequential(torch.nn.Conv1d(1,4,3),torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1),torch.nn.Flatten(),torch.nn.Linear(4,10)).double()
            other.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));other.eval()
            with torch.no_grad(): actual=other(x)
            assert torch.equal(expected,actual)
        result.update(status='passed',torch_version=torch.__version__,device='CPU',
            cuda_available=torch.cuda.is_available(),input_shape=list(x.shape),output_shape=list(expected.shape),
            dtype=str(x.dtype),forward_backward=True,optimizer_step=True,finite=True,checkpoint_reload_equal=True)
    except Exception:
        result.update(status='unavailable',traceback=traceback.format_exc())
    path=ROOT/'docs/neural_reference_review_v2/07_runtime_check.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':sys.exit(main())
