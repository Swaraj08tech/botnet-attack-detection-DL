# Runtime readiness for Notebook 08

## What happened

Notebook 07 originally recorded PyTorch `WinError 4551`: an Application Control policy blocked `torch.dll` or a dependency. That historical record is preserved in `docs/neural_reference_v1/07_environment.json`. The scikit-learn MLP ran successfully despite that environment limitation.

During this review, the **same project `.venv` Python 3.11.9 imported PyTorch 2.14.0+cpu successfully**. A synthetic CPU test passed Conv1D forward/backward computation, one Adam step, finite-value checks, and exact save/reload equality using `weights_only=True`. The input was `[2, 1, 195]`, float64, with ten outputs. It used invented numbers, not project records. See [the measured check](07_runtime_check.json) and [the smoke-test script](../../scripts/check_neural_runtime.py).

CUDA was unavailable in this CPU build. The project `.venv` inherits system site-packages; environment state can change outside the project. We made no package installation, security exception or policy change in this review. The reason import now succeeds is not established.

## What we checked about the old error

The referenced DLL exists. Windows Authenticode reports it as `NotSigned`; this alone does not establish why a particular DLL load was blocked. A restricted Code Integrity query was denied in the sandbox. An authorized read of the latest 120 events over three days returned no matching `torch` events. That limited query cannot establish which policy or dependency caused the earlier failure.

Microsoft documents using the **CodeIntegrity/Operational** log, especially correlated 3077 block and 3089 signature events, to investigate blocked DLLs. If the problem recurs, record its timestamp, exact Python executable and traceback, and ask the machine's security administrator to inspect those events and review an approved installation. Do not disable Smart App Control/App Control, add antivirus exclusions, unblock or relocate binaries to evade policy, or alter execution policy. [Microsoft troubleshooting guide](https://learn.microsoft.com/en-us/windows/security/application-security/application-control/app-control-for-business/operations/appcontrol-debugging-and-troubleshooting), [event meanings](https://learn.microsoft.com/en-us/windows/security/application-security/application-control/app-control-for-business/operations/event-id-explanations).

## The next action

Before implementing Notebook 08, run this in the **exact interpreter that will execute it**:

```powershell
.venv\Scripts\python.exe scripts/check_neural_runtime.py
```

The current legitimate starting point is the verified **local CPU PyTorch runtime**, with explicit float64 support for the existing inputs. The smoke test checks runtime plumbing, not Conv1D accuracy or performance on the dataset. Freeze its versions and device; then implement the predeclared tabular adaptation under the existing frozen protocol. Any dtype or input-order change needs explicit alignment and collision checks before fitting.

If import is blocked again, use an administrator-approved installation/runtime or a separately provisioned supported Linux environment, selected using [PyTorch's official installation guide](https://pytorch.org/get-started/locally/). WSL is **not currently installed** here; it is not a verified fallback. An approved Linux host/VM or WSL installation would need setup and the same smoke test before use. Do not promise it works, transfer project records, install another runtime, or weaken security as part of this review.

This environment investigation neither changes the MLP results nor establishes a DNS, temporal or botnet-specific capability.
