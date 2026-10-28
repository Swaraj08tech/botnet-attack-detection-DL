"""Download the official public sources for the Notebook 18 knowledge base (run once).

Sources (all public, official publishers):
- MITRE ATT&CK Enterprise (STIX 2.1 JSON) from the mitre/cti GitHub repository
- NIST SP 800-61 Rev. 3: Incident Response Recommendations and Considerations for Cybersecurity Risk Management (2025)
- NIST SP 800-94: Guide to Intrusion Detection and Prevention Systems (IDPS)
- CISA: Federal Government Cybersecurity Incident and Vulnerability Response Playbooks

Files go to data/knowledge_base/<source>/ with a SHA-256 manifest. Re-running skips files that already exist.
Usage:  .venv/Scripts/python.exe scripts/fetch_knowledge_base.py
"""
import datetime
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / 'data/knowledge_base'
SOURCES = {
    'mitre/enterprise-attack.json': 'https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json',
    'nist/NIST.SP.800-61r3.pdf': 'https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.pdf',
    'nist/NIST.SP.800-94.pdf': 'https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-94.pdf',
    'cisa/CISA_Incident_and_Vulnerability_Response_Playbooks.pdf':
        'https://www.cisa.gov/sites/default/files/publications/Federal_Government_Cybersecurity_Incident_and_Vulnerability_Response_Playbooks_508C.pdf',
}


def main():
    manifest = {}
    for name, url in SOURCES.items():
        path = KB / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            response = requests.get(url, timeout=300, headers={'User-Agent': 'Mozilla/5.0 (research project)'})
            response.raise_for_status()
            path.write_bytes(response.content)
        manifest[name] = {'url': url, 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"{name}: {manifest[name]['bytes']:,} bytes")
    (KB / 'MANIFEST.json').write_text(json.dumps({'downloaded': datetime.date.today().isoformat(), 'files': manifest}, indent=2))


if __name__ == '__main__':
    main()
