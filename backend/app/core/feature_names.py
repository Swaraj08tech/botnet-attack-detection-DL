"""Short, plain-English names for the 42 UNSW-NB15 flow features and the two engineered byte features.

The official one-line definitions come from NUSW-NB15_features.csv (serving bundle: feature_descriptions.json); these names only make
them easier to scan. Engineered features are defined by the formulas used in preprocessing (Notebook 04 / app.core.detector).
"""

FRIENDLY = {
    'dur': 'Flow duration', 'proto': 'Transport protocol', 'service': 'Application service', 'state': 'Connection state',
    'spkts': 'Packets from source', 'dpkts': 'Packets from destination', 'sbytes': 'Bytes from source', 'dbytes': 'Bytes from destination',
    'rate': 'Flow rate', 'sttl': 'Source time-to-live (TTL)', 'dttl': 'Destination time-to-live (TTL)', 'sload': 'Source bits per second',
    'dload': 'Destination bits per second', 'sloss': 'Source packets retransmitted or dropped', 'dloss': 'Destination packets retransmitted or dropped',
    'sinpkt': 'Source inter-packet time', 'dinpkt': 'Destination inter-packet time', 'sjit': 'Source jitter', 'djit': 'Destination jitter',
    'swin': 'Source TCP window', 'stcpb': 'Source TCP base sequence number', 'dtcpb': 'Destination TCP base sequence number',
    'dwin': 'Destination TCP window', 'tcprtt': 'TCP round-trip setup time', 'synack': 'TCP SYN to SYN-ACK time', 'ackdat': 'TCP SYN-ACK to ACK time',
    'smean': 'Mean source packet size', 'dmean': 'Mean destination packet size', 'trans_depth': 'HTTP request/response depth',
    'response_body_len': 'HTTP response body size', 'ct_srv_src': 'Recent connections: same service and source',
    'ct_state_ttl': 'Recent connections: same state and TTL range', 'ct_dst_ltm': 'Recent connections: same destination',
    'ct_src_dport_ltm': 'Recent connections: same source and destination port', 'ct_dst_sport_ltm': 'Recent connections: same destination and source port',
    'ct_dst_src_ltm': 'Recent connections: same source and destination', 'is_ftp_login': 'FTP login with user and password',
    'ct_ftp_cmd': 'FTP commands in the session', 'ct_flw_http_mthd': 'HTTP GET/POST methods in the session', 'ct_src_ltm': 'Recent connections: same source',
    'ct_srv_dst': 'Recent connections: same service and destination', 'is_sm_ips_ports': 'Same source and destination address and port',
    'fe_total_bytes': 'Total bytes (engineered)', 'fe_byte_balance': 'Byte balance (engineered)',
}

ENGINEERED = {
    'fe_total_bytes': 'Engineered feature: bytes from source + bytes from destination (sbytes + dbytes).',
    'fe_byte_balance': 'Engineered feature: (sbytes - dbytes) / (sbytes + dbytes), from -1 (all bytes from the destination) to 1 (all from the '
                       'source); 0 when no bytes were sent.',
}


def friendly(feature: str) -> str:
    return FRIENDLY.get(feature, feature)


def definition(feature: str, descriptions: dict) -> str:
    return ENGINEERED.get(feature) or descriptions.get(feature, '')

# Phrases for sentences in the Summary report (lower case, no feature codes). Falls back to the friendly name.
PLAIN = {
    'sttl': 'time-to-live (TTL) value set by the sending computer',
    'dttl': 'time-to-live value set by the responding computer (TTL)',
    'service': 'network service used (for example HTTP, FTP or DNS)', 'proto': 'transport protocol', 'state': 'connection state',
    'sbytes': 'number of bytes sent by the source', 'dbytes': 'number of bytes sent back', 'fe_total_bytes': 'total bytes in the flow',
    'fe_byte_balance': 'balance between bytes sent and bytes received', 'smean': 'average packet size sent by the source',
    'dmean': 'average packet size sent back', 'spkts': 'number of packets sent by the source', 'dpkts': 'number of packets sent back',
    'dur': 'flow duration', 'rate': 'packet rate',
    'ct_srv_dst': 'number of recent connections to the same service and destination',
    'ct_srv_src': 'number of recent connections from the same source to the same service',
    'ct_dst_ltm': 'number of recent connections to the same destination', 'ct_src_ltm': 'number of recent connections from the same source',
    'ct_dst_src_ltm': 'number of recent connections between the same source and destination',
    'ct_state_ttl': 'combination of connection state and TTL range', 'sload': 'data rate from the source', 'dload': 'data rate back to the source',
}


def plain(feature: str) -> str:
    return PLAIN.get(feature) or friendly(feature)[:1].lower() + friendly(feature)[1:]
