from __future__ import annotations

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from scapy.all import IP, TCP, UDP, PcapReader
from tqdm import tqdm

warnings.filterwarnings("ignore")

LABEL_MAPPING = {
    "vpn_facebook": "Facebook",
    "vpn_skype": "Skype",
    "vpn_hangouts": "Hangouts",
    "vpn_aim": "AIM",
    "vpn_bittorrent": "BitTorrent",
    "vpn_email": "Email",
    "vpn_ftps": "FTPS",
    "vpn_icq": "ICQ",
    "vpn_netflix": "Netflix",
    "vpn_sftp": "SFTP",
    "vpn_spotify": "Spotify",
    "vpn_vimeo": "Vimeo",
    "vpn_voipbuster": "VoIPBuster",
    "vpn_youtube": "Youtube",
    "gmail": "Gmail",
    "scp": "SCP",
}


def get_label_from_filename(filename: str) -> str | None:
    lower = filename.lower()
    for key, label in LABEL_MAPPING.items():
        if key in lower:
            return label
    return None


def get_stats(arr, prefix: str):
    if len(arr) == 0:
        return {f"{prefix}_{k}": 0.0 for k in ["mean", "std", "max", "min", "sum"]}
    arr = np.asarray(arr, dtype=np.float32)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_sum": float(np.sum(arr)),
    }


def extract_features(pcap_root: str, seq_len: int = 50) -> pd.DataFrame:
    all_rows = []
    pcap_files = []
    for ext in ("*.pcap", "*.pcapng", "*.cap"):
        pcap_files.extend(glob.glob(os.path.join(pcap_root, ext)))

    print(f"Scanning {len(pcap_files)} capture files in {pcap_root}")
    for pcap_path in tqdm(sorted(pcap_files), desc="Extracting"):
        filename = os.path.basename(pcap_path)
        label = get_label_from_filename(filename)
        if label is None:
            continue

        flows = {}
        try:
            with PcapReader(pcap_path) as packets:
                for pkt in packets:
                    if not pkt.haslayer(IP):
                        continue
                    ip = pkt[IP]
                    proto = int(ip.proto)
                    if proto not in (6, 17):
                        continue

                    if ip.src < ip.dst:
                        src, dst = ip.src, ip.dst
                        direction = 1
                    else:
                        src, dst = ip.dst, ip.src
                        direction = -1

                    sport = 0
                    dport = 0
                    window = 0
                    flags = 0
                    is_tcp = False

                    if pkt.haslayer(TCP):
                        tcp = pkt[TCP]
                        sport = int(tcp.sport)
                        dport = int(tcp.dport)
                        window = int(tcp.window)
                        flags = int(tcp.flags)
                        is_tcp = True
                    elif pkt.haslayer(UDP):
                        udp = pkt[UDP]
                        sport = int(udp.sport)
                        dport = int(udp.dport)

                    if ip.src < ip.dst:
                        flow_key = (src, dst, sport, dport, proto)
                    else:
                        flow_key = (src, dst, dport, sport, proto)

                    if flow_key not in flows:
                        flows[flow_key] = {
                            "pkts": [],
                            "all_lens": [],
                            "all_iats": [],
                            "ts": [],
                            "last_ts": float(pkt.time),
                            "ttls": [],
                            "wins": [],
                            "flags": [],
                        }

                    flow = flows[flow_key]
                    pkt_len = int(ip.len)
                    iat = (float(pkt.time) - flow["last_ts"]) * 1000.0
                    flow["all_lens"].append(pkt_len)
                    flow["all_iats"].append(iat)

                    if len(flow["pkts"]) < seq_len:
                        flow["pkts"].append({"len": pkt_len, "dir": direction, "iat": iat})

                    flow["ts"].append(float(pkt.time))
                    flow["last_ts"] = float(pkt.time)
                    flow["ttls"].append(int(ip.ttl))
                    if is_tcp:
                        flow["wins"].append(window)
                        flow["flags"].append(flags)
        except Exception as exc:
            print(f"Skip {filename}: {exc}")
            continue

        for key, flow_data in flows.items():
            if len(flow_data["pkts"]) < 3:
                continue

            seq_lens = [p["len"] for p in flow_data["pkts"]]
            seq_iats = [p["iat"] for p in flow_data["pkts"]]
            dirs = [p["dir"] for p in flow_data["pkts"]]
            all_lens = flow_data["all_lens"]
            all_iats = flow_data["all_iats"]
            duration = max(flow_data["ts"][-1] - flow_data["ts"][0], 1e-6)

            row = {
                "label": label,
                "protocol": int(key[4]),
                "duration": float(duration * 1000.0),
                "pkt_cnt": int(len(flow_data["ts"])),
                "byte_cnt": int(sum(all_lens)),
                "pps": float(len(flow_data["ts"]) / duration),
                "bps": float(sum(all_lens) / duration),
            }
            row.update(get_stats(all_lens, "len"))
            row.update(get_stats(all_iats, "iat"))
            row.update(get_stats(flow_data["ttls"], "ttl"))

            if flow_data["wins"]:
                row.update(get_stats(flow_data["wins"], "win"))
            else:
                row.update({f"win_{k}": 0.0 for k in ["mean", "std", "max", "min", "sum"]})

            row["flag_syn_cnt"] = int(sum(1 for f in flow_data["flags"] if f & 0x02))
            row["flag_fin_cnt"] = int(sum(1 for f in flow_data["flags"] if f & 0x01))
            row["flag_rst_cnt"] = int(sum(1 for f in flow_data["flags"] if f & 0x04))
            row["flag_psh_cnt"] = int(sum(1 for f in flow_data["flags"] if f & 0x08))

            for i in range(seq_len):
                if i < len(seq_lens):
                    row[f"seq_len_{i + 1}"] = seq_lens[i]
                    row[f"seq_dir_{i + 1}"] = dirs[i]
                    row[f"seq_iat_{i + 1}"] = seq_iats[i]
                else:
                    row[f"seq_len_{i + 1}"] = 0
                    row[f"seq_dir_{i + 1}"] = 0
                    row[f"seq_iat_{i + 1}"] = 0

            fft_input = seq_iats + [0.0] * max(0, seq_len - len(seq_iats))
            fft_vals = np.abs(np.fft.fft(fft_input))
            for f in range(1, 4):
                row[f"fft_{f}"] = float(fft_vals[f]) if len(fft_vals) > f else 0.0

            all_rows.append(row)

    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description="Extract shortcut-reduced ISCX-VPN features from PCAP files.")
    parser.add_argument("--pcap-root", required=True, help="Directory containing PCAP/PCAPNG/CAP files.")
    parser.add_argument("--output", default="ISCX_VPN_16Class_Task3_Final.csv", help="Output CSV path.")
    parser.add_argument("--seq-len", type=int, default=50, help="Maximum sequence length per flow.")
    args = parser.parse_args()

    df = extract_features(args.pcap_root, seq_len=int(args.seq_len))
    if df.empty:
        print("No valid flows were extracted.")
        return

    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} flows to {args.output}")
    print(f"Feature dimension: {df.shape[1] - 1}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
