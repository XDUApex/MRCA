import json
import math
import os
import resource
import time
import uuid
from contextlib import contextmanager
from datetime import datetime

import numpy as np


class RQ2Profiler:
    def __init__(self, dataset, script_name, stage, modality=None, experiment='rq1', variant='base', output_root='rq2_profiles'):
        self.dataset = dataset
        self.script_name = script_name
        self.stage = stage
        self.modality = modality
        self.experiment = experiment
        self.variant = variant
        self.output_root = output_root
        self.run_id = uuid.uuid4().hex[:12]
        self.started_at = datetime.utcnow().isoformat() + 'Z'

        self.size_bytes = 0
        self.n_records = 0
        self.logs_template_count = 0
        self.trace_spans_total = 0
        self.trace_count_total = 0
        self.metric_series = set()
        self.infer_latencies_ms = []
        self.phase_stats = {}
        self._active_phases = {}

    def _current_rss_gb(self):
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss_kb / (1024 ** 2)

    def _ensure_phase(self, phase_name):
        if phase_name not in self.phase_stats:
            self.phase_stats[phase_name] = {
                'wall_time_sec': 0.0,
                'cpu_time_sec': 0.0,
                'peak_rss_gb': 0.0,
            }
        return self.phase_stats[phase_name]

    def start_phase(self, phase_name):
        self._ensure_phase(phase_name)
        self._active_phases[phase_name] = {
            'wall_start': time.perf_counter(),
            'cpu_start': time.process_time(),
        }

    def end_phase(self, phase_name):
        active = self._active_phases.pop(phase_name, None)
        if active is None:
            return
        phase = self._ensure_phase(phase_name)
        phase['wall_time_sec'] += time.perf_counter() - active['wall_start']
        phase['cpu_time_sec'] += time.process_time() - active['cpu_start']
        phase['peak_rss_gb'] = max(phase['peak_rss_gb'], self._current_rss_gb())

    @contextmanager
    def phase(self, phase_name):
        self.start_phase(phase_name)
        try:
            yield self
        finally:
            self.end_phase(phase_name)

    def add_input_file(self, file_path, record_count=0):
        if os.path.exists(file_path):
            self.size_bytes += os.path.getsize(file_path)
        self.n_records += int(record_count or 0)

    def add_records(self, record_count):
        self.n_records += int(record_count or 0)

    def add_size_bytes(self, size_bytes):
        self.size_bytes += int(size_bytes or 0)

    def add_log_template_count(self, count):
        self.logs_template_count += int(count or 0)

    def observe_trace_dataframe(self, df):
        trace_id_col = next((col for col in ['TraceID', 'TraceId', 'trace_id', 'traceId'] if col in df.columns), None)
        if trace_id_col is None:
            return
        counts = df.groupby(trace_id_col).size()
        self.trace_spans_total += int(counts.sum())
        self.trace_count_total += int(counts.shape[0])

    def add_metric_series(self, service_name, metric_names):
        for metric_name in metric_names:
            self.metric_series.add(f'{service_name}:{metric_name}')

    def record_infer_latency(self, latency_ms):
        if latency_ms is None:
            return
        value = float(latency_ms)
        if math.isfinite(value):
            self.infer_latencies_ms.append(value)

    def to_dict(self):
        total_cpu_time = sum(phase['cpu_time_sec'] for phase in self.phase_stats.values())
        peak_rss = max((phase['peak_rss_gb'] for phase in self.phase_stats.values()), default=self._current_rss_gb())
        infer_p50 = float(np.percentile(self.infer_latencies_ms, 50)) if self.infer_latencies_ms else 0.0
        infer_p95 = float(np.percentile(self.infer_latencies_ms, 95)) if self.infer_latencies_ms else 0.0
        preprocess_time = self.phase_stats.get('preprocess', {}).get('wall_time_sec', 0.0)
        train_time = self.phase_stats.get('train', {}).get('wall_time_sec', 0.0)
        return {
            'run_id': self.run_id,
            'started_at': self.started_at,
            'finished_at': datetime.utcnow().isoformat() + 'Z',
            'dataset': self.dataset,
            'script_name': self.script_name,
            'stage': self.stage,
            'modality': self.modality,
            'experiment': self.experiment,
            'variant': self.variant,
            'size_gb': self.size_bytes / (1024 ** 3),
            'n_records': int(self.n_records),
            't_preprocess_sec': float(preprocess_time),
            't_train_sec': float(train_time),
            't_infer_p50_ms': infer_p50,
            't_infer_p95_ms': infer_p95,
            'peak_rss_gb': float(peak_rss),
            'cpu_time_sec': float(total_cpu_time),
            'metrics_active_series': len(self.metric_series),
            'logs_template_count': int(self.logs_template_count),
            'traces_avg_spans_per_trace': (
                float(self.trace_spans_total / self.trace_count_total) if self.trace_count_total else 0.0
            ),
            'phase_stats': self.phase_stats,
            'infer_latency_samples_ms': self.infer_latencies_ms,
        }

    def write_json(self):
        artifact_dir = os.path.join(self.output_root, self.dataset, self.script_name)
        os.makedirs(artifact_dir, exist_ok=True)
        file_name = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{self.run_id}.json"
        output_path = os.path.join(artifact_dir, file_name)
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)
        return output_path
