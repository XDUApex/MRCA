# Paper
MRCA: Metric-level Root Cause Analysis for Microservices via
Multi-Modal Data

### Operating environment
* PyTorch version: 1.13.1

### Run command
Feature learning for trace and log.
```
python -m MRCA.trace_processing --dataset gaia|aiops|ob|tt
python -m MRCA.log_processing --dataset gaia|aiops|ob|tt
python -m MRCA.data_aggregation --dataset gaia|aiops|ob|tt
python -m MRCA.data_conversion --dataset gaia|aiops|ob|tt
```
Anomaly detection.
```
python -m MRCA.anomaly_detection --dataset gaia|aiops|ob|tt
```
Root cause localization
```
python MRCA/root_cause_localization.py
```
Evaluation MRCA
```
python evaluation_at_log-level.py --dataset gaia|aiops|ob|tt
python evaluation_at_service-level.py
python evaluation_at_metric-level.py
```